#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemma-4-E4B を FlashAttention + QLoRA でファインチューンする。
埋め込み層 (262k語彙 x d ≈ 671Mパラメータ, bf16で約1.3GB) はCPUへオフロード。

構成:
  - attn_implementation="flash_attention_2" (flash-attnパッケージ)。
    未インストール/非対応なら "sdpa" に変えると PyTorch 内蔵の
    FlashAttention カーネルが自動選択される。
  - 重みは nf4 4bit 量子化 (QLoRA)。LoRA は attention + MLP に刺す。
  - embed_tokens は CPU でルックアップして GPU へ転送 (転送量は微小)。
  - lm_head (タイ済み重み) は重みをCPUに置いたまま、語彙をチャンク分割して
    GPU へストリーミングし、チャンク毎に交差エントロピーを加算。
    ロジット [b, s, 262144] をGPU上に全展開しない。
    各チャンクは checkpoint 化し、逆伝播時に再計算するので
    GPUに留まるのは常に1チャンク分のみ。

事前準備:
  pip install "transformers>=4.55" "peft>=0.15" "bitsandbytes>=0.45" "accelerate" "datasets"
  pip install flash-attn --no-build-isolation   # ビルドが重い。

注意:
  - MODEL_ID は仮置き。gemma-3n-E4B 等に差し替えればそのまま動くはず
    (3nは per-layer embedding 等が増えるがHFが吸収する。オフロード対象は主埋め込み層)。
  - VRAMに余裕があるなら、埋め込みをGPUに戻して Liger Kernel の
    fused CE を使う方が速い。このオフロードは「VRAMがカツカツ」用の逃し道。

開発者のメモ：これAIが作ったんだって、すごいね、流石に怖くなってきたよ。AIがAIを作るって。まるで繁ｓｙ（殴
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================
# 設定
# ============================================================
MODEL_ID    = "google/gemma-4-E4B"   # 手元パス/リビジョンに差し替え
OUTPUT_DIR  = "./qlora-out"
MAX_LEN     = 1024
ATTN_IMPL   = "flash_attention_2"    # flash-attnが入ってなければ "sdpa"
VOCAB_CHUNK = 16384                  # lm_headを何語彙ずつGPUへ流すか

set_seed(42)

# ============================================================
# 1. モデルのロード (4bit QLoRA + FlashAttention)
# ============================================================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,   # 新バージョンで警告が出たら dtype= に改名
    device_map="auto",
    attn_implementation=ATTN_IMPL,
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 量子化モデル用の前処理 (LayerNorm fp32化など)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ============================================================
# 2. 埋め込み層をCPUへオフロード
# ============================================================
# Gemmaは入力埋め込みとlm_headの重みがタイされている (=同一テンソル)。
# Embeddingモジュールを .cpu() すると、タイされている lm_head.weight の
# .data も一緒にCPUへ移る。GPU側には転送ラッパだけを残す。
class CPUOffloadedEmbedding(nn.Module):
    """input_ids をGPU→CPUへ送ってルックアップし、結果をGPUへ戻す。
    重みは凍結 (QLoRAでは学習しない) なので勾配は不要。"""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.inner = emb.cpu()
        self.inner.requires_grad_(False)
        self._dev = None

    @property
    def weight(self):
        # tie_weights() 等から参照されても実体 (CPUテンソル) を返す
        return self.inner.weight

    def forward(self, input_ids):
        self._dev = input_ids.device
        out = self.inner(input_ids.cpu())
        return out.to(self._dev, non_blocking=True)


embed_cpu = CPUOffloadedEmbedding(model.get_input_embeddings())
model.set_input_embeddings(embed_cpu)

# lm_head は自前のストリーミング損失で扱うのでフォワードから外す
VOCAB_WEIGHT = embed_cpu.weight              # CPU上の [V, d] bf16
model.lm_head = nn.Identity()
model.config.tie_word_embeddings = False     # save時に再タイされないように
model.enable_input_require_grads()           # 勾配チェックポイント用 (ラッパにhookが刺さる)

# ============================================================
# 3. 出力層の損失: 重みはCPU、語彙チャンクをGPUへストリーミング
# ============================================================
# 通常はロジット [b, s, 262144] 全体がGPUに乗る (bf16で s=1024 なら約0.5GB、
# fp32化や逆伝播用保存でさらに膨らむ)。ここでは展開せず、語彙を
# VOCAB_CHUNK ずつGPUへ持ってきて部分CEを足し合わせる。
# チャンク毎に checkpoint 化するので、weightチャンクのGPUコピーが
# 計算グラストに保持されることもない (逆伝播時に再転送・再計算)。
class StreamedLMHeadLoss(nn.Module):
    def __init__(self, weight_cpu: torch.Tensor, chunk: int = 16384, softcap=None):
        super().__init__()
        # buffer/register にすると .to() でGPUへ引っ張られるので生参照で持つ
        self._w = weight_cpu          # CPU, [V, d], bf16, 凍結
        self.chunk = chunk
        self.softcap = softcap       # Gemma系の final_logit_softcapping (Noneなら無効)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # hidden: [b, s, d] (GPU=Identity通過後の最終隠れ状態), labels: [b, s]
        flat_labels = labels.reshape(-1)
        keep = flat_labels != -100
        h = hidden.reshape(-1, hidden.shape[-1])[keep]
        t = flat_labels[keep]
        if h.numel() == 0:
            return hidden.new_zeros((), dtype=torch.float32)

        n = h.shape[0]
        V = self._w.shape[0]
        total = h.new_zeros((), dtype=torch.float32)
        for start in range(0, V, self.chunk):
            total = total + self._chunk_ce(h, t, start)
        return total / n

    def _chunk_ce(self, h, t, start):
        w_cpu, size, softcap = self._w, self.chunk, self.softcap

        def run(x):
            w = w_cpu[start:start + size].to(x.device, non_blocking=True)
            logits = x @ w.t()                                  # [n, size] bf16
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            local = t - start
            ce = F.cross_entropy(logits.float(), local.clamp(0, size - 1),
                                 reduction="none")             # fp32 CE (checkpoint内なので解放される)
            valid = (local >= 0) & (local < size)
            return torch.where(valid, ce, torch.zeros_like(ce)).sum()

        # use_reentrant=False: クロージャ内でweightを再転送して再計算させる
        return ckpt(run, h, use_reentrant=False)


softcap = getattr(model.config, "final_logit_softcapping", None)
lm_loss = StreamedLMHeadLoss(VOCAB_WEIGHT, chunk=VOCAB_CHUNK, softcap=softcap)

# ============================================================
# 4. LoRA
# ============================================================
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ============================================================
# 5. データセット (プレースホルダ: 適宜差し替え)
# ============================================================
def build_dataset(tokenizer, max_len):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    ds = ds.map(lambda ex: tokenizer(ex["text"]),
                batched=True, remove_columns=ds.column_names, num_proc=4)

    # 単純に連結して MAX_LEN ブロック化。
    # 指示チューニングをするならここを会話テンプレート組む処理に置き換える。
    def group(exs):
        ids = []
        for x in exs["input_ids"]:
            ids.extend(x)
        blocks = [ids[i:i + max_len] for i in range(0, len(ids) - max_len, max_len)]
        return {"input_ids": blocks, "labels": [list(b) for b in blocks]}

    return ds.map(group, batched=True, batch_size=1000, num_proc=4,
                  remove_columns=ds.column_names)


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids, lab, pad = f["input_ids"], f["labels"], maxlen - len(f["input_ids"])
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


train_ds = build_dataset(tokenizer, MAX_LEN)

# ============================================================
# 6. Trainer (lm_headを外しているので損失は自前計算)
# ============================================================
class OffloadTrainer(Trainer):
    def __init__(self, *args, lm_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False)
        hidden = out.logits        # lm_headがIdentityなので実体は最終隠れ状態
        loss = self.lm_loss(hidden, labels)
        return (loss, out) if return_outputs else loss


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
)

trainer = OffloadTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=PadCollator(tokenizer.pad_token_id),
    lm_loss=lm_loss,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/adapter")   # LoRA部分のみ保存
