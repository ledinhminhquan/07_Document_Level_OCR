"""Fine-tune the ByT5/T5 post-OCR corrector (HF Seq2SeqTrainer).

Resume-safe (``get_last_checkpoint``), bf16/tf32 on H100/A100, char-level ByT5 by
default. ``compute_metrics`` reports CER + exact-match; the headline CER-reduction
(vs raw OCR) is computed by ``evaluate``. Heavy imports are lazy.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import model_registry as reg
from ..models.text_utils import normalize_ws
from ..data.dataset import corpus_signature, load_or_build, to_hf_datasets
from . import metrics as M

logger = get_logger(__name__)


def _make_compute_metrics(tokenizer):
    import numpy as np

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        dp = [normalize_ws(s) for s in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        dl = [normalize_ws(s) for s in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {"cer": M.corpus_cer(dp, dl), "exact_match": M.exact_match(dp, dl)}

    return compute_metrics


def train_corrector(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
                    base_model: Optional[str] = None) -> Dict:
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    mc = cfg.model
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    torch.backends.cudnn.allow_tf32 = bool(mc.tf32)

    splits = load_or_build(cfg)
    if limit:
        for k in ("train", "val"):
            splits[k] = splits[k][:limit]
    ds = to_hf_datasets(splits)
    logger.info("Training %s on %d examples (val %d)", model_id, len(ds["train"]), len(ds.get("val", [])))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if hasattr(model.config, "dropout_rate"):
        model.config.dropout_rate = mc.dropout_rate
    prefix = mc.task_prefix

    def preprocess(batch):
        model_inputs = tokenizer([prefix + t for t in batch["noisy"]],
                                 max_length=mc.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["clean"], max_length=mc.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names, desc="tokenize")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    out_dir = mc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), overwrite_output_dir=False,
        num_train_epochs=mc.num_train_epochs, learning_rate=mc.learning_rate,
        per_device_train_batch_size=mc.per_device_train_batch_size,
        per_device_eval_batch_size=mc.per_device_eval_batch_size,
        gradient_accumulation_steps=mc.gradient_accumulation_steps,
        warmup_ratio=mc.warmup_ratio, weight_decay=mc.weight_decay,
        label_smoothing_factor=mc.label_smoothing_factor, max_grad_norm=mc.max_grad_norm,
        lr_scheduler_type="cosine", bf16=bool(mc.bf16), fp16=bool(mc.fp16),
        gradient_checkpointing=bool(mc.gradient_checkpointing), group_by_length=bool(mc.group_by_length),
        predict_with_generate=True, generation_max_length=mc.max_target_length,
        generation_num_beams=mc.generation_num_beams,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=mc.eval_steps, save_steps=mc.save_steps, logging_steps=mc.logging_steps,
        save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="cer", greater_is_better=False,
        seed=mc.seed, report_to=[], dataloader_num_workers=2,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized.get("val"),
        data_collator=collator, tokenizer=tokenizer, compute_metrics=_make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=mc.early_stopping_patience)],
    )
    last = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last:
        logger.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    metrics = trainer.evaluate()
    version = reg.make_version(model_id)
    final_dir = out_dir / version
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature=corpus_signature(cfg.data),
                       metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    logger.info("Training done. val CER=%.4f -> %s", metrics.get("eval_cer", 0.0), final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}


__all__ = ["train_corrector"]
