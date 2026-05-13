# GLiNER2 Fine-Tuning on FiNER-139

Fine-tune a [GLiNER2](https://huggingface.co/fastino/gliner2-base-v1) model with a LoRA adapter on the [FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) financial NER dataset, then evaluate it against the base model on both the held-out test set and hand-written sentences.

---

## Overview

[FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) is a financial named-entity recognition dataset with 139 entity types mapped to XBRL tags (e.g. `NetIncomeLoss`, `EarningsPerShareDiluted`, `CashAndCashEquivalentsAtCarryingValue`). The goal is to extract monetary values from financial text and tag them with the correct XBRL label.

This repo contains:

| File | Purpose |
|---|---|
| `load_dataset.sh` | Clone and unzip FiNER-139 from Hugging Face |
| `training.py` | Stratified sampling, LoRA fine-tuning, per-label F1 evaluation |
| `evaluate.py` | Full test-set comparison: base model vs. fine-tuned adapter |
| `evaluate_custom.py` | Qualitative evaluation on 20 hand-written financial sentences |

---

## Quickstart

### 1. Load the dataset

```bash
bash load_dataset.sh
```

This clones `nlpaueb/finer-139` from Hugging Face and unzips the contents into `./finer-139/`. You should end up with `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

### 2. Fine-tune the model

```bash
python training.py
```

This will:

- Load `train.jsonl` and `validation.jsonl`
- Stratified-sample up to 75 examples per label (min 20, hard cap 10,000 total)
- Convert BIO-tagged tokens into `InputExample` objects
- Train a LoRA adapter (`r=8`, `alpha=16`) on top of `fastino/gliner2-base-v1` for up to 10 epochs with early stopping (patience = 3)
- Save the best checkpoint to `./finer139_adapter_v2/best/`
- Print a per-label F1 table and save `weak_labels.json` for labels with F1 < 0.5

**Requirements:** A CUDA-capable GPU is expected. The script will print your GPU name and VRAM at startup.

### 3. Evaluate on the test set

```bash
python evaluate.py
```

Loads `./finer-139/test.jsonl` and scores both the base model and the fine-tuned adapter (loaded from `./finer139_adapter/best/`) on the first 5,000 examples. Prints a per-label table with Precision, Recall, F1, and delta between base and fine-tuned, plus a Macro F1 summary.

> **Note:** Update the adapter path in `evaluate.py` to `./finer139_adapter_v2/best/` if you trained with the current `training.py`.

### 4. Qualitative evaluation

```bash
python evaluate_custom.py
```

Runs both models on 20 hand-written financial sentences with manually defined gold spans (e.g. `"Tesla reported total revenues of $24.3 billion"` → `{"Revenues": ["24.3"]}`). Reports per-test pass/fail and an overall Precision/Recall/F1 summary.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `fastino/gliner2-base-v1` |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Epochs | 10 (with early stopping, patience=3) |
| Batch size | 16 |
| Encoder LR | 1e-5 |
| Task LR | 5e-4 |
| Scheduler | Cosine with 10% warmup |
| Max sequence length | 256 tokens |
| Min examples per label | 20 |
| Max examples per label | 75 |
| Total training cap | 10,000 |
| Mixed precision | FP16 |

---

## Span Matching

Both evaluation scripts use **partial span matching**: a prediction is counted as correct if the predicted text contains the gold span or vice versa. This means `"$29.9 billion"` correctly matches the gold span `"29.9"`. False positives are only counted when a prediction shares no overlap with any gold span.

---

## Outputs

| Path | Description |
|---|---|
| `./finer139_adapter_v2/best/` | Best LoRA adapter checkpoint (saved by `training.py`) |
| `weak_labels.json` | Labels with F1 < 0.5 on the validation set, for targeted resampling |

---

## Tips

- **Improving weak labels:** After training, check `weak_labels.json`. Re-run `training.py` with `MAX_PER_LABEL = 150` to upsample those labels.
- **Adapter path mismatch:** `evaluate.py` and `evaluate_custom.py` load from `./finer139_adapter/best/` (the v1 path). If you used `training.py` as-is, change this to `./finer139_adapter_v2/best/`.
- **CPU-only:** Remove the `fp16=True` flag from `TrainingConfig` and expect significantly longer training times.
