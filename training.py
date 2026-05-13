import random
import json
import os
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
import torch

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42           # random seed for reproducibility 
MIN_PER_LABEL = 20           # minimum examples per label
MAX_PER_LABEL = 75           # maximum examples sampled per label
HARD_CAP      = 10000        # absolute max training examples after sampling 
MAX_SEQ_LEN   = 256          # max token length per sentence
OUTPUT_DIR    = "./finer139_adapter"   # where the trained adapter is saved
BASE_MODEL    = "fastino/gliner2-base-v1" # GLiNER2 base model
DATA_DIR      = "./finer-139"             # path to your FiNER-139 JSONL files
NUM_WORKERS   = os.cpu_count() - 1       # CPU cores for parallel data loading
random.seed(SEED)


if __name__ == "__main__":
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPUs:    {os.cpu_count()} cores  |  Using {NUM_WORKERS} for data loading")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading FiNER-139 from local JSONL files...")

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    # Load train and val in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_train = executor.submit(load_jsonl, f"{DATA_DIR}/train.jsonl")
        future_val   = executor.submit(load_jsonl, f"{DATA_DIR}/validation.jsonl")
        train_raw    = future_train.result()
        val_raw      = future_val.result()

    # Collect label names using all CPU cores
    def extract_labels_from_chunk(chunk):
        labels = set()
        for ex in chunk:
            for tag in ex["ner_tags"]:
                if tag != "O":
                    labels.add(tag[2:])
        return labels

    chunk_size = len(train_raw) // NUM_WORKERS
    chunks     = [train_raw[i:i+chunk_size] for i in range(0, len(train_raw), chunk_size)]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results    = executor.map(extract_labels_from_chunk, chunks)
        label_names = sorted(set().union(*results))

    print(f"  Train: {len(train_raw):,}  Val: {len(val_raw):,}  Labels: {len(label_names)}")

    # ── 2. Sample ──────────────────────────────────────────────────────────────
    print("\n[2/6] Stratified sampling...")

    # Build label_example_map in parallel
    def map_labels_chunk(chunk):
        local_map = defaultdict(list)
        for ex in chunk:
            seen = set()
            for tag in ex["ner_tags"]:
                if tag.startswith("B-"):
                    seen.add(tag[2:])
            for label in seen:
                local_map[label].append(ex)
        return local_map

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        local_maps = list(executor.map(map_labels_chunk, chunks))

    # Merge local maps
    label_example_map = defaultdict(list)
    for local_map in local_maps:
        for label, exs in local_map.items():
            label_example_map[label].extend(exs)

    excluded     = []
    selected_ids = set()
    sampled      = []

    for label, examples in sorted(label_example_map.items()):
        if len(examples) < MIN_PER_LABEL:
            excluded.append(label)
            continue
        random.shuffle(examples)
        added = 0
        for ex in examples:
            ex_id = id(ex)
            if ex_id not in selected_ids:
                selected_ids.add(ex_id)
                sampled.append(ex)
                added += 1
            if added >= MAX_PER_LABEL:
                break

    # Hard cap at 10k with shuffle to avoid label bias
    random.shuffle(sampled)
    sampled = sampled[:10000]

    print(f"  Sampled {len(sampled):,} examples  |  Excluded {len(excluded)} rare labels")

    # ── 3. Convert ─────────────────────────────────────────────────────────────
    print("\n[3/6] Converting to InputExamples...")

    def bio_to_spans(tokens, ner_tags):
        entities    = defaultdict(list)
        current     = []
        current_lbl = None
        for token, tag in zip(tokens, ner_tags):
            if tag.startswith("B-"):
                if current and current_lbl:
                    entities[current_lbl].append(" ".join(current))
                current, current_lbl = [token], tag[2:]
            elif tag.startswith("I-") and current_lbl == tag[2:]:
                current.append(token)
            else:
                if current and current_lbl:
                    entities[current_lbl].append(" ".join(current))
                current, current_lbl = [], None
        if current and current_lbl:
            entities[current_lbl].append(" ".join(current))
        return dict(entities)

    def convert_chunk(chunk):
        out              = []
        skipped_entities = 0
        skipped_length   = 0
        for ex in chunk:
            if len(ex["tokens"]) > MAX_SEQ_LEN:
                skipped_length += 1
                continue
            text     = " ".join(ex["tokens"])
            entities = bio_to_spans(ex["tokens"], ex["ner_tags"])
            if not entities:
                skipped_entities += 1
                continue
            out.append(InputExample(
                text=text,
                entities=entities,
                entity_descriptions={l: f"Financial entity: {l}" for l in entities}
            ))
        return out, skipped_entities, skipped_length

    def convert_parallel(raw_examples):
        chunk_size = max(1, len(raw_examples) // NUM_WORKERS)
        chunks     = [raw_examples[i:i+chunk_size] for i in range(0, len(raw_examples), chunk_size)]

        all_out              = []
        total_skip_entities  = 0
        total_skip_length    = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for out, se, sl in executor.map(convert_chunk, chunks):
                all_out.extend(out)
                total_skip_entities += se
                total_skip_length   += sl

        print(f"  Converted:             {len(all_out):,}")
        print(f"  Skipped (no entities): {total_skip_entities:,}")
        print(f"  Skipped (too long):    {total_skip_length:,}")
        return all_out

    train_examples = convert_parallel(sampled)
    val_examples   = convert_parallel(val_raw[:1000])

    # ── 4. Validate ────────────────────────────────────────────────────────────
    print("\n[4/6] Validating...")
    TrainingDataset(train_examples).validate()
    print("  Validation passed.")

    # ── 5. Train ───────────────────────────────────────────────────────────────
    print("\n[5/6] Training...")
    model  = GLiNER2.from_pretrained(BASE_MODEL)
    config = TrainingConfig(
        output_dir              = OUTPUT_DIR,
        experiment_name         = "finer139_lora",
        num_epochs              = 10,
        batch_size              = 16,
        encoder_lr              = 1e-5,
        task_lr                 = 5e-4,
        use_lora                = True,
        lora_r                  = 8,
        lora_alpha              = 16,
        lora_dropout            = 0.1,
        save_adapter_only       = True,
        fp16                    = True,
        warmup_ratio            = 0.1,
        scheduler_type          = "cosine",
        eval_strategy           = "epoch",
        save_best               = True,
        early_stopping          = True,
        early_stopping_patience = 3,
    )
    GLiNER2Trainer(model, config).train(
        train_data = TrainingDataset(train_examples),
        eval_data  = val_examples
    )

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating per-label F1...")
    model = GLiNER2.from_pretrained(BASE_MODEL)
    model.load_adapter(f"{OUTPUT_DIR}/best")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Batch evaluation instead of one-by-one
    eval_examples = val_examples[:2000]
    batch_size    = 32

    for i in range(0, len(eval_examples), batch_size):
        batch = eval_examples[i:i + batch_size]
        print(f"  Evaluating: {min(i+batch_size, len(eval_examples))}/{len(eval_examples)}", end="\r")

        for ex in batch:
            all_labels = list(ex.entities.keys())
            if not all_labels:
                continue
            pred = model.extract_entities(ex.text, all_labels).get("entities", {})
            for label in all_labels:
                gold_spans = set(ex.entities.get(label, []))
                pred_spans = set(pred.get(label, []))
                tp[label] += len(gold_spans & pred_spans)
                fp[label] += len(pred_spans - gold_spans)
                fn[label] += len(gold_spans - pred_spans)

    print(f"\n{'Label':<60} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 78)
    f1s         = []
    weak_labels = []

    for label in sorted(set(list(tp) + list(fn))):
        p  = tp[label] / (tp[label] + fp[label] + 1e-9)
        r  = tp[label] / (tp[label] + fn[label] + 1e-9)
        f1 = 2*p*r / (p+r+1e-9)
        f1s.append(f1)
        flag = " ←" if f1 < 0.5 else ""
        print(f"{label:<60} {p:>6.3f} {r:>6.3f} {f1:>6.3f}{flag}")
        if f1 < 0.5:
            weak_labels.append(label)

    print(f"\nMacro F1:             {sum(f1s)/len(f1s):.3f}")
    print(f"Labels below 0.5 F1: {len(weak_labels)}")
    print(f"\nNext step: increase MAX_PER_LABEL to 150 and re-run,")
    print(f"targeting these {len(weak_labels)} weak labels.")

    with open("weak_labels.json", "w") as f:
        json.dump(weak_labels, f, indent=2)
    print("Saved weak_labels.json for targeted resampling.")