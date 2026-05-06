"""
evaluate.py
─────────────────
Evaluates and compares the base GLiNER2 model against a fine-tuned LoRA adapter
on the FiNER-139 financial NER test set.

What it does:
  1. Loads tokenized test examples from FiNER-139 test.jsonl
  2. Runs both the base model and fine-tuned adapter on each example
  3. Compares predicted entity spans against gold BIO annotations
  4. Reports per-label Precision / Recall / F1 and overall Macro F1

"""

import json
from collections import defaultdict
from gliner2 import GLiNER2

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def bio_to_spans(tokens, ner_tags):
    entities, current, current_lbl = defaultdict(list), [], None
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

def spans_match_partial(gold_spans, pred_spans):
    """
    A predicted span is correct if it contains the gold span or vice versa.
    e.g. '$29.9 billion' matches gold '29.9'
    """
    matched_gold = set()
    matched_pred = set()

    for gold in gold_spans:
        for pred in pred_spans:
            if gold in pred or pred in gold:
                matched_gold.add(gold)
                matched_pred.add(pred)
                break

    tp = len(matched_gold)
    fp = len(pred_spans - matched_pred)
    fn = len(gold_spans - matched_gold)
    return tp, fp, fn

def score_model(model, test_examples, batch_size=32, max_seq_len=256):
    """
    Score a model against gold BIO annotations from the dataset.
    Uses partial span matching so '$29.9 billion' correctly matches gold '29.9'.
    Filters long sequences to match what the fine-tuned model was trained on.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    skipped = 0

    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        texts, golds, label_sets = [], [], []

        for tokens, ner_tags in batch:
            # Skip long sequences — model wasn't trained on these
            if len(tokens) > max_seq_len:
                skipped += 1
                continue

            gold   = bio_to_spans(tokens, ner_tags)
            labels = list(gold.keys())
            if not labels:
                continue

            texts.append(" ".join(tokens))
            golds.append(gold)
            label_sets.append(labels)

        if not texts:
            continue

        for text, gold, labels in zip(texts, golds, label_sets):
            pred = model.extract_entities(text, labels).get("entities", {})

            # ── Debug: print first 3 examples ─────────────────────────────────
            if i < 3:
                print(f"\n  Text: {text[:80]}")
                for label in labels:
                    g = gold.get(label, [])
                    p = pred.get(label, [])
                    if g or p:
                        print(f"    [{label}]")
                        print(f"      Gold: {g}")
                        print(f"      Pred: {p}")
            # ──────────────────────────────────────────────────────────────────

            for label in labels:
                gold_spans = set(gold.get(label, []))
                pred_spans = set(pred.get(label, []))

                tp_l, fp_l, fn_l = spans_match_partial(gold_spans, pred_spans)
                tp[label] += tp_l
                fp[label] += fp_l
                fn[label] += fn_l

        done = min(i + batch_size, len(test_examples))
        print(f"  Progress: {done}/{len(test_examples)}", end="\r")

    print(f"  Done. Skipped {skipped} long sequences.        ")

    # Per-label F1
    results = {}
    for label in set(list(tp) + list(fn)):
        p  = tp[label] / (tp[label] + fp[label] + 1e-9)
        r  = tp[label] / (tp[label] + fn[label] + 1e-9)
        f1 = 2*p*r / (p+r+1e-9)
        results[label] = {"p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3)}

    macro_f1 = sum(v["f1"] for v in results.values()) / len(results)
    return results, round(macro_f1, 3)


if __name__ == "__main__":
    test_raw      = load_jsonl("./finer-139/test.jsonl")
    test_examples = [(ex["tokens"], ex["ner_tags"]) for ex in test_raw[:5000]]

    print(f"Loaded {len(test_examples)} test examples")

    base_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    ft_model   = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    ft_model.load_adapter("./finer139_adapter/best")

    print("\nScoring base model...")
    base_scores, base_macro = score_model(base_model, test_examples)

    print("\nScoring fine-tuned model...")
    ft_scores, ft_macro = score_model(ft_model, test_examples)

    # ── Per-label results ─────────────────────────────────────────────────────
    all_labels = sorted(set(list(base_scores) + list(ft_scores)))

    print(f"\n{'Label':<55} {'Base F1':>8} {'FT F1':>8} {'Delta':>8}")
    print("-" * 82)

    improved  = 0
    degraded  = 0
    unchanged = 0

    for label in all_labels:
        b     = base_scores.get(label, {"f1": 0})["f1"]
        ft    = ft_scores.get(label,   {"f1": 0})["f1"]
        delta = ft - b

        if delta > 0.05:
            flag = " ↑"
            improved += 1
        elif delta < -0.05:
            flag = " ↓"
            degraded += 1
        else:
            flag = ""
            unchanged += 1

        print(f"{label:<55} {b:>8.3f} {ft:>8.3f} {delta:>+8.3f}{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 82)
    print(f"{'MACRO F1':<55} {base_macro:>8.3f} {ft_macro:>8.3f} {ft_macro - base_macro:>+8.3f}")
    print("=" * 82)
    print(f"\nLabels improved  (delta > +0.05): {improved}")
    print(f"Labels degraded  (delta < -0.05): {degraded}")
    print(f"Labels unchanged               :  {unchanged}")
    print(f"\nNote: Partial span matching used — '$29.9 billion' correctly matches gold '29.9'.")
    print(f"Both models evaluated on identical tokenized text and gold spans from FiNER-139 test.jsonl.")