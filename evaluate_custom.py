"""
evaluate_custom.py
───────────────────────
Qualitative evaluation of the base GLiNER2 model vs a fine-tuned LoRA adapter
using hand-written test sentences with manually defined ground truth.

What it does:
  1. Loads both the base model and fine-tuned adapter
  2. Runs each model on a set of hand-written financial sentences
  3. Compares predictions against manually defined gold spans
  4. Reports per-test pass/fail (✓/✗) and an overall Precision/Recall/F1 summary
"""

from gliner2 import GLiNER2

# ── Load models ───────────────────────────────────────────────────────────────
base_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
finetuned  = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
finetuned.load_adapter("./finer139_adapter/best")

# ── Test texts with ground truth ──────────────────────────────────────────────
test_cases = [
    (
        "Tesla reported total revenues of $24.3 billion for the third quarter of 2023.",
        {"Revenues": ["24.3"]}
    ),
    (
        "Amazon's net income fell to $2.9 billion, down from $9.9 billion in the prior year.",
        {"NetIncomeLoss": ["2.9", "9.9"]}
    ),
    (
        "The company recorded research and development expenses of $4.1 billion during the period.",
        {"ResearchAndDevelopmentExpense": ["4.1"]}
    ),
    (
        "Google's operating income increased to $21.3 billion compared to $17.1 billion last year.",
        {"OperatingIncomeLoss": ["21.3", "17.1"]}
    ),
    (
        "Total assets on the balance sheet were $512.0 billion as of December 31 2023.",
        {"Assets": ["512.0"]}
    ),
    (
        "The firm issued long term debt of $8.5 billion to fund its acquisition activities.",
        {"LongTermDebt": ["8.5"]}
    ),
    (
        "Earnings per share for the quarter was $2.18 basic and $2.15 diluted.",
        {
            "EarningsPerShareBasic":   ["2.18"],
            "EarningsPerShareDiluted": ["2.15"]
        }
    ),
    (
        "Cost of goods sold was $31.2 billion representing 62 percent of total net sales.",
        {"CostOfGoodsSold": ["31.2"]}
    ),
    (
        "The company paid dividends of $0.92 per share totalling $3.8 billion for the year.",
        {"Dividends": ["0.92", "3.8"]}
    ),
    (
        "Cash and cash equivalents at end of period were $29.9 billion up from $21.0 billion.",
        {"CashAndCashEquivalentsAtCarryingValue": ["29.9", "21.0"]}
    ),
    (
    "Microsoft announced a net income of $24.7 billion for the fiscal quarter ending December 31, 2025.",
    {"NetIncome": ["24.7"]}
    ),

    (
        "Diluted earnings per share for the fourth quarter was $3.15, compared to $2.75 in the same period last year.",
        {"EarningsPerShare": ["3.15"]}
    ),

    (
        "The company recorded total operating expenses of $12.8 billion, a 5% increase year-over-year.",
        {"OperatingExpenses": ["12.8"]}
    ),

    (
        "As of the end of the fiscal year, total assets were valued at $450.2 billion, driven primarily by property, plant, and equipment.",
        {"TotalAssets": ["450.2"]}
    ),

    (
        "Cash and cash equivalents stood at $18.9 billion on the balance sheet as of March 31, 2026.",
        {"CashAndCashEquivalents": ["18.9"]}
    ),

    (
        "The firm reported a gross profit of $8.4 billion on net sales of $22.1 billion.",
        {"GrossProfit": ["8.4"], "Revenues": ["22.1"]}
    ),

    (
        "Long-term debt was reduced to $15.6 billion, following the repayment of senior notes due this year.",
        {"LongTermDebt": ["15.6"]}
    ),

    (
        "Research and development costs rose to $4.2 billion due to investments in AI infrastructure.",
        {"ResearchAndDevelopmentExpenses": ["4.2"]}
    ),

    (
        "The company's effective tax rate for the period was 21%, resulting in a tax provision of $1.2 billion.",
        {"IncomeTaxProvision": ["1.2"]}
    ),

    (
        "Total shareholders' equity increased to $112.5 billion, reflecting strong retained earnings growth.",
        {"TotalShareholdersEquity": ["112.5"]}
    )
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_all_labels(model, text, labels):
    result   = model.extract_entities(text, labels)
    entities = result.get("entities", {})
    return {
        label: [e["text"] if isinstance(e, dict) else e for e in spans]
        for label, spans in entities.items()
    }

def spans_match_partial(gold_spans, pred_spans):
    """
    A predicted span is correct if it contains the gold span or vice versa.
    e.g. '$29.9 billion' correctly matches gold '29.9'.
    Returns matched_gold, matched_pred sets for accurate FP counting.
    """
    matched_gold = set()
    matched_pred = set()

    for gold in gold_spans:
        for pred in pred_spans:
            if gold in pred or pred in gold:
                matched_gold.add(gold)
                matched_pred.add(pred)
                break

    return matched_gold, matched_pred

# ── Run evaluation ────────────────────────────────────────────────────────────
print("=" * 80)
print(f"{'HAND-WRITTEN TEST EVALUATION':^80}")
print("=" * 80)

total_tp_base = total_fp_base = total_fn_base = 0
total_tp_ft   = total_fp_ft   = total_fn_ft   = 0

for i, (text, gold) in enumerate(test_cases, 1):
    labels = list(gold.keys())

    base_pred = extract_all_labels(base_model, text, labels)
    ft_pred   = extract_all_labels(finetuned,  text, labels)

    print(f"\nTest {i}: {text[:75]}...")
    print(f"  Labels tested: {labels}")

    for label in labels:
        gold_spans = set(gold.get(label, []))
        base_spans = set(base_pred.get(label, []))
        ft_spans   = set(ft_pred.get(label,   []))

        # Partial matching
        base_matched_gold, base_matched_pred = spans_match_partial(gold_spans, base_spans)
        ft_matched_gold,   ft_matched_pred   = spans_match_partial(gold_spans, ft_spans)

        tp_base = len(base_matched_gold)
        fp_base = len(base_spans - base_matched_pred)   # predicted but no gold match
        fn_base = len(gold_spans - base_matched_gold)   # gold but no pred match

        tp_ft   = len(ft_matched_gold)
        fp_ft   = len(ft_spans - ft_matched_pred)
        fn_ft   = len(gold_spans - ft_matched_gold)

        total_tp_base += tp_base
        total_fp_base += fp_base
        total_fn_base += fn_base
        total_tp_ft   += tp_ft
        total_fp_ft   += fp_ft
        total_fn_ft   += fn_ft

        base_correct = "✓" if tp_base == len(gold_spans) and fp_base == 0 else "✗"
        ft_correct   = "✓" if tp_ft   == len(gold_spans) and fp_ft   == 0 else "✗"

        print(f"  [{label}]")
        print(f"    Expected:   {sorted(gold_spans)}")
        print(f"    Base   {base_correct}: {sorted(base_spans) if base_spans else '—'}")
        print(f"    FT     {ft_correct}: {sorted(ft_spans)   if ft_spans   else '—'}")

# ── Summary ───────────────────────────────────────────────────────────────────
def compute_f1(tp, fp, fn):
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2*p*r / (p+r+1e-9)
    return f1, p, r

base_f1, base_p, base_r = compute_f1(total_tp_base, total_fp_base, total_fn_base)
ft_f1,   ft_p,   ft_r   = compute_f1(total_tp_ft,   total_fp_ft,   total_fn_ft)

print("\n" + "=" * 80)
print(f"{'SUMMARY':^80}")
print("=" * 80)
print(f"{'Metric':<30} {'Base':>12} {'Fine-tuned':>12} {'Delta':>10}")
print(f"{'Precision':<30} {base_p:>12.3f} {ft_p:>12.3f} {ft_p - base_p:>+10.3f}")
print(f"{'Recall':<30} {base_r:>12.3f} {ft_r:>12.3f} {ft_r - base_r:>+10.3f}")
print(f"{'F1':<30} {base_f1:>12.3f} {ft_f1:>12.3f} {ft_f1 - base_f1:>+10.3f}")
print("=" * 80)
print(f"\nNote: Partial span matching used — '$29.9 billion' correctly matches gold '29.9'.")
print(f"FP is counted only when a prediction shares no overlap with any gold span.")