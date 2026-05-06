import json

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

# Update these paths to match where you cloned the repo
DATA_DIR = "./finer-139"

train_raw = load_jsonl(f"{DATA_DIR}/train.jsonl")
val_raw   = load_jsonl(f"{DATA_DIR}/validation.jsonl")
test_raw  = load_jsonl(f"{DATA_DIR}/test.jsonl")

print(f"Train: {len(train_raw):,}")
print(f"Val:   {len(val_raw):,}")
print(f"Test:  {len(test_raw):,}")

# Inspect a single example
print(train_raw[0])
# {
#   'tokens':   ['Apple', 'Inc.', 'reported', '$', '1.2', 'billion'],
#   'ner_tags': ['O', 'O', 'O', 'B-Revenues', 'I-Revenues', 'I-Revenues']
# }

# Collect all unique label names from the training set
# (tags are strings directly, no integer mapping needed)
label_names_set = set()
for ex in train_raw:
    for tag in ex["ner_tags"]:
        if tag != "O":
            label_names_set.add(tag[2:])   # Strip B-/I- prefix

label_names = sorted(label_names_set)
print(f"\n{len(label_names)} unique entity labels")
print(label_names[:5], "...")