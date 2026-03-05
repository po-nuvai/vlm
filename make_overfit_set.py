"""Create a tiny balanced dataset for overfitting test."""
import json

with open("./training_data/train.json") as f:
    data = json.load(f)

# Pick 5 examples per class
seen = {}
for item in data:
    gt = json.loads(item["conversations"][1]["value"])
    op = gt["dominant_operation"]
    if op not in seen:
        seen[op] = []
    if len(seen[op]) < 5:
        seen[op].append(item)

# Duplicate each to get 200 examples total (25 per class × 8 classes)
overfit = []
for op, items in seen.items():
    for item in items:
        overfit.extend([item] * 5)  # 5 copies each = 25 per class

print(f"Total: {len(overfit)} examples")
for op in seen:
    print(f"  {op}: {len(seen[op]) * 5}")

with open("./training_data/overfit_train.json", "w") as f:
    json.dump(overfit, f)
