import json
with open("./training_data/train.json") as f:
    data = json.load(f)
for i in [0, 100, 300]:
    item = data[i]
    gt = json.loads(item["conversations"][1]["value"])
    p = item["conversations"][0]["value"]
    op = gt["dominant_operation"]
    seg = gt["temporal_segment"]
    print(f"Example {i}: op={op}, seg={seg}")
    print(f"  Prompt: ...{p[130:300]}")
    print()
