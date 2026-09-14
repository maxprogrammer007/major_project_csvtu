"""Inspect the exact schema of the ajimeno/PubTabNet HuggingFace mirror.
Streams a few items and prints full nested structure so we know exactly
what fields are available before writing token extraction code.
"""
from datasets import load_dataset
import json

def describe(v, indent=2, depth=0):
    pad = " " * indent * depth
    t = type(v).__name__
    if isinstance(v, dict):
        print(f"{pad}dict, keys={list(v.keys())}")
        for k2, v2 in list(v.items())[:6]:  # first 6 keys
            print(f"{pad}  [{k2!r}]:")
            describe(v2, indent, depth + 2)
    elif isinstance(v, list):
        print(f"{pad}list[{len(v)}]", end="")
        if v:
            print(f", elem type={type(v[0]).__name__}")
            if hasattr(v[0], "keys") or isinstance(v[0], dict):
                print(f"{pad}  [0]:")
                describe(v[0], indent, depth + 2)
            else:
                print(f"{pad}  [0] = {repr(v[0])[:80]}")
        else:
            print(" (empty)")
    else:
        r = repr(v)
        print(f"{pad}{t} = {r[:120]}")

ds = load_dataset("ajimeno/PubTabNet", split="train", streaming=True)

for sample_i, item in enumerate(ds):
    if sample_i >= 3:
        break
    print(f"\n{'='*60}")
    print(f"Item {sample_i}  __key__={item.get('__key__', 'N/A')}")
    print(f"{'='*60}")
    for k, v in item.items():
        print(f"\nField: {k!r}")
        describe(v, indent=2, depth=1)

print("\nDone.")
