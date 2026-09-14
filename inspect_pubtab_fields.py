from datasets import load_dataset

ds = load_dataset("ajimeno/PubTabNet", split="train", streaming=True)
for i, item in enumerate(ds):
    print(f"Record {i} keys: {list(item.keys())}")
    for k in list(item.keys()):
        v = item[k]
        try:
            t = type(v)
            print(f"  {k}: type={t}, repr_len={len(repr(v))}")
        except Exception:
            print(f"  {k}: <unprintable>")
    if i >= 5:
        break
