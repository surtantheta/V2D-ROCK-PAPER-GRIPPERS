import sys, torch, numpy as np
def show(d, pre="", depth=0):
    if depth > 3: return
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)) and not torch.is_tensor(v):
                print(f"{pre}{k}: {type(v).__name__} len={len(v)}"); show(v if isinstance(v, dict) else {}, pre + "  ", depth + 1)
            else:
                s = tuple(v.shape) if hasattr(v, "shape") else (v if isinstance(v, (int, float, str)) else type(v).__name__)
                print(f"{pre}{k}: {s}")
for p in sys.argv[1:]:
    print("==", p); show(torch.load(p, map_location="cpu", weights_only=False))
