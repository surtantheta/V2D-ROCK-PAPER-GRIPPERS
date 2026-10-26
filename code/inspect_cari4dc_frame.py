import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
f0 = d["frames"][0]; m0 = d["frame_meta"][0]
print("frames[0] type", type(f0).__name__, f0 if isinstance(f0, (str, int)) else "")
def show(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict): print(pre + k + ":"); show(v, pre + "  ")
            else: print(f"{pre}{k}: {tuple(v.shape) if hasattr(v,'shape') else (v if isinstance(v,(int,float,str)) else type(v).__name__)}")
show(f0) if isinstance(f0, dict) else None
print("frame_meta[0]:"); show(m0) if isinstance(m0, dict) else print(m0)
other = [k for k in d if k not in ("frames", "frame_meta", "config", "mhr_inference", "checkpoint")]
print("other top keys:", other)
for k in other:
    v = d[k]; print(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
    if isinstance(v, dict): show(v, "   ")
