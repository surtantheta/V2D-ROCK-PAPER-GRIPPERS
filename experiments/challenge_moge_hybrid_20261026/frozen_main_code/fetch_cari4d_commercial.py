import hashlib, os
from huggingface_hub import snapshot_download
d = snapshot_download("nvidia/cari4d_commercial", local_dir="/home/tanuj/V2D_challenge/models/cari4d_commercial")
p = os.path.join(d, "2026-08-25-09-35-57/step200000.pth")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
print("downloaded", d, os.path.getsize(p), "sha256", h,
      "MATCH" if h == "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3" else "MISMATCH")
