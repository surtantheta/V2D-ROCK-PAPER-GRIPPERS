import os
from huggingface_hub import hf_hub_download, list_repo_files, whoami
print("user", whoami().get("name"))
repo = "nvidia/cari4d_commercial"
try:
    files = list_repo_files(repo)
    print("listing", len(files), files[:20])
except Exception as e:
    print("LIST FAIL", type(e).__name__, str(e)[:200])
try:
    small = [f for f in files if f.endswith((".json", ".md", ".yaml", ".yml", ".txt"))][:1]
    p = hf_hub_download(repo, small[0], local_dir=os.path.expanduser("~/V2D_challenge/tmp_probe/cari4d_commercial"))
    print("DOWNLOAD OK", small[0])
except Exception as e:
    print("DOWNLOAD FAIL", type(e).__name__, str(e)[:200])
