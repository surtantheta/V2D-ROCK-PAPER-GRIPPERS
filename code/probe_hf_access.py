import os
from huggingface_hub import hf_hub_download, whoami
print("user", whoami().get("name"))
for repo in ["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"]:
    try:
        hf_hub_download(repo, "README.md", local_dir=os.path.expanduser("~/V2D_challenge/tmp_probe/" + repo.split("/")[1]))
        print("ACCESS OK", repo)
    except Exception as e:
        print("ACCESS FAIL", repo, type(e).__name__, str(e)[:200])
