"""Stream-verify pinned model sources before queued model construction."""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def verify_model_group(group):
    receipt=ROOT/'labeled_predictions_v1/MODEL_SOURCE_RECEIPT.json';data=json.loads(receipt.read_text())
    for path,meta in data['groups'][group].items():
        assert Path(path).stat().st_size==meta['bytes'] and sha256(path)==meta['sha256'],path
    return {'model_group':group,'model_source_receipt':str(receipt),'model_source_receipt_sha256':sha256(receipt),'verified_files':len(data['groups'][group])}
