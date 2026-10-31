#!/usr/bin/env python3
"""One authorized five-metric round; polling never uploads and retries are explicit.

No automatic retry of the whole round is allowed. The official uploader records
each accepted ID before continuing. A connection-ambiguous metric must first be
reconciled with fresh Kaggle history by root before --resume-missing is usable.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/data/assets/submission_kit/v2d_submission_kit')
ARTIFACT = ROOT / 'ready_v2/track1_pi3x_joint_physics.parquet'
RECEIPT = ROOT / 'ready_v2/JOINT_KAGGLE_RECEIPT_V2.json'
ATTEMPT = ROOT / 'ready_v2/JOINT_UPLOAD_ATTEMPT_V2.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validated_artifact():
    gate_path = ROOT / 'LAUNCH_UPLOAD_V2.json'
    gate = json.loads(gate_path.read_text())
    assert gate['stage'] == 'JOINT_FULL30_UPLOAD_V2'
    assert gate['user_authorized_one_round'] is True
    assert gate['file_sha256'] == sha(ARTIFACT)
    for path, digest in gate['sha256'].items():
        assert sha(path) == digest, ('upload source/input changed', path)
    audit_path = ROOT / 'validation/FINAL_UPLOAD_AUDIT_V2.json'
    audit = json.loads(audit_path.read_text())
    report = json.loads((ROOT / 'ready_v2/PACK_VALIDATION.json').read_text())
    assert audit['status'] == 'PASS_UPLOAD_READY'
    assert (audit['cases'], audit['full_frames'], audit['scored_frames'], audit['rows']) == (30, 16563, 9877, 740780)
    assert audit['GT_used_for_inference'] is False
    assert sha(ARTIFACT) == audit['file_sha256'] == report['file_sha256']
    assert gate['independent_audit_sha256'] == sha(audit_path)
    commit = (ROOT / 'SOURCE_COMMIT_URL_V2.txt').read_text().strip()
    assert audit['code_commit_url'] == report['code_commit_url'] == gate['code_commit_url'] == commit
    publication = json.loads((ROOT / 'SOURCE_PUBLICATION_RECEIPT_V2.json').read_text())
    assert publication['published'] and publication['public_repo_verified']
    assert commit.endswith('/' + publication['commit_sha'])
    for path, digest in publication['source_sha256'].items():
        assert sha(path) == digest, ('published source changed', path)
    return publication, commit


def report_status(official, code):
    receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {'submissions': {}}
    entries = receipt['submissions']
    accepted = {key: entry['ref'] for key, entry in entries.items() if entry.get('ref')}
    complete = len(entries) == 5 and all(entry.get('state') == 'COMPLETE' for entry in entries.values())
    status = dict(status='ALL_FIVE_SCORES_COMPLETE' if complete else 'INSPECT_EXACT_RECEIPT_DO_NOT_REUPLOAD_ROUND',
                  accepted_submission_ids=accepted,
                  states={key: entry.get('state') for key, entry in entries.items()},
                  scores={key: entry.get('public_score') for key, entry in entries.items()},
                  official_return_code=code, file_sha256=sha(ARTIFACT),
                  receipt=str(RECEIPT), checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  automatic_round_retry=False, poll_mode_never_uploads=True)
    write(ROOT / 'ready_v2/UPLOAD_STATUS.json', status)
    print(json.dumps(status), flush=True)
    return 0 if complete else 2


def main():
    assert os.environ.get('PBS_JOBID'), 'Upload and polling run in PBS'
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--upload', action='store_true')
    mode.add_argument('--poll', action='store_true')
    mode.add_argument('--resume-missing', action='store_true')
    a = parser.parse_args()
    preflight = load(ROOT / 'code/preflight_v1.py', 'joint_readonly_preflight_for_upload')
    official = preflight.official_module()
    if a.poll:
        assert RECEIPT.exists(), 'There are no saved IDs to poll'
        result = subprocess.run([sys.executable, '-I', '-B', '-u', str(KIT / 'submit/v2d_submit.py'),
                                 '--track', 'track_1', '--poll-receipt', str(RECEIPT)], cwd=KIT)
        raise SystemExit(report_status(official, result.returncode))
    publication, commit = validated_artifact()
    message = 'Rock Paper Grippers - Pi3X joint temporal collision optimization - ' + publication['commit_sha'][:8] + ' ' + sha(ARTIFACT)[:8]
    if a.upload:
        assert not (ROOT / 'ready/JOINT_UPLOAD_ATTEMPT.json').exists(), 'An earlier round attempt must be reconciled before starting V2'
        assert not (ROOT / 'ready/JOINT_KAGGLE_RECEIPT.json').exists(), 'An earlier round receipt must be reconciled before starting V2'
        assert not RECEIPT.exists() and not ATTEMPT.exists(), 'Inspect saved IDs/attempt before any retry'
        report = preflight.collect(require_available=True)
        write(ROOT / 'ready_v2/PREFLIGHT_BEFORE_UPLOAD.json', report)
        attempt = dict(track='track_1', file=str(ARTIFACT), file_sha256=sha(ARTIFACT),
                       code_commit_url=commit, message=message,
                       initial_submission_refs={key: [entry['ref'] for entry in rows] for key, rows in report['submissions_last_seven_days'].items()},
                       authorized_rounds=1, PBS_JOBID=os.environ['PBS_JOBID'])
        # Reserve a second exclusive marker before the official process starts;
        # an interruption before it creates its receipt cannot cause a duplicate.
        with ATTEMPT.open('x') as f:
            json.dump(attempt, f, indent=2)
        result = subprocess.run([sys.executable, '-I', '-B', '-u', str(KIT / 'submit/v2d_submit.py'),
                                 str(ARTIFACT), '--track', 'track_1', '-m', message,
                                 '--receipt', str(RECEIPT), '--wait'], cwd=KIT)
        raise SystemExit(report_status(official, result.returncode))
    # Root must reconcile the saved IDs and any ambiguous uploads against fresh
    # history before explicitly authorizing only the metrics confirmed absent.
    resume_path = ROOT / 'RESUME_UPLOAD_V2.json'
    resume = json.loads(resume_path.read_text())
    assert resume['root_confirmed_not_accepted'] is True
    assert resume['receipt_sha256'] == sha(RECEIPT)
    assert resume['attempt_sha256'] == sha(ATTEMPT)
    assert resume['file_sha256'] == sha(ARTIFACT)
    receipt = json.loads(RECEIPT.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    assert attempt['message'] == message and attempt['code_commit_url'] == commit
    metrics = list(resume['confirmed_missing_metrics'])
    assert metrics and len(set(metrics)) == len(metrics)
    fresh = preflight.collect(require_available=False)
    for key in metrics:
        assert key in fresh['metrics']
        assert not receipt['submissions'].get(key, {}).get('ref'), ('metric already accepted', key)
        observed = sorted(entry['ref'] for entry in fresh['submissions_last_seven_days'][key])
        assert observed == sorted(resume['observed_submission_refs'][key]), ('history changed since reconciliation', key)
        assert len(observed) < 5 and fresh['metrics'][key].get('remaining') != 0
    auxiliary = ROOT / 'ready_v2' / resume['new_receipt_name']
    assert auxiliary.parent.resolve() == (ROOT / 'ready_v2').resolve() and not auxiliary.exists()
    lock = ROOT / 'ready_v2' / (resume['new_receipt_name'] + '.attempt')
    with lock.open('x') as f:
        json.dump(dict(resume_authorization_sha256=sha(resume_path), metrics=metrics), f)
    result = subprocess.run([sys.executable, '-I', '-B', '-u', str(KIT / 'submit/v2d_submit.py'),
                             str(ARTIFACT), '--track', 'track_1', '-m', message, '--only', *metrics,
                             '--receipt', str(auxiliary), '--wait'], cwd=KIT)
    if auxiliary.exists():
        extra = json.loads(auxiliary.read_text())
        assert extra['file'] == str(ARTIFACT.resolve()) and extra['message'] == message
        for key, entry in extra['submissions'].items():
            assert key in metrics and not receipt['submissions'].get(key, {}).get('ref')
            receipt['submissions'][key] = entry
        snapshot = ROOT / 'ready_v2' / (resume['new_receipt_name'] + '.prior-primary.json')
        assert not snapshot.exists()
        write(snapshot, json.loads(RECEIPT.read_text()))
        official.save_receipt(RECEIPT, receipt)
    raise SystemExit(report_status(official, result.returncode))


if __name__ == '__main__':
    main()
