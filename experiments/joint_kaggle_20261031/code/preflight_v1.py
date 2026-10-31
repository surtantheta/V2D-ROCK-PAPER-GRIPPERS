#!/usr/bin/env python3
"""Read-only Track 1 identity, limits, and submission-history audit in PBS."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge')
KIT = PROJECT / 'data/assets/submission_kit/v2d_submission_kit'


def official_module():
    spec = importlib.util.spec_from_file_location('official_joint_kaggle_submit', KIT / 'submit/v2d_submit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_date(value):
    if not isinstance(value, datetime.datetime):
        value = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return value.replace(tzinfo=datetime.timezone.utc) if value.tzinfo is None else value


def collect(metrics=None, require_available=True):
    official = official_module()
    api = official.build_api()
    all_competitions = json.loads((KIT / 'config/tracks.json').read_text())['tracks']['track_1']['competitions']
    competitions = all_competitions if metrics is None else {key: all_competitions[key] for key in metrics}
    info = official.preflight(api, competitions)
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=7)
    history = {}
    allowed = True
    for metric, slug in competitions.items():
        entry = info[metric]
        assert entry['entered'] is True
        assert entry['team_name'] == 'Rock Paper Grippers', ('team identity mismatch', metric)
        assert entry['user'] == 'surtantheta', ('account mismatch', metric)
        records = api.competition_submissions(slug, page_size=100) or []
        rows = []
        for item in records:
            when = normalized_date(item.date)
            if when < cutoff:
                continue
            rows.append(dict(ref=str(item.ref), date_utc=when.isoformat(),
                             team_name=getattr(item, 'team_name', None),
                             user=getattr(item, 'submitted_by_ref', None),
                             file_name=getattr(item, 'file_name', None),
                             description=getattr(item, 'description', None),
                             public_score=getattr(item, 'public_score', None),
                             status=str(getattr(item, 'status', ''))))
        history[metric] = rows
        # Deliberately conservative until the organizer's exact unlimited-window
        # boundary is confirmed; each metric retains its own accepted-round count.
        eligible = len(rows) < 5 and entry.get('remaining') != 0
        allowed = allowed and eligible
    result = dict(status='PASS_READ_ONLY_PREFLIGHT' if allowed else 'LIMIT_BLOCKED_READ_ONLY_PREFLIGHT',
                  checked_utc=now.isoformat(), read_only=True, Kaggle_upload_performed=False,
                  team='Rock Paper Grippers', user='surtantheta', metrics=info,
                  submissions_last_seven_days=history,
                  conservative_weekly_allowance_per_metric={k: max(0, 5 - len(v)) for k, v in history.items()},
                  requested_metrics=list(competitions), available_for_one_round=allowed,
                  organizer_rule='five per week; final-three-day exception not automatically inferred',
                  PBS_JOBID=os.environ.get('PBS_JOBID'))
    if require_available:
        assert allowed, 'Submission allowance unavailable; inspect this read-only history before uploading'
    return result


def main():
    assert os.environ.get('PBS_JOBID'), 'Preflight runs in PBS'
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = collect(require_available=False)
    assert not a.out.exists(), 'Preserve old preflight snapshots'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], allowance=report['conservative_weekly_allowance_per_metric'],
                          identity='Rock Paper Grippers / surtantheta', output=str(a.out))), flush=True)


if __name__ == '__main__':
    main()
