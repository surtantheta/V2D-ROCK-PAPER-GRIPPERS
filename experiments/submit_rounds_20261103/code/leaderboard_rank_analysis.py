#!/usr/bin/env python3
"""Read-only snapshot of the five Track 1 public leaderboards, plus a sum-of-ranks analysis.

No submission or team calls. Ranks use competition ranking (1 + number of strictly better scores),
which treats ties as equal; tie-break by submission time is reported separately where it matters.
"""
import datetime, hashlib, json, sys
from pathlib import Path
from zoneinfo import ZoneInfo
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

OUT = Path(sys.argv[1])
METRICS = {'cd_h_cm': 'v2d-challenge-track1-cd-h', 'cd_o_cm': 'v2d-challenge-track1-cd-o',
           'acc_h_cm': 'v2d-challenge-track1-acc-h', 'acc_o_cm': 'v2d-challenge-track1-acc-o',
           'interpenetration_cm': 'v2d-challenge-track1-pen'}
OURS = ['rock paper', 'grippers', 'tanuj_sur', 'tanuj sur']

api = KaggleApi(); api.authenticate()
boards = {}
for metric, slug in METRICS.items():
    rows, token, pages = [], None, 0
    with api.build_kaggle_client() as client:
        while True:
            req = ApiGetLeaderboardRequest(); req.competition_name = slug; req.page_size = 100
            if token:
                req.page_token = token
            resp = client.competitions.competition_api_client.get_leaderboard(req); pages += 1
            rows.extend(x.to_dict() for x in (resp.submissions or []) if x is not None)
            token = resp.next_page_token
            if not token or pages >= 50:
                break
    for x in rows:
        x['score_numeric'] = float(x['score'])
    scores = [x['score_numeric'] for x in rows]
    for x in rows:
        x['rank'] = 1 + sum(s < x['score_numeric'] for s in scores)
    boards[metric] = rows

def is_ours(name):
    return any(t in name.lower() for t in OURS)

def rank_of(metric, value):
    return 1 + sum(x['score_numeric'] < value for x in boards[metric])

teams = {}
for metric, rows in boards.items():
    for x in rows:
        teams.setdefault(x['teamName'], {})[metric] = x
complete = {t: m for t, m in teams.items() if len(m) == 5}
combined = sorted(((sum(m[k]['rank'] for k in METRICS), t) for t, m in complete.items()))
ours_names = [t for t in teams if is_ours(t)]
ours = complete.get(ours_names[0]) if ours_names else None

summary = {'queried_sgt': datetime.datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
           'teams_on_any_board': len(teams), 'teams_on_all_five': len(complete),
           'our_team_names': ours_names, 'metrics': {}, 'combined_top': [], 'what_if': {}}
for metric, rows in boards.items():
    sc = sorted(x['score_numeric'] for x in rows)
    d = {'rows': len(rows), 'best': sc[0], 'thresholds_by_rank': {k: sc[k - 1] for k in (1, 3, 5, 10, 15, 20, 25, 30) if k <= len(sc)},
         'count_at_best': sum(s == sc[0] for s in sc)}
    if ours:
        d['ours'] = ours[metric]['score_numeric']; d['our_rank'] = ours[metric]['rank']
    cari = [(x['teamName'], x['score_numeric'], x['rank']) for x in rows if 'cari' in x['teamName'].lower()]
    d['cari_rows'] = cari
    summary['metrics'][metric] = d
for s, t in combined[:25]:
    summary['combined_top'].append({'team': t, 'rank_sum': s, 'ranks': {k: complete[t][k]['rank'] for k in METRICS},
                                    'scores': {k: complete[t][k]['score_numeric'] for k in METRICS}})
if ours:
    our_sum = sum(ours[k]['rank'] for k in METRICS)
    summary['ours_rank_sum'] = our_sum
    summary['ours_combined_position'] = 1 + sum(s < our_sum for s, _ in combined)
    # What-if: change one metric, keep the others, other teams unchanged.
    candidates = {'cd_o_cm': [30.0, 25.0, 23.1625, 20.0, 18.0, 16.5], 'cd_h_cm': [14.5, 14.0, 13.5, 13.0, 12.5],
                  'acc_o_cm': [0.12, 0.10, 0.09, 0.08, 0.07], 'acc_h_cm': [0.15, 0.145, 0.14, 0.135],
                  'interpenetration_cm': [0.002, 0.001, 0.0005, 0.0001, 0.0]}
    for metric, vals in candidates.items():
        out = []
        for v in vals:
            others = our_sum - ours[metric]['rank']
            # Exclude our own row when ranking the hypothetical value.
            r = 1 + sum(x['score_numeric'] < v for x in boards[metric] if not is_ours(x['teamName']))
            new_sum = others + r
            pos = 1 + sum(s < new_sum for s, t in combined if not is_ours(t))
            out.append({'value': v, 'metric_rank': r, 'rank_sum': new_sum, 'combined_position': pos})
        summary['what_if'][metric] = out
report = {'summary': summary, 'boards': boards,
          'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'read_only': True, 'submissions_performed': 0}
OUT.write_text(json.dumps(report, indent=1, default=str) + '\n')
print(json.dumps(summary, indent=1, default=str))
