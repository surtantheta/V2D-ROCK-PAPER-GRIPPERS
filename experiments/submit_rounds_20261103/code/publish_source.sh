#!/bin/bash
# Publish the post-processing source to a new branch of the team repo and verify it remotely.
# Usage (PBS): publish_source.sh <branch> <message-file>
set -euo pipefail
BR=$1; MSG=$2
REPO=/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/joint_kaggle_20261031/source_checkout
WT=/rhome/tanuj/v2d_publish_20261103/$BR
V=/rhome/tanuj/home_moved_20261029/V2D_challenge
cd $REPO
git fetch -q origin
if [ ! -d $WT ]; then git worktree add -q -b $BR $WT origin/joint-pi3x-20261031; fi
cd $WT
for d in experiments/physics_pp_20261101 experiments/submit_rounds_20261103; do
  mkdir -p $d/code
  cp $V/$d/code/*.py $V/$d/code/*.sh $d/code/ 2>/dev/null || true
  cp $V/$d/*.pbs $d/ 2>/dev/null || true
done
git add -A experiments/physics_pp_20261101 experiments/submit_rounds_20261103
git -c user.name="Tanuj Sur" -c user.email="41645324+surtantheta@users.noreply.github.com" commit -q -F $MSG || echo "nothing new to commit"
git push -q origin HEAD:refs/heads/$BR
SHA=$(git rev-parse HEAD)
git fetch -q origin $BR
test "$(git rev-parse origin/$BR)" = "$SHA"
python3 - "$WT" "$SHA" "$BR" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
wt, sha, br = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = subprocess.run(['git', '-C', str(wt), 'ls-tree', '-r', '--name-only', sha, 'experiments/physics_pp_20261101', 'experiments/submit_rounds_20261103'], capture_output=True, text=True, check=True).stdout.split()
out = {}
for f in files:
    remote = subprocess.run(['git', '-C', str(wt), 'show', f'origin/{br}:{f}'], capture_output=True, check=True).stdout
    local = (Path('/rhome/tanuj/home_moved_20261029/V2D_challenge') / f).read_bytes()
    assert hashlib.sha256(remote).hexdigest() == hashlib.sha256(local).hexdigest(), f
    out[f] = hashlib.sha256(local).hexdigest()
body = subprocess.run(['git', '-C', str(wt), 'log', '-1', '--format=%an <%ae>%n%B', sha], capture_output=True, text=True, check=True).stdout
assert 'Co-Authored-By' not in body and 'Claude' not in body
rec = {'repository': 'surtantheta/V2D-ROCK-PAPER-GRIPPERS', 'branch': br, 'commit_sha': sha,
       'commit_url': f'https://github.com/surtantheta/V2D-ROCK-PAPER-GRIPPERS/commit/{sha}',
       'remote_file_hashes_verified': len(out), 'source_sha256': out, 'author': body.splitlines()[0]}
p = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/submit_rounds_20261103') / f'SOURCE_RECEIPT_{sha[:8]}.json'
p.write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != 'source_sha256'}))
PY
