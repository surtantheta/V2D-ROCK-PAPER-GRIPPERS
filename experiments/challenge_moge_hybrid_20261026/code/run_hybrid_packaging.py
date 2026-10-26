#!/usr/bin/env python3
"""Reserved-CPU orchestration only; preserves isolated decoder/packer environments."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT=Path('/home/tanuj/V2D_challenge/experiments/challenge_moge_hybrid_20261026')
CARI=Path('/home/tanuj/.conda/envs/cari4d/bin/python')
V2D=Path('/home/tanuj/.conda/envs/v2d/bin/python')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--commit-url');p.add_argument('--published-receipt',type=Path);p.add_argument('--source-repository',type=Path)
 a=p.parse_args();assert os.environ.get('PBS_JOBID'),'requires allocated CPU job'
 optional=[a.commit_url,a.published_receipt,a.source_repository]
 assert not any(optional) or all(optional),'provide all three finalization arguments or none'
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
          NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
 commands=[[str(CARI),'-I','-u',str(ROOT/'code/challenge_hybrid.py'),'--export'],
           [str(V2D),'-I','-u',str(ROOT/'code/pack_challenge_hybrid.py'),'--stage']]
 if all(optional):
  commands.append([str(V2D),'-I','-u',str(ROOT/'code/pack_challenge_hybrid.py'),'--finalize',
                   '--commit-url',a.commit_url,'--published-receipt',str(a.published_receipt),
                   '--source-repository',str(a.source_repository)])
 for cmd in commands:
  print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
