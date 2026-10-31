#!/usr/bin/env python3
"""Rebuild the5840-point hand-motion proxy from the public MHR model only.

This is the exact existing predicted-only index-generation method. It does not
recover or claim to match the hidden512 hosted role vertices.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODEL_SHA256 = '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('PBS_JOBID'), 'Model evaluation requires PBS'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-indices-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    assert sha(a.model) == MODEL_SHA256
    assert not a.out.exists()
    import torch
    torch.set_num_threads(2)
    model = torch.jit.load(str(a.model), map_location='cpu').eval()
    shape = np.zeros((1, 45), np.float32)
    parameters = np.zeros((55, 204), np.float32)
    parameters[1 + np.arange(54), np.arange(68, 122)] = .3
    values = []
    with torch.inference_mode():
        for start in range(0, len(parameters), 8):
            p = parameters[start:start + 8]
            s = np.repeat(shape, len(p), axis=0)
            v, _ = model(torch.from_numpy(s), torch.from_numpy(p), torch.zeros(len(p), 72), True)
            values.append(v.double().numpy() / 100 * np.array([1., -1., -1.]))
    vertices = np.concatenate(values)
    indices = np.flatnonzero(np.linalg.norm(vertices[1:] - vertices[:1], axis=-1).max(axis=0) > 1e-4)
    assert len(indices) == 5840
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, indices)
    assert sha(a.out) == a.expected_indices_sha256, 'Hand proxy differs from frozen source'
    a.out.with_suffix('.json').write_text(json.dumps(dict(status='PUBLIC_MHR_ONLY_REBUILD_PASS',
        vertices=5840, model_sha256=MODEL_SHA256, indices_sha256=sha(a.out),
        GT_used=False, hidden_host_roles_recovered=False, PBS_JOBID=os.environ['PBS_JOBID']), indent=2) + '\n')
    print('PUBLIC_MHR_ONLY_REBUILD_PASS', sha(a.out), flush=True)


if __name__ == '__main__':
    main()
