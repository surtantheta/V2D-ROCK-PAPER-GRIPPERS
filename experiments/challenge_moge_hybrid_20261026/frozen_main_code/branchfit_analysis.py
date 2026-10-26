#!/usr/bin/env python3
"""V2D (copied from validator 05/06 v6_analysis.py; ASSEMBLE json optional, output under work/branchfit/analysis): per-episode sanity + PEN proxy on scored frames. Inputs = submission_v1 npz/ply (proven bit-identical
to the parquet in v5_contract.py). Human decoded with kit TorchScript MHR in the host frame (diag(1,-1,-1) @ MHR/100)."""
import importlib.util, json, sys, time
import numpy as np, pandas as pd, torch, trimesh
from scipy.spatial import cKDTree
torch.set_num_threads(4)
R = "/home/tanuj/V2D_challenge"; KIT = R + "/data/assets/submission_kit/v2d_submission_kit"
OUT = "/home/tanuj/V2D_challenge/work/branchfit/analysis"; import os; os.makedirs(OUT, exist_ok=True)
SUB = sys.argv[3]; TAG = sys.argv[4]
sys.path.insert(0, KIT)
from v2dlb.mesh_budget import budget_mesh
spec = importlib.util.spec_from_file_location("cdh", KIT + "/metric_code/track_1/PEN.py")
pen = importlib.util.module_from_spec(spec); spec.loader.exec_module(pen)
ts = torch.jit.load(R + "/models/mhr/mhr_model.pt", map_location="cpu").eval()
FLIP = np.array([1.0, -1.0, -1.0])

def mhr(params, shape):
    vs, js = [], []
    with torch.no_grad():
        for a in range(0, len(params), 64):
            p = torch.from_numpy(np.asarray(params[a:a + 64], np.float32))
            z = torch.from_numpy(np.asarray(shape, np.float32).reshape(1, 45)).expand(len(p), -1).contiguous()
            v, sk = ts(z, p, torch.zeros(len(p), 72), True)
            vs.append(v.double().numpy()); js.append(sk[..., :3].double().numpy())
    return np.concatenate(vs) / 100 * FLIP, np.concatenate(js) / 100 * FLIP

v0, _ = mhr(np.zeros((1, 204)), np.zeros(45))
P = np.zeros((54, 204)); P[np.arange(54), np.arange(68, 122)] = 0.3
vm, _ = mhr(P, np.zeros(45))
hand_idx = np.flatnonzero(np.linalg.norm(vm - v0, axis=-1).max(0) > 1e-4)

s = pd.read_parquet(KIT + "/data/track_1_sample_submission.parquet", columns=["row_id"])
pp = s.row_id.str.split("_", expand=True)
key = pd.DataFrame({"ep": pp[1].astype(int), "fr": pp[2].astype(int), "arr": pp[3].astype(int)})
scored = {e: np.sort(g.fr.unique()) for e, g in key[key.arr == 4].groupby("ep")}
eps = [int(x) for x in sys.argv[1].split(",")]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 80
for e in eps:
    t0 = time.time()
    d = np.load(f"{R}/work/{SUB}/episode_{e:06d}.npz")
    _a = f"{R}/work/{SUB}/ASSEMBLE_{e:06d}.json"; asm = json.load(open(_a)) if os.path.exists(_a) else {}
    fr = scored[e]
    mv, mf = budget_mesh(f"{R}/work/{SUB}/episode_{e:06d}_object.ply", 4096, 4096)
    real = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1) > 0
    surf = pen._sample_mesh_surface(mv, mf[real], 2000, 0)
    sc = float(d["object_scale"]); Rr = d["object_rotation"]; tt = d["object_translation"]
    # object translation stats over ALL scored frames
    z = tt[fr, 2]
    br = np.flatnonzero(np.diff(fr) != 1) + 1; edges = [0, *br.tolist(), len(fr)]
    acc = np.concatenate([np.linalg.norm(tt[fr[a:b]][2:] - 2 * tt[fr[a:b]][1:-1] + tt[fr[a:b]][:-2], axis=1) for a, b in zip(edges[:-1], edges[1:])]) * 100
    sub = np.unique(np.concatenate([[fr[0]], fr[np.linspace(0, len(fr) - 1, min(NF, len(fr))).round().astype(int)]]))
    params = np.concatenate([d["pose"][sub], np.repeat(d["scales"][None], len(sub), 0)], 1)
    V, J = mhr(params, d["shape"])
    rows = []
    hands_all = V[:, hand_idx]
    local = np.einsum("tpi,tij->tpj", hands_all - tt[sub][:, None], Rr[sub]) / sc
    depth = pen._penetration_depth(local.reshape(-1, 3), mv, mf).reshape(local.shape[:2]) * sc  # metres (pre-alignment)
    for k, f in enumerate(sub):
        O = sc * surf @ Rr[f].T + tt[f]
        tree = cKDTree(V[k]); dmin = tree.query(O)[0].min()
        dh = cKDTree(hands_all[k]).query(O)[0].min()
        hk = hands_all[k]; c = O.mean(0); nh = hk[np.argmin(np.linalg.norm(hk - c, axis=1))]
        rows.append((int(f), float(dmin), float(dh), float(O[:, 2].mean() - V[k][:, 2].mean()),
                     float(np.linalg.norm(O.mean(0) - J[k, 1])), float(np.linalg.norm(c - nh)), float(c[2] - nh[2]), float(c[2] - J[k, 1, 2])))
    rows = np.array(rows)
    pf = depth.mean(1) * 100  # cm per frame (mean over all hand verts)
    res = dict(ep=e, n_scored=len(fr), n_sub=len(sub), first=int(fr[0]),
               first_dmin_body_cm=rows[0, 1] * 100, first_dmin_hand_cm=rows[0, 2] * 100, first_dz_obj_minus_human_m=rows[0, 3],
               first_obj_pelvis_m=rows[0, 4],
               dmin_body_cm_median=float(np.median(rows[:, 1]) * 100), dmin_body_cm_p90=float(np.percentile(rows[:, 1], 90) * 100),
               dmin_body_cm_max=float(rows[:, 1].max() * 100),
               frac_dmin_gt30cm=float((rows[:, 1] > 0.30).mean()), frac_dmin_gt100cm=float((rows[:, 1] > 1.0).mean()),
               dmin_hand_cm_median=float(np.median(rows[:, 2]) * 100),
               dz_median_m=float(np.median(rows[:, 3])),
               c_hand_cm_median=float(np.median(rows[:, 5]) * 100), c_hand_dz_cm_median=float(np.median(rows[:, 6]) * 100),
               c_pel_dz_cm_median=float(np.median(rows[:, 7]) * 100), first_c_hand_cm=rows[0, 5] * 100, first_c_pel_dz_cm=rows[0, 7] * 100,
               frac_dhand_lt5cm=float((rows[:, 2] < 0.05).mean()), dz_max_m=float(rows[:, 3].max()),
               obj_depth_scored_m=[float(z.min()), float(np.median(z)), float(z.max())],
               human_depth_sub_m=[float(V[:, :, 2].mean(1).min()), float(V[:, :, 2].mean(1).max())],
               obj_acc_scored=[float(acc.mean()), float(np.percentile(acc, 95)), float(acc.max())],
               pen_proxy_cm=float(pf.mean()), pen_frames_any_inside=float((depth > 0).any(1).mean()),
               pen_inside_frac=float((depth > 0).mean()), pen_max_depth_cm=float(depth.max() * 100),
               pen_frame_max_cm=float(pf.max()),
               worst_far=[(int(r[0]), round(float(r[1]) * 100, 1)) for r in rows[np.argsort(-rows[:, 1])[:4]]],
               asm={k: asm.get(k) for k in ("object_source", "s6_mean_iou", "low_confidence_track", "far_from_subject_frames",
                                            "far_limit_m", "invalid_object_frames", "static_frac", "rot_segments", "t_sigma")},
               secs=round(time.time() - t0, 1))
    json.dump(res, open(f"{OUT}/{TAG}_ep{e:02d}.json", "w"), indent=1, default=float)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
