#!/usr/bin/env python3
"""Joint bounded camera translation from predicted depth, motion and hand tips.

No GT/scorer imports. Rotation, mesh and global human-anchored scale are fixed.
All losses average over the complete trajectory; inactive observations are zero.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

PROTOCOL = dict(outer_iterations=6, line_search_steps=10, surface_samples=4096, surface_seed=271828,
    max_depth_points=256, minimum_depth_points=32, mask_confidence=.8,
    depth_sigma_m=.03, huber_delta_m=.03, initial_depth_surface_gate_m=.25,
    fp_prior_sigma_m=.10, acceleration_sigma_m_s2=5., translation_bound_m=.25,
    contact_sigma_m=.05, contact_weight=.25, contact_hinge_m=.03,
    initial_contact_surface_gate_m=.15, contact_mask_distance_model_px=4.,
    contact_proxy='10 named SMPLH fingertips mapped from predicted MHR; not kit PEN',
    rotation_fixed=True, object_scale_fixed=True)


def acceleration_matrix(times, cadence=None, frame_ids=None, frame_step=1):
    times = np.asarray(times, float); n = len(times); rows = []
    assert times.shape == (n,) and np.isfinite(times).all() and np.all(np.diff(times) > 0)
    ids = np.arange(n) if frame_ids is None else np.asarray(frame_ids)
    assert ids.shape == (n,)
    step = 1 if frame_ids is None else frame_step
    for i in range(1, n-1):
        a, b = times[i]-times[i-1], times[i+1]-times[i]
        # Missing source frames determine gaps. Valid camera timestamps may jitter.
        if ids[i]-ids[i-1] != step or ids[i+1]-ids[i] != step: continue
        row = np.zeros(n); row[i-1] = 2/(a*(a+b)); row[i] = -2/(a*b); row[i+1] = 2/(b*(a+b))
        rows.append(row)
    return np.stack(rows) if rows else np.zeros((0, n))


def bounded_quadratic(diagonal, rhs, base, times, cadence, temporal, frame_ids=None, frame_step=1):
    n = len(base); h = np.diag(diagonal)
    d = acceleration_matrix(times, cadence, frame_ids, frame_step)
    if temporal and len(d): h += (d.T @ d) * n / (len(d)*PROTOCOL['acceleration_sigma_m_s2']**2)
    # L.T @ L=H and L.T @ b=rhs; bounded least squares solves the original quadratic.
    a = cholesky(h, lower=False); result = np.empty_like(base)
    bound = PROTOCOL['translation_bound_m']
    for axis in range(3):
        b = solve_triangular(a.T, rhs[:, axis], lower=True)
        lower = base[:, axis]-bound
        if axis == 2: lower = np.maximum(lower, .05)
        fit = lsq_linear(a, b, bounds=(lower, base[:, axis]+bound), tol=1e-10, max_iter=200)
        assert fit.success and np.isfinite(fit.x).all(), fit.message
        result[:, axis] = fit.x
    return result


def solve(surface, rotations, baseline, clouds, quality, depth_confidence,
          times, cadence, contacts, use_depth, use_temporal, use_contact,
          frame_ids=None, frame_step=1):
    surface = np.asarray(surface, float); baseline = np.asarray(baseline, float)
    n = len(baseline); rotations = np.asarray(rotations, float); tree = cKDTree(surface)
    assert baseline.shape == (n, 3) and rotations.shape == (n, 3, 3)
    assert np.isfinite(surface).all() and np.isfinite(baseline).all() and np.all(baseline[:, 2] > 0)
    trust = np.maximum(np.asarray(quality), .1)/PROTOCOL['fp_prior_sigma_m']**2
    # Freeze admissible observed pixels at the initial pose; later overlap cannot vanish.
    fixed = []
    for i, points in enumerate(clouds):
        points = np.asarray(points, float).reshape(-1, 3)
        distance = tree.query((points-baseline[i]) @ rotations[i], workers=1)[0] if len(points) else np.zeros(0)
        retained = points[distance <= PROTOCOL['initial_depth_surface_gate_m']]
        fixed.append(retained if len(retained) >= PROTOCOL['minimum_depth_points'] else np.empty((0, 3)))
    d2 = acceleration_matrix(times, cadence, frame_ids, frame_step)
    def objective(candidate):
        terms = dict(depth=0., prior=float(.5*np.mean(trust*np.sum((candidate-baseline)**2, axis=1))),
                     temporal=0., contact=0.)
        if use_temporal and len(d2):
            terms['temporal'] = float(.5*np.mean(np.sum((d2@candidate)**2, axis=1))/PROTOCOL['acceleration_sigma_m_s2']**2)
        for i in range(n):
            if use_depth and len(fixed[i]):
                distances = tree.query((fixed[i]-candidate[i]) @ rotations[i], workers=1)[0]
                delta = PROTOCOL['huber_delta_m']
                huber = np.where(distances <= delta, .5*distances**2, delta*(distances-.5*delta))
                terms['depth'] += float(depth_confidence[i]*huber.mean()/PROTOCOL['depth_sigma_m']**2)/n
            hands = contacts.get(i, []) if use_contact else []
            for hand in hands:
                distance = float(tree.query((np.asarray(hand)-candidate[i]) @ rotations[i], workers=1)[0].min())
                excess = max(distance-PROTOCOL['contact_hinge_m'], 0.)
                terms['contact'] += .5*PROTOCOL['contact_weight']*excess**2/(n*len(hands)*PROTOCOL['contact_sigma_m']**2)
        terms['total'] = sum(terms.values())
        assert all(np.isfinite(x) and x >= 0 for x in terms.values())
        return terms
    t = baseline.copy(); history = []; initial_objective = objective(t)
    for iteration in range(PROTOCOL['outer_iterations']):
        diag = trust.copy(); rhs = trust[:, None]*baseline; depth_cost = 0.; contact_cost = 0.; active = 0
        for i in range(n):
            if use_depth and len(fixed[i]):
                p = fixed[i]; local = (p-t[i]) @ rotations[i]; nearest = tree.query(local, workers=1)[1]
                q = surface[nearest] @ rotations[i].T; error = np.linalg.norm(q+t[i]-p, axis=1)
                robust = np.minimum(1., PROTOCOL['huber_delta_m']/np.maximum(error, 1e-12))
                weights = depth_confidence[i]*robust/(len(p)*PROTOCOL['depth_sigma_m']**2)
                diag[i] += weights.sum(); rhs[i] += (weights[:, None]*(p-q)).sum(0)
                delta = PROTOCOL['huber_delta_m']
                huber = np.where(error <= delta, .5*error**2, delta*(error-.5*delta))
                depth_cost += float(depth_confidence[i]*huber.mean()/PROTOCOL['depth_sigma_m']**2)
            hands = contacts.get(i, []) if use_contact else []
            for hand in hands:
                hand = np.asarray(hand, float).reshape(-1, 3)
                distance, ids = tree.query((hand-t[i]) @ rotations[i], workers=1)
                index = int(np.argmin(distance)); q = surface[ids[index]] @ rotations[i].T
                vector = hand[index]-(q+t[i]); length = float(np.linalg.norm(vector))
                excess = max(length-PROTOCOL['contact_hinge_m'], 0.)
                contact_cost += .5*PROTOCOL['contact_weight']*excess**2/(len(hands)*PROTOCOL['contact_sigma_m']**2)
                if excess <= 0: continue
                # Closest predicted tip/surface pair receives a one-sided proximity bound.
                desired = hand[index]-q-PROTOCOL['contact_hinge_m']*vector/length
                weight = PROTOCOL['contact_weight']/(len(hands)*PROTOCOL['contact_sigma_m']**2)
                diag[i] += weight; rhs[i] += weight*desired; active += 1
        new = bounded_quadratic(diag, rhs, baseline, times, cadence, use_temporal, frame_ids, frame_step)
        old_loss = objective(t); accepted_fraction = 0.
        for step in range(PROTOCOL['line_search_steps']):
            fraction = .5**step; candidate = t+fraction*(new-t); candidate_loss = objective(candidate)
            if candidate_loss['total'] <= old_loss['total']+1e-10:
                new = candidate; accepted_fraction = fraction; break
        if accepted_fraction == 0: new = t.copy()
        history.append(dict(iteration=iteration, depth_loss_before_solve=depth_cost/n,
            contact_loss_before_solve=contact_cost/n, active_hand_pairs=active,
            max_update_m=float(np.linalg.norm(new-t, axis=1).max()),
            accepted_step_fraction=accepted_fraction, objective_after=objective(new)))
        t = new
        if accepted_fraction == 0: break
    assert np.isfinite(t).all() and (t[:, 2] > 0).all()
    return t, dict(history=history, initial_objective=initial_objective, final_objective=objective(t),
        frozen_depth_points_per_frame=[len(p) for p in fixed],
        contact_frames=sorted(contacts) if use_contact else [],
        translation_changed_frames=int((np.abs(t-baseline).max(1)>1e-9).sum()),
        max_translation_shift_m=float(np.linalg.norm(t-baseline, axis=1).max()),
        max_axis_bound_used_m=float(np.abs(t-baseline).max()))


def self_test():
    times = np.arange(6)*.1; d = acceleration_matrix(times, .1)
    assert np.allclose(d @ times, 0., atol=1e-10)
    assert np.allclose(d @ (.5*3*times**2), 3., atol=1e-10)
    gap = np.array([0., .1, .3, .4]); assert len(acceleration_matrix(gap, .1, [0, 1, 3, 4])) == 0
    irregular = np.array([0., .031, .067, .101, .145])
    assert np.allclose(acceleration_matrix(irregular, .033) @ (.5*3*irregular**2), 3.)
    xx, yy = np.meshgrid(np.linspace(-.1, .1, 9), np.linspace(-.1, .1, 9))
    q = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    base = np.tile([0., 0., 1.1], (3, 1)); r = np.tile(np.eye(3), (3, 1, 1)); p = [q+[0, 0, 1.]]*3
    kwargs = dict(surface=q, rotations=r, baseline=base, clouds=p, quality=np.ones(3),
        depth_confidence=np.ones(3), times=np.arange(3)*.1, cadence=.1, contacts={})
    fitted, _ = solve(**kwargs, use_depth=True, use_temporal=False, use_contact=False)
    assert np.max(abs(fitted[:, 2]-1.)) < .01
    unchanged, _ = solve(**kwargs, use_depth=False, use_temporal=False, use_contact=False)
    assert np.allclose(unchanged, base, atol=1e-12)
    plain, _ = solve(**kwargs, use_depth=True, use_temporal=True, use_contact=False)
    empty, _ = solve(**kwargs, use_depth=True, use_temporal=True, use_contact=True)
    assert np.array_equal(plain, empty)
    return dict(known_plane_depth_recovery_m=float(fitted[0, 2]),
        acceleration_units_and_gaps=True, no_observation_identity=True,
        empty_contact_bit_identical=True, depth_rotation_and_scale_fixed=True)


if __name__ == '__main__': print(self_test())
