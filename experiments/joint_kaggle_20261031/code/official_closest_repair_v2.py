"""Rare-branch nearest-point repair consistent with the official SDF oracle.

Keep the original Trimesh path verbatim whenever its original 1e-7 check passes.
Only a row that would previously fail that check uses the official Ericson
region arithmetic to recover the point associated with the oracle's distance.
No distance tolerance, objective, solver, regularizer or admission gate changes.
"""
import ast
import hashlib
import inspect
import textwrap

import numpy as np


def official_point_kernel(pen):
    import numba
    distance = pen._PENETRATION_KERNELS[1]
    function = distance.py_func
    closure = dict(zip(function.__code__.co_freevars,
                       [cell.cell_contents for cell in function.__closure__]))
    original = closure['closest_sq'].py_func
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    node = tree.body[0]
    assert node.name == 'closest_sq'
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    expected = ast.parse('(px - q0) ** 2 + (py - q1) ** 2 + (pz - q2) ** 2', mode='eval').body
    assert ast.dump(returns[0].value) == ast.dump(expected)
    node.name = 'official_point_with_same_region_arithmetic'
    returns[0].value = ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load()) for x in ['q0', 'q1', 'q2']], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    namespace = dict(numba=numba, np=np)
    exec(compile(tree, '<official-distance-point-return-only>', 'exec'), namespace)
    point = namespace[node.name]

    @numba.njit(cache=False)
    def nearest(query, v0, v1, v2):
        best = np.inf
        result = np.zeros(3)
        selected = -1
        for face in range(len(v0)):
            q0, q1, q2 = point(query[0], query[1], query[2],
                               v0[face, 0], v0[face, 1], v0[face, 2],
                               v1[face, 0], v1[face, 1], v1[face, 2],
                               v2[face, 0], v2[face, 1], v2[face, 2])
            squared = (query[0] - q0) ** 2 + (query[1] - q1) ** 2 + (query[2] - q2) ** 2
            if squared < best:
                best = squared
                result[0], result[1], result[2] = q0, q1, q2
                selected = face
        return result, best, selected

    return nearest, dict(official_closest_squared_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         transform='only function name and final squared-distance return replaced by closest-point tuple',
                         entire_triangle_pool=True, official_distance_oracle_unchanged=True)


def install(core, pen):
    import scipy.sparse
    original_class = core.SignedMesh
    original_source = inspect.getsource(original_class.linearization)
    target = "            assert abs(norm*self.scale-abs(signed[i,k]))<1e-7,('distance-gradient mismatch',i,k)"
    assert original_source.count(target) == 1
    replacement = """            if not abs(norm*self.scale-abs(signed[i,k]))<1e-7:
                q,closest,vec,norm = self.repair_official_point(i,k,t0,delta,signed,q,closest,norm)
            assert abs(norm*self.scale-abs(signed[i,k]))<1e-7,('distance-gradient mismatch',i,k)"""
    changed = textwrap.dedent(original_source.replace(target, replacement))
    namespace = dict(np=np, sparse=scipy.sparse)
    exec(compile(changed, '<rare-official-point-gradient-repair>', 'exec'), namespace)
    nearest, oracle_proof = official_point_kernel(pen)
    events = []

    def repair(self, i, k, t0, delta, signed, old_query, old_closest, old_norm):
        # Match both grouping and batched matmul of SignedMesh.signed exactly.
        local_frame = (self.hands[i] - (t0[i] + delta[i])) @ self.rot[i] / self.scale
        query = np.ascontiguousarray(local_frame[k])
        closest, squared, face = nearest(query, self.v0, self.v1, self.v2)
        vector = query - closest
        norm = float(np.linalg.norm(vector))
        assert np.isfinite(closest).all() and np.isfinite(norm) and face >= 0
        assert norm >= 1e-10, ('repaired surface gradient undefined', i, k)
        assert abs(norm * self.scale - abs(signed[i, k])) < 1e-7, ('official closest-point replay mismatch', i, k)
        events.append(dict(frame=int(i), hand_point=int(k),
                           old_distance_m=float(old_norm * self.scale),
                           oracle_distance_m=float(abs(signed[i, k])),
                           repaired_distance_m=float(norm * self.scale),
                           original_discrepancy_m=float(abs(old_norm * self.scale - abs(signed[i, k]))),
                           canonical_query_difference_model_units=float(np.linalg.norm(query - old_query)),
                           official_triangle_index=int(face)))
        return query, closest, vector, norm

    repaired_class = type('RareOfficialClosestRepairSignedMesh', (original_class,),
                          dict(linearization=namespace['linearization'], repair_official_point=repair))
    core.SignedMesh = repaired_class
    proof = dict(schema='official-closest-rare-branch-repair-v2',
                 original_linearization_source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                 patched_linearization_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                 unchanged_threshold_scene_m=1e-7,
                 branch_condition='exact logical negation of the former failing assertion',
                 original_fast_path_unchanged=True,
                 successful_V1_reuse_proof='Every completed V1 fit passed each old assertion; the new branch would never execute, leaving every evaluated numeric expression and iterate unchanged.',
                 objective_oracle_policy_config_unchanged=True,
                 point_kernel=oracle_proof)
    return proof, events
