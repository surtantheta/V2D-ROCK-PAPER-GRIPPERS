"""Bounded nearest-SO(3) repair of stored floating-point rotation drift."""
import numpy as np

TOLERANCE = 1e-4


def project(R):
    value = np.asarray(R, dtype=np.float64)
    if value.ndim not in (2, 3) or value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError('finite 3x3 or Nx3x3 rotations required')
    batched = value.reshape(-1, 3, 3)
    orth = np.max(np.abs(batched@batched.swapaxes(-1, -2)-np.eye(3)), axis=(1, 2))
    determinant = np.linalg.det(batched)
    if np.any(determinant <= 0) or np.any(orth > TOLERANCE) or np.any(abs(determinant-1) > TOLERANCE):
        raise ValueError('rotation error exceeds bounded precision repair or contains reflection')
    left, _, right = np.linalg.svd(batched)
    repaired = left@right
    if np.any(np.linalg.det(repaired) <= 0):
        raise ValueError('nearest rotation unexpectedly contains reflection')
    correction = repaired-batched
    record = dict(
        tolerance=TOLERANCE,
        frames=len(batched),
        orthogonality_error_before=orth.tolist(),
        determinant_before=determinant.tolist(),
        Frobenius_correction=np.linalg.norm(correction, axis=(1, 2)).tolist(),
        orthogonality_error_after=np.max(abs(repaired@repaired.swapaxes(-1, -2)-np.eye(3)), axis=(1, 2)).tolist(),
        determinant_after=np.linalg.det(repaired).tolist(),
        max_Frobenius_correction=float(np.linalg.norm(correction, axis=(1, 2)).max()),
        input_mutated=False,
        GT_used=False,
        interpretation='Only bounded numerical SO3 projection; object size, translation, mesh and human are unchanged here')
    return repaired.reshape(value.shape), record
