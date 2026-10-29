"""Prediction-only surface visibility for the diagnostic's fixed human mesh."""
import numpy as np
from numba import njit
from scipy import ndimage
from scipy.spatial import cKDTree


@njit(cache=True)
def raster_depth(vertices, faces, K, height, width):
    zbuf = np.full((height, width), np.inf)
    facebuf = np.full((height, width), -1, dtype=np.int64)
    for fi in range(len(faces)):
        face = faces[fi]
        p = vertices[face]
        if min(p[0, 2], p[1, 2], p[2, 2]) <= 1e-3:
            continue
        uv = np.empty((3, 2))
        for k in range(3):
            uv[k, 0] = (K[0, 0]*p[k, 0]+K[0, 1]*p[k, 1]+K[0, 2]*p[k, 2])/p[k, 2]
            uv[k, 1] = (K[1, 0]*p[k, 0]+K[1, 1]*p[k, 1]+K[1, 2]*p[k, 2])/p[k, 2]
        lo_x = max(0, int(np.ceil(min(uv[0, 0], uv[1, 0], uv[2, 0]))))
        hi_x = min(width-1, int(np.floor(max(uv[0, 0], uv[1, 0], uv[2, 0]))))
        lo_y = max(0, int(np.ceil(min(uv[0, 1], uv[1, 1], uv[2, 1]))))
        hi_y = min(height-1, int(np.floor(max(uv[0, 1], uv[1, 1], uv[2, 1]))))
        den = ((uv[1, 1]-uv[2, 1])*(uv[0, 0]-uv[2, 0]) +
               (uv[2, 0]-uv[1, 0])*(uv[0, 1]-uv[2, 1]))
        if abs(den) < 1e-12:
            continue
        for y in range(lo_y, hi_y+1):
            for x in range(lo_x, hi_x+1):
                b0 = ((uv[1, 1]-uv[2, 1])*(x-uv[2, 0]) + (uv[2, 0]-uv[1, 0])*(y-uv[2, 1]))/den
                b1 = ((uv[2, 1]-uv[0, 1])*(x-uv[2, 0]) + (uv[0, 0]-uv[2, 0])*(y-uv[2, 1]))/den
                b2 = 1-b0-b1
                if min(b0, b1, b2) < -1e-9:
                    continue
                iz = b0/p[0, 2]+b1/p[1, 2]+b2/p[2, 2]
                if iz <= 0:
                    continue
                z = 1/iz
                if z < zbuf[y, x]:
                    zbuf[y, x] = z
                    facebuf[y, x] = fi
    return zbuf, facebuf


def visible_pairs(vertices, joints, local, confidence, A, Krgb, objmask,
                  faces, torso_positions, max_points=512):
    """Ray-intersect the predicted mesh; observed clothing/masks remain uncertain."""
    v = np.asarray(vertices, np.float64)
    km = np.asarray(A @ Krgb, np.float64)
    z, face_id = raster_depth(v, np.asarray(faces, np.int64), km, *local.shape[:2])
    nearest = cKDTree(joints).query(v, workers=1)[1]
    torso_vertex = np.isin(nearest, torso_positions)
    torso_face = torso_vertex[faces].sum(axis=1) >= 2
    supported = face_id >= 0
    torso = supported.copy()
    torso[supported] = torso_face[face_id[supported]]
    torso = ndimage.binary_erosion(torso, iterations=3)
    allowed = (torso & ~ndimage.binary_dilation(objmask, iterations=3) &
               np.isfinite(local).all(axis=-1) & (local[..., 2] > 0) &
               np.isfinite(confidence) & (confidence > 0))
    ys, xs = np.nonzero(allowed)
    if len(xs):
        keep = confidence[ys, xs] >= np.median(confidence[ys, xs])
        ys, xs = ys[keep], xs[keep]
        ix = np.linspace(0, len(xs)-1, min(len(xs), max_points)).round().astype(int)
        ys, xs = ys[ix], xs[ix]
    pixels = np.stack([xs, ys], axis=1)
    ray = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(km).T
    targets = ray*z[ys, xs, None]
    return (local[ys, xs].astype(float), targets, pixels,
            dict(body_pixels=len(pixels), rendered_body_pixels=int(supported.sum()),
                 visibility='frontmost perspective triangle surface of exact submitted MAIN prediction; not observed clothing/GT segmentation'))


def preflight():
    K = np.array([[100., 0, 64], [0, 100., 64], [0, 0, 1.]])
    # Parallel front/back squares: a sparse vertex test need not see the front
    # at an interior pixel, whereas the complete triangle surface must.
    front = np.array([[-.4, -.4, 2], [.4, -.4, 2], [.4, .4, 2], [-.4, .4, 2]])
    back = front.copy(); back[:, :2] *= 1.5; back[:, 2] = 3
    vertices = np.r_[front, back]
    faces = np.array([[0,1,2], [0,2,3], [4,6,5], [4,7,6]])
    z, f = raster_depth(vertices, faces, K, 128, 128)
    assert abs(z[64,64]-2) < 1e-12 and f[64,64] < 2
    # Perspective depth is inverse-depth interpolation, not screen-linear Z.
    tri = np.array([[-.4,-.4,1.], [.8,-.8,2.], [0,1.2,3.]])
    z, _ = raster_depth(tri, np.array([[0,1,2]]), K, 128, 128)
    n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    expected = float(n@tri[0]/n[2])
    perspective_error = abs(z[64,64]-expected)
    assert perspective_error < 1e-12
    # Face order/winding cannot expose a back surface.
    zr, _ = raster_depth(vertices, faces[::-1, ::-1].copy(), K, 128, 128)
    assert abs(zr[64,64]-2) < 1e-12
    return dict(front_back_surface_test=True, face_order_winding_invariant=True,
                perspective_plane_depth_error_m=perspective_error,
                native_GT_access=False)
