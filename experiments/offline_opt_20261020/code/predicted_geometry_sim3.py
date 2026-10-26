"""Positive proper Sim3 alignment of two generated point clouds; NumPy/SciPy only.

No file I/O, camera metadata, scorer imports, labels, or benchmark references.
Returns source-to-target map: source_points @ R.T * scale + t.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
ITERATIONS=60

def proper_pca_starts(x,y):
    def basis(p):
        _,v=np.linalg.eigh(np.cov(p.T));v=v[:,::-1]
        if np.linalg.det(v)<0:v[:,-1]*=-1
        return v
    px,py=basis(x),basis(y)
    out=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.,1.],repeat=3):
            a=np.eye(3)[:,perm]@np.diag(signs)
            if np.linalg.det(a)>.99:out.append(py@a@px.T)
    assert len(out)==24
    return out

def fit_sim3(x,y):
    mx,my=x.mean(0),y.mean(0);a=x-mx;b=y-my
    u,d,vt=np.linalg.svd(a.T@b)
    signs=np.ones(3);signs[-1]=1 if np.linalg.det(vt.T@u.T)>0 else -1
    r=vt.T@np.diag(signs)@u.T
    scale=float(np.clip(np.dot(d,signs)/np.square(a).sum(),.25,4.))
    t=my-scale*mx@r.T
    return scale,r,t

def align_clouds(pred, target):
    cp,cg=pred.mean(0),target.mean(0)
    rp=float(np.sqrt(np.square(pred-cp).sum(1).mean()));rg=float(np.sqrt(np.square(target-cg).sum(1).mean()))
    if min(rp,rg)<1e-8:raise ValueError('collapsed shape')
    x,y=(pred-cp)/rp,(target-cg)/rg
    target_tree=cKDTree(y);solutions=[]
    for index,r in enumerate(proper_pca_starts(x,y)):
        scale,t=1.,np.zeros(3);previous=float('inf');best=None
        for step in range(ITERATIONS):
            posed=scale*x@r.T+t
            da,ia=target_tree.query(posed,workers=1)
            db,ib=cKDTree(posed).query(y,workers=1)
            loss=float(np.square(da).mean()+np.square(db).mean())
            if best is None or loss<best[0]:best=(loss,scale,r.copy(),t.copy(),step)
            if abs(previous-loss)<1e-10:break
            previous=loss
            scale,r,t=fit_sim3(np.concatenate([x,x[ib]]),np.concatenate([y[ia],y]))
        loss,scale,r,t,last=best
        # Convert normalized-space map back to original object coordinates.
        s=scale*rg/rp;translation=cg+rg*t-s*cp@r.T
        solutions.append({'start':index,'squared_normalized_fit_objective':loss,'scale':s,'R':r.tolist(),'t':translation.tolist(),
                          'normalized_scale':scale,'iterations':last+1,'scale_bound_active':abs(scale-.25)<1e-8 or abs(scale-4)<1e-8})
    selected=min(solutions,key=lambda a:a['squared_normalized_fit_objective'])
    return selected,solutions,rg
