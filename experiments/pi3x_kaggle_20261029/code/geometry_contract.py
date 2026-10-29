"""Pure numerical camera contract shared by the read-only pilot, never a GT fit."""
import numpy as np

def resize_map(source_wh,target_wh,crop_top=0):
    w,h=source_wh; tw,th=target_wh
    # Official model K convention: continuous pixel-coordinate scale. Sampling
    # uses PIL's centre-based interpolation; retain that distinction in metadata.
    return np.array([[tw/w,0,0],[0,th/h,-crop_top],[0,0,1]],dtype=np.float64)

def unproject(depth,K):
    n,h,w=depth.shape
    yy,xx=np.mgrid[:h,:w]
    grid=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
    return np.einsum('nij,hwj->nhwi',np.linalg.inv(K),grid)*depth[...,None]

def estimate_pinhole(local):
    n,h,w,_=local.shape; yy,xx=np.mgrid[:h,:w]; out=[]; errors=[]
    for p in local:
        z=p[...,2];valid=np.isfinite(p).all(-1)&(z>1e-6)
        # Fixed deterministic spatial subsample, no confidence or GT selection.
        valid[1::2]=False;valid[:,1::2]=False
        xn=p[...,0][valid]/z[valid];yn=p[...,1][valid]/z[valid]
        fx,cx=np.linalg.lstsq(np.stack([xn,np.ones_like(xn)],1),xx[valid],rcond=None)[0]
        fy,cy=np.linalg.lstsq(np.stack([yn,np.ones_like(yn)],1),yy[valid],rcond=None)[0]
        out.append([[fx,0,cx],[0,fy,cy],[0,0,1]])
        e=np.sqrt((fx*xn+cx-xx[valid])**2+(fy*yn+cy-yy[valid])**2)
        errors.append({'median_px':float(np.median(e)),'p95_px':float(np.quantile(e,.95))})
    return np.array(out),errors

def as_w2c(extrinsic):
    e=np.asarray(extrinsic,dtype=np.float64)
    if e.shape[-2:]==(3,4):
        bottom=np.broadcast_to(np.array([0,0,0,1.]),(len(e),1,4))
        e=np.concatenate([e,bottom],axis=1)
    assert e.shape[-2:]==(4,4)
    return e

def validate(depth,K,w2c):
    assert depth.ndim==3 and len(depth)==16
    assert K.shape==(16,3,3) and w2c.shape==(16,4,4)
    assert np.isfinite(K).all() and np.isfinite(w2c).all()
    assert (K[:,0,0]>0).all() and (K[:,1,1]>0).all()
    r=w2c[:,:3,:3];err=np.max(np.abs(r@r.transpose(0,2,1)-np.eye(3)))
    assert err<.01 and np.min(np.linalg.det(r))>.99,(err,np.linalg.det(r))
    valid=np.isfinite(depth)&(depth>0)
    fractions=valid.mean((1,2));assert np.min(fractions)>.5,fractions
    return {'positive_finite_depth_fraction_per_frame':fractions.tolist(),
            'rotation_orthogonality_max_error':float(err),'rotation_determinant_min':float(np.linalg.det(r).min())}

def self_test():
    k=np.array([[[103,0,11.2],[0,97,7.6],[0,0,1.]]]);d=np.ones((1,17,23))*2.5
    p=unproject(d,k);fit,err=estimate_pinhole(p)
    assert np.max(np.abs(fit-k))<1e-10
    a=resize_map((1440,1080),(504,378))
    crop=np.array([[1,0,-240],[0,1,0],[0,0,1.]])
    kn=np.array([[1000.,0,960],[0,1000,540],[0,0,1]])
    point=np.array([.3,.4,2.]);native=kn@point;native/=native[2]
    lhs=a@crop@native;rhs=(a@crop@kn)@point;rhs/=rhs[2]
    error=float(np.max(np.abs(lhs-rhs)));assert error<1e-10
    return {'pinhole_recovery_max_error':float(np.max(np.abs(fit-k))),
            'anisotropic_crop_projection_max_error':error,'source_pixel_convention':'official integer model grids'}

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
