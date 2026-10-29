"""Fixed-camera real mesh refinement; rendered ICP hypotheses, never measured tracks.

OpenCV metres, integer pixel centres. X=s*(V-mu)@R.T+C. No files or GT.
The outer renderer is exact opaque triangle frontmost visibility. Inner depth/
silhouette terms use frozen surface correspondences; they are an alternating
surrogate, NOT an exact differentiable rasterizer or a global optimum.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


@njit(cache=True)
def raster(vertices, faces, K, height, width):
    """Two-sided opaque triangles, perspective-correct barycentrics and Z."""
    zbuf=np.full((height,width),np.inf)
    fbuf=np.full((height,width),-1,np.int64)
    bary=np.zeros((height,width,3))
    for fi in range(len(faces)):
        p=vertices[faces[fi]]
        if min(p[:,2]) <= .001: continue
        uv=np.empty((3,2))
        for k in range(3):
            h=K@p[k];uv[k]=h[:2]/h[2]
        den=(uv[1,1]-uv[2,1])*(uv[0,0]-uv[2,0])+(uv[2,0]-uv[1,0])*(uv[0,1]-uv[2,1])
        if abs(den)<1e-12: continue
        for y in range(max(0,int(np.ceil(min(uv[:,1])))),min(height-1,int(np.floor(max(uv[:,1]))))+1):
            for x in range(max(0,int(np.ceil(min(uv[:,0])))),min(width-1,int(np.floor(max(uv[:,0]))))+1):
                b0=((uv[1,1]-uv[2,1])*(x-uv[2,0])+(uv[2,0]-uv[1,0])*(y-uv[2,1]))/den
                b1=((uv[2,1]-uv[0,1])*(x-uv[2,0])+(uv[0,0]-uv[2,0])*(y-uv[2,1]))/den
                b2=1-b0-b1
                if min(b0,b1,b2)<-1e-9: continue
                b=np.array([b0/p[0,2],b1/p[1,2],b2/p[2,2]])
                iz=b.sum()
                if iz>0 and 1/iz<zbuf[y,x]:
                    zbuf[y,x]=1/iz;fbuf[y,x]=fi;bary[y,x]=b/iz
    return zbuf,fbuf,bary


def proper(R):
    R=np.asarray(R,float)
    if R.shape[-2:]!=(3,3) or not np.isfinite(R).all() or not np.allclose(R@R.swapaxes(-1,-2),np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(R),1,atol=1e-6):
        raise ValueError('proper rotations required; scale/reflection must not be baked into R')
    return R


def area_centroid(V,F):
    t=V[F];a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    if not np.isfinite(a).all() or a.sum()<=0: raise ValueError('positive mesh area required')
    return np.average(t.mean(1),axis=0,weights=a)


def camera(V,R,t,s): return s*(V@R.T)+t


def project(X,K):
    h=X@K.T
    return h[:,:2]/np.maximum(h[:,2,None],.001)


def take(n,budget):
    return np.unique(np.rint(np.linspace(0,n-1,min(n,budget))).astype(int)) if n else np.zeros(0,int)


def normalized_huber(value,sigma=1.,weight=1.):
    """2*Huber cost averaged by scalar count, robustification BEFORE normalization.

    Duplicating a residual group exactly does not change its squared cost.
    Frame/group weights are independent of mesh/observation counts.
    """
    x=np.asarray(value,float).reshape(-1)/sigma
    if not len(x): return x
    a=np.abs(x)
    return np.sign(x)*np.sqrt(np.where(a<=1,x*x,2*a-1))*np.sqrt(weight/len(x))


@dataclass(frozen=True)
class Config:
    outer_iterations:int=3
    max_nfev:int=35
    surface_budget:int=384
    observation_budget:int=256
    depth_sigma_m:float=.03
    silhouette_sigma_px:float=2.
    depth_weight:float=1.
    coverage_weight:float=.5
    free_space_weight:float=.5
    centroid_prior_sigma_m:float=.15
    rotation_prior_sigma_rad:float=.35
    log_scale_prior_sigma:float=.25
    correction_accel_sigma_m_s2:float=3.
    angular_accel_sigma_rad_s2:float=12.
    temporal_weight:float=.1
    scale_bounds:tuple=(.7,1.4)
    centroid_bound_m:float=.5
    minimum_z_m:float=.02
    min_depth_points:int=16


def associations(V,F,R,t,s,K,obs,cfg):
    """Render first; return finite image-supported hypotheses without 3D-distance admission."""
    mask=np.asarray(obs['mask'],bool);H,W=mask.shape
    if not obs.get('enabled',True) or not mask.any():
        # Unknown/absent evidence is not an observed empty image. In particular,
        # it must not pull a mesh toward a predicted human occluder.
        return dict(depth_local=np.zeros((0,3)),cloud=np.zeros((0,3)),coverage_local=np.zeros((0,3)),positive_uv=np.zeros((0,2)),
                    visible_local=np.zeros((0,3)),distance=np.zeros((H,W)),rendered_pixels=0,visible_pixels=0,
                    unknown_human_occluded_pixels=0,finite_cloud_points=0,correspondence_type='disabled: no depth/coverage/free-space data')
    z,fi,b=raster(camera(V,R,t,s),F,K,H,W)
    human=np.asarray(obs.get('human_z',np.full((H,W),np.inf)),float)
    if human.shape!=(H,W): raise ValueError('human_z grid differs')
    # Positive object labels take precedence over imperfect body overlap. Else
    # a rendered object behind the predicted human is unknown, not free space.
    occ=np.isfinite(human)&(human+.01<z)&~mask
    visible=(fi>=0)&~occ
    yy,xx=np.nonzero(visible);sample=take(len(xx),cfg.surface_budget);yy,xx=yy[sample],xx[sample]
    local=np.einsum('ni,nij->nj',b[yy,xx],V[F[fi[yy,xx]]]) if len(xx) else np.zeros((0,3))
    rendered=camera(local,R,t,s)
    pos=np.argwhere(mask)[:,::-1].astype(float);pos=pos[take(len(pos),cfg.observation_budget)]
    cloud=np.asarray(obs.get('cloud',np.zeros((0,3))),float)
    if cloud.ndim!=2 or cloud.shape[1]!=3: raise ValueError('cloud shape')
    valid=np.isfinite(cloud).all(1)&(cloud[:,2]>.02)
    cloud=cloud[valid];cloud=cloud[take(len(cloud),cfg.observation_budget)]
    depth_local=np.zeros((0,3));coverage_local=np.zeros((0,3))
    if len(local):
        if len(cloud):depth_local=local[cKDTree(rendered).query(cloud,workers=1)[1]]
        if len(pos):coverage_local=local[cKDTree(np.c_[xx,yy]).query(pos,workers=1)[1]]
    else:
        cloud=cloud[:0];pos=pos[:0]
    # Occlusion mask is recomputed from the current exact renderer each outer
    # iteration and frozen within its inner solve. It never fills mask holes.
    allowed=mask|occ
    distance=ndimage.distance_transform_edt(~allowed) if allowed.any() else np.full(mask.shape,float(max(H,W)))
    return dict(depth_local=depth_local,cloud=cloud,coverage_local=coverage_local,
                positive_uv=pos,visible_local=local,distance=distance,
                rendered_pixels=int((fi>=0).sum()),visible_pixels=int(visible.sum()),
                unknown_human_occluded_pixels=int(((fi>=0)&occ).sum()),
                finite_cloud_points=int(valid.sum()),
                correspondence_type='outer-iteration nearest rendered frontmost surface hypotheses; NOT persistent tracks')


def render_diagnostics(V,F,R,t,s,K,obs):
    mask=np.asarray(obs['mask'],bool);z,face,_=raster(camera(V,R,t,s),F,K,*mask.shape)
    hz=np.asarray(obs.get('human_z',np.full(mask.shape,np.inf)))
    unknown=np.isfinite(hz)&(hz+.01<z)&~mask
    rendered=(face>=0)&~unknown
    union=(rendered|mask).sum();inter=(rendered&mask).sum()
    return dict(visible_raster_IoU=float(inter/max(1,union)),positive_coverage=float(inter/max(1,mask.sum())),
                visible_false_positive_fraction=float((rendered&~mask).sum()/max(1,rendered.sum())),
                rendered_pixels=int((face>=0).sum()),human_occluded_pixels=int(((face>=0)&unknown).sum()))


def solve(V,F,R0,t0,s0,K,times,observations,config=None):
    cfg=Config() if config is None else config
    V=np.asarray(V,float);F=np.asarray(F,np.int64);R0=proper(R0);t0=np.asarray(t0,float)
    K=np.asarray(K,float);times=np.asarray(times,float);n=len(R0)
    if V.ndim!=2 or V.shape[1]!=3 or not np.isfinite(V).all():raise ValueError('finite mesh vertices required')
    if F.ndim!=2 or F.shape[1]!=3 or F.min()<0 or F.max()>=len(V):raise ValueError('mesh faces')
    if t0.shape!=(n,3) or not np.isfinite(t0).all() or len(observations)!=n or times.shape!=(n,) or not np.all(np.diff(times)>0):raise ValueError('frame inputs/timestamps')
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):raise ValueError('fixed OpenCV K')
    if not np.isscalar(s0) or not np.isfinite(s0) or s0<=0:raise ValueError('one shared positive scale required')
    if cfg.outer_iterations<1 or cfg.max_nfev<1 or not 0<cfg.scale_bounds[0]<1<cfg.scale_bounds[1]:raise ValueError('invalid configuration')
    positive=[cfg.depth_sigma_m,cfg.silhouette_sigma_px,cfg.centroid_prior_sigma_m,cfg.rotation_prior_sigma_rad,cfg.log_scale_prior_sigma,cfg.correction_accel_sigma_m_s2,cfg.angular_accel_sigma_rad_s2,cfg.centroid_bound_m,cfg.minimum_z_m]
    if not np.isfinite(positive).all() or min(positive)<=0:raise ValueError('configuration sigmas/bounds must be positive')
    mu=area_centroid(V,F);C0=t0+s0*np.einsum('fij,j->fi',R0,mu)
    enabled=[bool(o.get('enabled',True)) and np.asarray(o['mask']).any() for o in observations]
    effective=[{**o,'enabled':e} for o,e in zip(observations,enabled)]
    admitted_metric_counts=[]
    for e,o in zip(enabled,observations):
        cloud=np.asarray(o.get('cloud',np.zeros((0,3))),float)
        if cloud.ndim!=2 or cloud.shape[1]!=3:raise ValueError('cloud shape')
        admitted_metric_counts.append(int((np.isfinite(cloud).all(1)&(cloud[:,2]>.02)).sum()) if e else 0)
    metric=any(k>=cfg.min_depth_points for k in admitted_metric_counts)
    result=dict(R=R0.copy(),t=t0.copy(),scale=float(s0),fallback=None,diagnostics={'config':asdict(cfg),'metric_anchor_prior_present':metric,
                'monocular_absolute_scale_gauge_unanchored':not metric,'full_scale_observability_or_unbiasedness_certified':False,
                'finite_positive_metric_anchor_counts':admitted_metric_counts,
                'metric_anchor_is_predicted_not_GT':True,'enabled_observation_frames':sum(enabled),'contacts_enabled':False,'collision_enabled':False})
    if not any(enabled):
        result['fallback']='no_observations';return result
    # Perturb rotations around baseline and optimize physical area-centroids.
    x=np.zeros(1+6*n);x[1+3*n:]=C0.ravel()
    lo=np.full_like(x,-np.inf);hi=np.full_like(x,np.inf)
    lo[0]=np.log(cfg.scale_bounds[0]) if metric else -1e-12
    hi[0]=np.log(cfg.scale_bounds[1]) if metric else 1e-12
    lo[1:1+3*n]=-.6;hi[1:1+3*n]=.6
    lo[1+3*n:]=(C0-cfg.centroid_bound_m).ravel();hi[1+3*n:]=(C0+cfg.centroid_bound_m).ravel()
    def unpack(p):
        s=float(s0*np.exp(p[0]));R=Rotation.from_rotvec(p[1:1+3*n].reshape(n,3)).as_matrix()@R0
        C=p[1+3*n:].reshape(n,3);t=C-s*np.einsum('fij,j->fi',R,mu)
        return R,t,s,C
    history=[]
    for outer in range(cfg.outer_iterations):
        R,t,s,C=unpack(x)
        pair=[associations(V,F,R[f],t[f],s,K,effective[f],cfg) for f in range(n)]
        if outer==0 and not any(len(a['visible_local']) for a in pair):
            result['fallback']='no_rendered_surface_support'
            result['diagnostics']['initial_rendered_surface_counts']=[len(a['visible_local']) for a in pair]
            return result
        def residual(p):
            R,t,s,C=unpack(p);groups=[]
            for f,a in enumerate(pair):
                # Each frame has equal declared weight, irrespective of count.
                if len(a['depth_local']):
                    groups.append(normalized_huber(camera(a['depth_local'],R[f],t[f],s)-a['cloud'],cfg.depth_sigma_m,cfg.depth_weight/n))
                if len(a['coverage_local']):
                    groups.append(normalized_huber(project(camera(a['coverage_local'],R[f],t[f],s),K)-a['positive_uv'],cfg.silhouette_sigma_px,cfg.coverage_weight/n))
                if len(a['visible_local']):
                    uv=project(camera(a['visible_local'],R[f],t[f],s),K);h,w=a['distance'].shape
                    dist=ndimage.map_coordinates(a['distance'],[uv[:,1],uv[:,0]],order=1,mode='nearest')
                    # Outside-FOV points are not silently rewarded by clamping.
                    outside=np.maximum(0,-uv[:,0])+np.maximum(0,uv[:,0]-(w-1))+np.maximum(0,-uv[:,1])+np.maximum(0,uv[:,1]-(h-1))
                    groups.append(normalized_huber(dist+outside,cfg.silhouette_sigma_px,cfg.free_space_weight/n))
                z=camera(V,R[f],t[f],s)[:,2]
                groups.append(normalized_huber(np.maximum(0,cfg.minimum_z_m-z),.01,1/n))
            groups.extend([normalized_huber(C-C0,cfg.centroid_prior_sigma_m),
                           normalized_huber(p[1:1+3*n],cfg.rotation_prior_sigma_rad),
                           normalized_huber([p[0]],cfg.log_scale_prior_sigma)])
            if n>=3:
                dt=np.diff(times);mid=(dt[:-1]+dt[1:])/2
                # Regularize corrections rather than erase the MAIN motion.
                vel=np.diff(C-C0,axis=0)/dt[:,None]
                acc=np.diff(vel,axis=0)/mid[:,None]
                dR=R@R0.swapaxes(-1,-2)
                omega=Rotation.from_matrix(dR[1:]@dR[:-1].swapaxes(-1,-2)).as_rotvec()/dt[:,None]
                aa=np.diff(omega,axis=0)/mid[:,None]
                groups.extend([normalized_huber(acc,cfg.correction_accel_sigma_m_s2,cfg.temporal_weight),normalized_huber(aa,cfg.angular_accel_sigma_rad_s2,cfg.temporal_weight)])
            return np.concatenate(groups)
        before=float(np.dot(residual(x),residual(x)))
        # Dense finite difference is intentional for the bounded initial pilot.
        fit=least_squares(residual,x,bounds=(lo,hi),loss='linear',max_nfev=cfg.max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        after=float(np.dot(residual(fit.x),residual(fit.x)))
        accept=np.isfinite(fit.x).all() and np.isfinite(after) and after<=before+1e-10
        history.append(dict(outer=outer,cost_before=before,cost_after=after,accepted=bool(accept),nfev=fit.nfev,solver_success=bool(fit.success),
                            depth_counts=[len(a['cloud']) for a in pair],coverage_counts=[len(a['positive_uv']) for a in pair],render_counts=[len(a['visible_local']) for a in pair]))
        if accept:x=fit.x
        else:break
    R,t,s,C=unpack(x);minz=min(float(camera(V,R[f],t[f],s)[:,2].min()) for f in range(n))
    result['diagnostics'].update(history=history,min_fullmesh_Z_m=minz,
        before=[render_diagnostics(V,F,R0[f],t0[f],s0,K,effective[f]) for f in range(n)],
        after=[render_diagnostics(V,F,R[f],t[f],s,K,effective[f]) for f in range(n)])
    if not np.isfinite(t).all() or minz<cfg.minimum_z_m:
        result['fallback']='invalid_fullmesh_geometry'
    else:result.update(R=R,t=t,scale=s)
    return result


def expand_corrections(V,F,R0,t0,s0,times,sample_ids,fit):
    """Actual-time interpolate baseline-relative centroid/SO3 corrections.

    Unobserved frames are predictions, never invented measurements. Outside the
    sampled span corrections are held at endpoint; whole-case invalid mesh
    fallback is the caller's obligation. Human arrays and local mesh unchanged.
    """
    R0=proper(R0);times=np.asarray(times);ix=np.asarray(sample_ids,int)
    mu=area_centroid(np.asarray(V),np.asarray(F));C0=t0+s0*np.einsum('fij,j->fi',R0,mu)
    if fit['fallback'] is not None:return R0.copy(),t0.copy(),float(s0)
    Cs=fit['t']+fit['scale']*np.einsum('fij,j->fi',fit['R'],mu)
    dc=Cs-C0[ix];delta=np.stack([np.interp(times,times[ix],dc[:,j]) for j in range(3)],1)
    dR=fit['R']@R0[ix].swapaxes(-1,-2)
    rotations=Slerp(times[ix],Rotation.from_matrix(dR))(np.clip(times,times[ix[0]],times[ix[-1]])).as_matrix() if len(ix)>1 else np.repeat(dR,len(times),axis=0)
    R=rotations@R0;t=C0+delta-fit['scale']*np.einsum('fij,j->fi',R,mu)
    return proper(R),t,float(fit['scale'])
