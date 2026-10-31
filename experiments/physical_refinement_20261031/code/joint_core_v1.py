"""Joint bounded trajectory optimization with active signed-distance factors.

All dimensions are metres in one fixed predicted camera gauge. No files or GT
are accessed by the optimizer. The contact-pool normalization is frozen from the
baseline; adding active points cannot lower the cost denominator.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class Config:
    temporal_native_strength: float=64.
    collision_strength: float=10.
    clearance_margin_m: float=.002
    maximum_correction_m: float=.02
    outer_iterations: int=4
    inner_max_nfev: int=80
    maximum_point_depth_increase_m: float=.002
    mean_penetration_squared_tolerance_m2: float=1e-12
    reference_hand_sampling_density: int=5840

def second_difference(native_ids):
    ids=np.asarray(native_ids,np.int64)
    assert len(ids)>=3 and (np.diff(ids)>0).all()
    steps,counts=np.unique(np.diff(ids),return_counts=True)
    stride=int(steps[np.argmax(counts)])
    good=np.flatnonzero((np.diff(ids)[:-1]==stride)&(np.diff(ids)[1:]==stride))+1
    rr=[];cc=[];vv=[]
    for row,i in enumerate(good):
        rr.extend([row]*3);cc.extend([i-1,i,i+1]);vv.extend([1/stride**2,-2/stride**2,1/stride**2])
    d=sparse.coo_matrix((vv,(rr,cc)),shape=(len(good),len(ids))).tocsr()
    return sparse.kron(d,sparse.eye(3),format='csr'),dict(native_stride=stride,eligible_triples=len(good),gap_triples_not_bridged=len(ids)-2-len(good))

class SignedMesh:
    """Exact official winding/distance kernels and bounded nearest-point search."""
    def __init__(self,vertices,faces,rotations,scale,hands,pen):
        self.v=np.asarray(vertices,float);self.rot=np.asarray(rotations,float)
        self.scale=float(scale);self.hands=np.asarray(hands,float)
        assert self.scale>0 and self.hands.ndim==3 and self.hands.shape[-1]==3
        tri=self.v[np.asarray(faces,np.int64)]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        self.tri=np.ascontiguousarray(tri[area>0]);assert len(self.tri)>0
        self.v0,self.v1,self.v2=(np.ascontiguousarray(self.tri[:,k]) for k in range(3))
        self.lo=self.tri.min(axis=(0,1));self.hi=self.tri.max(axis=(0,1))
        self.tri_lo=self.tri.min(axis=1);self.tri_hi=self.tri.max(axis=1)
        # Every tree point belongs to a nondegenerate surface triangle. Padded
        # unused vertices cannot create an invalid nearest-surface upper bound.
        self.tree=cKDTree(self.tri.reshape(-1,3))
        assert pen._PENETRATION_KERNELS is not None
        self.winding,self.distance=pen._PENETRATION_KERNELS
        assert np.isfinite(self.tri).all() and np.isfinite(self.hands).all()

    def signed(self,translations,query_radius_m):
        t=np.asarray(translations,float);assert t.shape==(len(self.hands),3) and np.isfinite(t).all()
        result=np.ones(self.hands.shape[:2],np.float64)
        padding=float(query_radius_m)/self.scale
        for i,(h,r,tr) in enumerate(zip(self.hands,self.rot,t)):
            local=(h-tr)@r/self.scale
            eligible=((local>=self.lo-padding)&(local<=self.hi+padding)).all(axis=1)
            q=np.ascontiguousarray(local[eligible])
            if not len(q):continue
            inside=np.abs(self.winding(q,self.v0,self.v1,self.v2))>.5
            distance=self.distance(q,self.v0,self.v1,self.v2)*self.scale
            assert np.isfinite(distance).all() and (distance>=0).all()
            result[i,eligible]=np.where(inside,-distance,distance)
        assert np.isfinite(result).all()
        return result

    def closest(self,q):
        import trimesh
        upper=float(self.tree.query(q)[0])
        lower=np.maximum(np.maximum(self.tri_lo-q,q-self.tri_hi),0)
        possible=(lower*lower).sum(axis=1)<=upper*upper+1e-16
        assert possible.any()
        tri=self.tri[possible]
        nearest=trimesh.triangles.closest_point(tri,np.repeat(q[None],len(tri),axis=0))
        distances=np.linalg.norm(nearest-q,axis=1)
        assert np.isfinite(nearest).all() and np.isfinite(distances).all()
        return nearest[int(np.argmin(distances))]

    def linearization(self,t0,delta,signed,normalizers,cfg):
        frames,points=np.nonzero(signed<cfg.clearance_margin_m)
        rr=[];cc=[];vv=[];base=[];weights=[];joined=[]
        for i,k in zip(frames,points):
            q=(self.hands[i,k]-t0[i]-delta[i])@self.rot[i]/self.scale
            closest=self.closest(q);vec=q-closest;norm=float(np.linalg.norm(vec))
            if norm<1e-10:continue
            assert abs(norm*self.scale-abs(signed[i,k]))<1e-7,('distance-gradient mismatch',i,k)
            point_normal=(vec/norm)*(1 if signed[i,k]>=0 else -1)
            world_normal=point_normal@self.rot[i].T
            row=len(base);rr.extend([row]*3);cc.extend([3*i,3*i+1,3*i+2]);vv.extend(world_normal)
            # dphi/dOBJECTtranslation = -normal; d(margin-phi)/dt = +normal.
            base.append(cfg.clearance_margin_m-signed[i,k]-float(world_normal@delta[i]))
            weights.append(np.sqrt(cfg.collision_strength/normalizers[i]));joined.append((int(i),int(k)))
        a=sparse.coo_matrix((vv,(rr,cc)),shape=(len(base),t0.size)).tocsr()
        return a,np.asarray(base),np.asarray(weights),joined

def frozen_normalizers(initial_signed,cfg):
    n=initial_signed.shape[1]
    # Under uniform duplication of all hand points, both active count and the
    # density floor scale identically. The resulting objective is unchanged.
    floor=n/float(cfg.reference_hand_sampling_density)
    counts=(initial_signed<cfg.clearance_margin_m).sum(axis=1)
    return np.maximum(counts.astype(float),floor),counts

def penetration_stats(signed):
    depth=np.maximum(-signed,0)
    return dict(mean_m=float(depth.mean()),mean_squared_m2=float((depth*depth).mean()),maximum_m=float(depth.max(initial=0)),affected_frames=int((depth>1e-6).any(axis=1).sum()),inside_points=int((depth>0).sum()))

def merit(delta,t0,d,signed,normalizers,cfg):
    collision=np.maximum(cfg.clearance_margin_m-signed,0)
    fidelity=float((delta*delta).sum())
    acceleration=float(cfg.temporal_native_strength*np.square(d@(t0+delta).reshape(-1)).sum())
    contact=float(cfg.collision_strength*np.sum(np.square(collision).sum(axis=1)/normalizers))
    return fidelity+acceleration+contact,dict(fidelity=fidelity,trajectory_acceleration=acceleration,soft_clearance_collision=contact)

def inner_functions(t0,d,a,b,w,cfg):
    identity=sparse.eye(t0.size,format='csr');weighted_d=np.sqrt(cfg.temporal_native_strength)*d
    def fun(x):
        h=a@x+b
        return np.r_[x,weighted_d@(t0.reshape(-1)+x),w*np.maximum(h,0)]
    def jac(x):
        h=a@x+b
        return sparse.vstack([identity,weighted_d,sparse.diags(w*(h>0))@a],format='csr')
    return fun,jac

def solve(vertices,faces,rotations,scale,hands,t0,native_ids,pen,cfg=Config(),image_guard=None):
    t0=np.asarray(t0,float);assert t0.shape==(len(hands),3) and np.isfinite(t0).all()
    mesh=SignedMesh(vertices,faces,rotations,scale,hands,pen)
    d,time_info=second_difference(native_ids);delta=np.zeros_like(t0)
    radius=cfg.clearance_margin_m+cfg.maximum_correction_m
    initial=mesh.signed(t0,radius);signed=initial.copy()
    normalizers,active_counts=frozen_normalizers(initial,cfg)
    baseline_pen=penetration_stats(initial)
    initial_merit,initial_terms=merit(delta,t0,d,initial,normalizers,cfg)
    current_merit=initial_merit;history=[];rejections=[]
    for outer in range(cfg.outer_iterations):
        a,b,w,joined=mesh.linearization(t0,delta,signed,normalizers,cfg)
        fun,jac=inner_functions(t0,d,a,b,w,cfg)
        fit=least_squares(fun,delta.reshape(-1),jac=jac,method='trf',tr_solver='lsmr',bounds=(-cfg.maximum_correction_m,cfg.maximum_correction_m),max_nfev=cfg.inner_max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-10,tr_options={'atol':1e-12,'btol':1e-12})
        proposed=fit.x.reshape(t0.shape)
        maximum=float(np.linalg.norm(proposed,axis=1).max(initial=0))
        if maximum>cfg.maximum_correction_m:proposed*=cfg.maximum_correction_m/maximum
        accepted=None
        for alpha in [1.,.5,.25,.125]:
            trial=delta+alpha*(proposed-delta)
            sdf=mesh.signed(t0+trial,radius)
            value,terms=merit(trial,t0,d,sdf,normalizers,cfg);ps=penetration_stats(sdf)
            reasons=[]
            if not value<current_merit-1e-12:reasons.append('actual_joint_merit_not_lower')
            if ps['mean_squared_m2']>baseline_pen['mean_squared_m2']+cfg.mean_penetration_squared_tolerance_m2:reasons.append('fullpoint_penetration_squared_guard')
            if ps['maximum_m']>baseline_pen['maximum_m']+cfg.maximum_point_depth_increase_m:reasons.append('maximum_depth_guard')
            if not reasons and image_guard is not None:
                ok,visibility=image_guard(trial)
                if not ok:reasons.append('image_depth_fidelity_guard')
            else:visibility=None
            if reasons:
                rejections.append(dict(outer=outer,alpha=alpha,reasons=reasons,merit=value,penetration=ps,visibility=visibility));continue
            accepted=(trial,sdf,value,terms,ps,alpha,visibility);break
        if accepted is None:break
        trial,sdf,value,terms,ps,alpha,visibility=accepted
        movement=float(np.linalg.norm(trial-delta,axis=1).max(initial=0))
        history.append(dict(outer=outer,alpha=alpha,actual_merit=value,terms=terms,penetration=ps,maximum_step_m=movement,active_linearized_points=len(joined),inner_nfev=fit.nfev,inner_status=int(fit.status),visibility=visibility))
        delta,signed,current_merit=trial,sdf,value
        if movement<5e-5:break
    maximum=float(np.linalg.norm(delta,axis=1).max(initial=0))
    assert maximum<=cfg.maximum_correction_m+1e-12 and current_merit<=initial_merit+1e-12
    final_pen=penetration_stats(signed)
    assert final_pen['mean_squared_m2']<=baseline_pen['mean_squared_m2']+cfg.mean_penetration_squared_tolerance_m2
    assert final_pen['maximum_m']<=baseline_pen['maximum_m']+cfg.maximum_point_depth_increase_m
    return t0+delta,dict(config=asdict(cfg),time=time_info,initial_merit=initial_merit,initial_terms=initial_terms,final_merit=current_merit,initial_penetration=baseline_pen,final_penetration=final_pen,baseline_active_counts=active_counts.tolist(),frozen_perframe_normalizers=normalizers.tolist(),maximum_correction_m=maximum,history=history,rejections=rejections,objective='joint_all_translation_fidelity_native_time_D2_and_soft_signed_distance',posthoc_smoothing=False),initial,signed
