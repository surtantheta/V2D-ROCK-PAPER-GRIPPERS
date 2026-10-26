#!/usr/bin/env python3
"""Predicted-only clip-shared positive object scale and camera surface-centroid fit.

Inputs surface_metric/mu_metric/clouds/contacts already use the SAME metric gauge.
Never rescale human vertices or depth here. No GT/scorer imports. Fixed rotations.
Missing image/depth observations retain their frames and the full trajectory prior.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

PROTOCOL = dict(version='centroid_scale_trajectory_v2', outer_iterations=8,
    inner_max_nfev=30, line_search_steps=10, max_image_surface_samples=256,
    max_image_target_samples=128, image_sigma_px=4., image_huber_px=8.,
    depth_sigma_m=.03, depth_huber_m=.03, centroid_prior_sigma_m=.5,
    log_scale_prior_sigma=float(np.log(2.)), relative_scale_bounds=[.25, 4.],
    centroid_bound_min_m=1., centroid_bound_depth_multiple=2.,
    acceleration_sigma_m_s2=5., contact_sigma_m=.05, contact_weight=.25,
    contact_hinge_m=.03, minimum_surface_z_m=.02, camera_barrier_sigma_m=.02,
    rotation_fixed=True, body_depth_scale_fixed=True, initial_3d_admission=False,
    contact_state='caller predicted-2D association only; no initial 3D gate',
    bounds_origin='declared broad geometric policy; no native labels',
    pixel_coordinates='pixel centres x+.5/y+.5; distance-field array uses uv-.5')


def acceleration_matrix(times, frame_ids=None, frame_step=1):
    times = np.asarray(times, float); n = len(times)
    assert times.shape == (n,) and np.isfinite(times).all() and np.all(np.diff(times)>0)
    ids = np.arange(n) if frame_ids is None else np.asarray(frame_ids)
    assert ids.shape == (n,) and np.all(np.diff(ids)>0)
    step = 1 if frame_ids is None else frame_step
    rows=[]
    for i in range(1,n-1):
        if ids[i]-ids[i-1]!=step or ids[i+1]-ids[i]!=step: continue
        a=times[i]-times[i-1]; b=times[i+1]-times[i]
        row=np.zeros(n); row[i-1]=2/(a*(a+b)); row[i]=-2/(a*b); row[i+1]=2/(b*(a+b)); rows.append(row)
    return np.stack(rows) if rows else np.zeros((0,n))


def _huber(r, delta):
    r=np.asarray(r); return np.where(r<=delta,.5*r*r,delta*(r-.5*delta))


def _robust_weight(r,delta):
    return np.minimum(1.,delta/np.maximum(r,1e-12))


def _projection(points,K):
    z=np.maximum(points[:,2],.001); h=points@K.T
    uv=h[:,:2]/z[:,None]
    jac=np.zeros((len(points),2,3))
    jac[:,:,0]=K[:2,0]/z[:,None]; jac[:,:,1]=K[:2,1]/z[:,None]
    jac[:,:,2]=K[:2,2]/z[:,None]-h[:,:2]/z[:,None]**2
    # z clipping only a numerical safeguard; explicit camera barrier supplies recovery.
    clipped=points[:,2]<=.001
    jac[clipped,:,2]=K[:2,2]/z[clipped,None]
    return uv,jac


def _distance_field(field,uv):
    """Bilinear distance plus Euclidean out-of-image distance; no silent clipping."""
    uv=np.asarray(uv,float)-.5
    h,w=field.shape; x=np.clip(uv[:,0],0,w-1); y=np.clip(uv[:,1],0,h-1)
    x0=np.minimum(np.floor(x).astype(int),w-2); y0=np.minimum(np.floor(y).astype(int),h-2)
    dx=x-x0; dy=y-y0
    a=field[y0,x0]; b=field[y0,x0+1]; c=field[y0+1,x0]; d=field[y0+1,x0+1]
    val=(1-dx)*(1-dy)*a+dx*(1-dy)*b+(1-dx)*dy*c+dx*dy*d
    grad=np.c_[(1-dy)*(b-a)+dy*(d-c),(1-dx)*(c-a)+dx*(d-b)]
    inside=(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    grad[:,0]*=(uv[:,0]>=0)&(uv[:,0]<=w-1); grad[:,1]*=(uv[:,1]>=0)&(uv[:,1]<=h-1)
    outside=uv-np.c_[x,y]; dist=np.linalg.norm(outside,axis=1)
    grad+=outside/np.maximum(dist[:,None],1e-12)
    return val+dist,grad


def solve(surface_metric, rotations, baseline_translation, clouds, quality,
          depth_confidence, times, image_observations, contacts=None, *, mu_metric,
          allow_scale=True, use_depth=True, use_temporal=True, use_contact=False,
          frame_ids=None, frame_step=1):
    """Return (original-origin translation[N,3], relative_scale, diagnostics).

    Images: None, or {Kmodel, maskuv[N,2], maskdistancefield[H,W], confidence,
    enabled}. Field is distance to allowed object OR occluder region; positive
    maskuv targets remain accepted object-only pixels. Model pixel centres are
    x+.5/y+.5; distance-field array interpolation uses uv-.5.
    Contacts: {frame: [hand_tip_array[<=5,3], ...]}, pre-gated by predicted 2D
    interaction. No body/hand contact is inferred for a board or an unknown state.
    mu_metric MUST be the exact triangle-area centroid, not a sample mean.
    """
    surface=np.asarray(surface_metric,float); mu=np.asarray(mu_metric,float)
    base=np.asarray(baseline_translation,float); R=np.asarray(rotations,float); n=len(base)
    assert surface.ndim==2 and surface.shape[1]==3 and len(surface)>=4
    assert mu.shape==(3,) and base.shape==(n,3) and R.shape==(n,3,3) and n>0
    assert all(np.isfinite(a).all() for a in (surface,mu,base,R))
    assert np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-5) and np.all(np.linalg.det(R)>0)
    assert len(clouds)==len(image_observations)==n
    q=np.asarray(quality,float); conf=np.asarray(depth_confidence,float)
    assert q.shape==conf.shape==(n,) and np.isfinite(q).all() and np.isfinite(conf).all() and np.all(conf>=0)
    centered=surface-mu; tree=cKDTree(centered); C0=base+np.einsum('nij,j->ni',R,mu)
    assert np.all(C0[:,2]>.05), 'camera centroids must start in front of camera'
    D=acceleration_matrix(times,frame_ids,frame_step)
    cs=[np.asarray(p,float).reshape(-1,3) for p in clouds]
    assert all(np.isfinite(p).all() and np.all(p[:,2]>0) for p in cs)
    subset=np.linspace(0,len(surface)-1,min(len(surface),PROTOCOL['max_image_surface_samples']),dtype=int)
    rotated=np.einsum('nij,mj->nmi',R,centered)
    img=[]
    for ob in image_observations:
        if ob is None or not ob.get('enabled',True): img.append(None); continue
        K=np.asarray(ob['Kmodel'],float); uv=np.asarray(ob['maskuv'],float).reshape(-1,2)
        field=np.asarray(ob['maskdistancefield'],float); score=float(ob.get('confidence',1.))
        assert K.shape==(3,3) and np.allclose(K[2],[0,0,1]) and np.isfinite(K).all()
        assert field.ndim==2 and min(field.shape)>=2 and np.isfinite(field).all() and np.all(field>=0)
        assert np.isfinite(uv).all() and np.isfinite(score) and score>=0
        if len(uv)==0: img.append(None); continue
        idx=np.linspace(0,len(uv)-1,min(len(uv),PROTOCOL['max_image_target_samples']),dtype=int)
        img.append((K,uv[idx],field,score))
    contacts={} if contacts is None else contacts
    for i,hands in contacts.items():
        assert 0<=i<n
        for hand in hands: assert np.asarray(hand).size and np.isfinite(hand).all()
    trust=np.maximum(q,.1)/PROTOCOL['centroid_prior_sigma_m']**2
    bound=np.maximum(PROTOCOL['centroid_bound_min_m'],PROTOCOL['centroid_bound_depth_multiple']*C0[:,2])
    lower=C0-bound[:,None]; upper=C0+bound[:,None]; lower[:,2]=np.maximum(lower[:,2],.05)
    if allow_scale:
        lo=np.r_[np.log(PROTOCOL['relative_scale_bounds'][0]),lower.ravel()]
        hi=np.r_[np.log(PROTOCOL['relative_scale_bounds'][1]),upper.ravel()]
        x=np.r_[0.,C0.ravel()]; offset=1
    else: lo=lower.ravel(); hi=upper.ravel(); x=C0.ravel().copy(); offset=0
    def unpack(v): return (np.exp(v[0]),v[1:].reshape(n,3)) if allow_scale else (1.,v.reshape(n,3))
    def associations(v):
        s,C=unpack(v); out=[]
        for i in range(n):
            p=cs[i]; data={}
            if use_depth and len(p) and conf[i]>0:
                ids=tree.query(((p-C[i])@R[i])/s,workers=1)[1]
                err=np.linalg.norm(s*rotated[i,ids]+C[i]-p,axis=1)
                data['depth']=(ids,_robust_weight(err,PROTOCOL['depth_huber_m']))
            if img[i] is not None:
                K,uv,field,score=img[i]; proj,_=_projection(s*rotated[i,subset]+C[i],K)
                dist,ids=cKDTree(proj).query(uv,workers=1)
                val,_=_distance_field(field,proj)
                data['image']=(subset[ids],_robust_weight(dist,PROTOCOL['image_huber_px']),_robust_weight(val,PROTOCOL['image_huber_px']))
            hc=[]
            for hand in contacts.get(i,[]) if use_contact else []:
                hand=np.asarray(hand,float).reshape(-1,3)
                ds,ids=tree.query(((hand-C[i])@R[i])/s,workers=1); j=int(np.argmin(ds))
                hc.append((hand[j],int(ids[j])))
            data['hands']=hc; out.append(data)
        return out
    def residual(v,assoc,want_jac=False):
        s,C=unpack(v); blocks=[]; rr=[]; cc=[]; vv=[]; row=0
        def add(val,frame=None,dg=None,dc=None):
            nonlocal row
            val=np.asarray(val,float).ravel(); m=len(val); blocks.append(val)
            if want_jac:
                if allow_scale and dg is not None:
                    rr.extend(range(row,row+m)); cc.extend([0]*m); vv.extend(np.asarray(dg).ravel())
                if frame is not None and dc is not None:
                    dc=np.asarray(dc).reshape(m,3)
                    rr.extend(np.repeat(np.arange(row,row+m),3)); cc.extend(np.tile(np.arange(offset+3*frame,offset+3*frame+3),m)); vv.extend(dc.ravel())
            row+=m
        if allow_scale: add([np.sqrt(n)*np.log(s)/PROTOCOL['log_scale_prior_sigma']],dg=[np.sqrt(n)/PROTOCOL['log_scale_prior_sigma']])
        for i in range(n):
            weight=np.sqrt(trust[i]); add(weight*(C[i]-C0[i]),i,dc=weight*np.eye(3))
            data=assoc[i]
            if 'depth' in data:
                ids,rw=data['depth']; qs=s*rotated[i,ids]
                w=np.sqrt(conf[i]*rw/len(ids))/PROTOCOL['depth_sigma_m']
                add((qs+C[i]-cs[i])*w[:,None],i,dg=(qs*w[:,None]).ravel(),dc=np.tile(np.eye(3),(len(ids),1))*np.repeat(w,3)[:,None])
            if 'image' in data:
                K,uv,field,score=img[i]; ids,rw,reverse_rw=data['image']; qs=s*rotated[i,ids]
                proj,jp=_projection(qs+C[i],K); w=np.sqrt(score*rw/len(ids))/PROTOCOL['image_sigma_px']
                add((proj-uv)*w[:,None],i,dg=np.einsum('mjk,mk->mj',jp,qs)*w[:,None],dc=jp.reshape(-1,3)*np.repeat(w,2)[:,None])
                qs=s*rotated[i,subset]; proj,jp=_projection(qs+C[i],K); val,grad=_distance_field(field,proj)
                w=np.sqrt(score*reverse_rw/len(subset))/PROTOCOL['image_sigma_px']; jxyz=np.einsum('mj,mjk->mk',grad,jp)
                add(val*w,i,dg=np.einsum('mk,mk->m',jxyz,qs)*w,dc=jxyz*w[:,None])
            hands=data['hands']
            for hand,j in hands:
                qs=s*rotated[i,j]; delta=qs+C[i]-hand; dist=np.linalg.norm(delta)
                w=np.sqrt(PROTOCOL['contact_weight']/len(hands))/PROTOCOL['contact_sigma_m']
                active=dist>PROTOCOL['contact_hinge_m']; deriv=delta/max(dist,1e-12) if active else np.zeros(3)
                add([w*max(dist-PROTOCOL['contact_hinge_m'],0)],i,dg=[w*deriv@qs],dc=[w*deriv])
            # All fixed model sample points remain present even outside image/behind camera.
            qs=s*rotated[i,subset]; z=qs[:,2]+C[i,2]; active=z<PROTOCOL['minimum_surface_z_m']
            w=1/(np.sqrt(len(subset))*PROTOCOL['camera_barrier_sigma_m']); dc=np.zeros((len(subset),3)); dc[:,2]=-active.astype(float)*w
            add(np.maximum(PROTOCOL['minimum_surface_z_m']-z,0)*w,i,dg=-active.astype(float)*qs[:,2]*w,dc=dc)
        if use_temporal and len(D):
            w=np.sqrt(n/len(D))/PROTOCOL['acceleration_sigma_m_s2']; val=(D@C)*w
            blocks.append(val.ravel()); m=val.size
            if want_jac:
                for k,j in zip(*np.nonzero(D)):
                    for axis in range(3): rr.append(row+3*k+axis); cc.append(offset+3*j+axis); vv.append(w*D[k,j])
            row+=m
        r=np.concatenate(blocks)
        return coo_matrix((vv,(rr,cc)),shape=(len(r),len(v))).tocsr() if want_jac else r
    def energy(v):
        s,C=unpack(v); terms=dict(prior=float(.5*np.sum(trust[:,None]*(C-C0)**2)/n),scale=float(.5*(np.log(s)/PROTOCOL['log_scale_prior_sigma'])**2),depth=0.,image=0.,temporal=0.,contact=0.,camera=0.)
        if use_temporal and len(D): terms['temporal']=float(.5*np.mean(np.sum((D@C)**2,axis=1))/PROTOCOL['acceleration_sigma_m_s2']**2)
        for i in range(n):
            if use_depth and len(cs[i]) and conf[i]>0:
                dist=tree.query(((cs[i]-C[i])@R[i])/s,workers=1)[0]*s
                terms['depth']+=float(conf[i]*np.mean(_huber(dist,PROTOCOL['depth_huber_m']))/PROTOCOL['depth_sigma_m']**2)/n
            if img[i] is not None:
                K,uv,field,score=img[i]; proj,_=_projection(s*rotated[i,subset]+C[i],K)
                ds=cKDTree(proj).query(uv,workers=1)[0]; val,_=_distance_field(field,proj)
                terms['image']+=float(score*(np.mean(_huber(ds,PROTOCOL['image_huber_px']))+np.mean(_huber(val,PROTOCOL['image_huber_px'])))/PROTOCOL['image_sigma_px']**2)/n
            hands=contacts.get(i,[]) if use_contact else []
            for hand in hands:
                ds=tree.query(((np.asarray(hand).reshape(-1,3)-C[i])@R[i])/s,workers=1)[0]*s
                terms['contact']+=float(.5*PROTOCOL['contact_weight']*max(ds.min()-PROTOCOL['contact_hinge_m'],0)**2/(len(hands)*PROTOCOL['contact_sigma_m']**2))/n
            z=(s*rotated[i,subset]+C[i])[:,2]
            terms['camera']+=float(.5*np.mean(np.maximum(PROTOCOL['minimum_surface_z_m']-z,0)**2)/PROTOCOL['camera_barrier_sigma_m']**2)/n
        terms['total']=sum(terms.values()); return terms
    initial=energy(x); history=[]
    # Pure-prior/no-observation identity is exact, including nonzero mesh origins.
    has_data=any(use_depth and len(p) and conf[i]>0 for i,p in enumerate(cs)) or any(z is not None for z in img) or (use_contact and any(contacts.values()))
    if has_data or use_temporal:
        for outer in range(PROTOCOL['outer_iterations']):
            assoc=associations(x); before=energy(x)
            fit=least_squares(lambda v:residual(v,assoc),x,jac=lambda v:residual(v,assoc,True),bounds=(lo,hi),method='trf',tr_solver='lsmr',x_scale='jac',max_nfev=PROTOCOL['inner_max_nfev'],ftol=1e-7,xtol=1e-7,gtol=1e-7)
            assert np.isfinite(fit.x).all()
            accepted=0.; candidate=x
            for step in range(PROTOCOL['line_search_steps']):
                fraction=.5**step; trial=x+fraction*(fit.x-x)
                if energy(trial)['total']<=before['total']+1e-10: candidate=trial; accepted=fraction; break
            history.append(dict(iteration=outer,accepted_fraction=accepted,nfev=int(fit.nfev),solver_status=int(fit.status),objective_after=energy(candidate)))
            shift=float(np.linalg.norm(candidate-x)); x=candidate
            if accepted==0 or shift<1e-7: break
    ratio,C=unpack(x); t=C-ratio*np.einsum('nij,j->ni',R,mu)
    if not has_data and not use_temporal: t=base.copy()
    assert np.isfinite(t).all() and np.isfinite(ratio) and ratio>0
    final_energy=energy(x)
    min_z=float(np.min(ratio*rotated[:,:,2]+C[:,None,2]))
    finite_objective=bool(all(np.isfinite(v) for v in final_energy.values()))
    return t,float(ratio),dict(protocol=PROTOCOL,centroids_metric=C.tolist(),mu_metric=mu.tolist(),history=history,initial_objective=initial,final_objective=final_energy,min_camera_sample_z_m=min_z,finite_final_objective=finite_objective,valid_fit_numerics=bool(finite_objective and min_z>0),relative_scale=float(ratio),depth_points_per_frame=[len(p) if use_depth else 0 for p in cs],image_frames=[i for i,z in enumerate(img) if z is not None],contact_frames=sorted(contacts) if use_contact else [],valid_temporal_triples=len(D),all_frames_retained=n,scale_at_bound=bool(allow_scale and min(abs(x[0]-lo[0]),abs(x[0]-hi[0]))<1e-5),max_centroid_shift_m=float(np.linalg.norm(C-C0,axis=1).max()))
