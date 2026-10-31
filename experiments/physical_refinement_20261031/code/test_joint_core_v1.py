"""Physics-selected normalization, genuine joint Jacobian and frame contracts."""
from pathlib import Path
import json,os,sys
import numpy as np
sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
sys.path.insert(0,str(Path(__file__).parent))
import temporal_v2 as util
from joint_core_v1 import Config,SignedMesh,solve,second_difference,frozen_normalizers,inner_functions

if __name__=='__main__':
    assert os.environ.get('PBS_JOBID')
    import numba,trimesh
    from scipy.spatial.transform import Rotation
    numba.set_num_threads(2)
    pe=util.module(util.PRIOR/'code/evaluate_penetration.py','joint_test_pe')
    pen=util.module(pe.PEN,'joint_test_official_pen')
    box=trimesh.creation.box(extents=[2.,2.,2.]);v=np.r_[np.asarray(box.vertices),np.zeros((1,3))];f=np.r_[np.asarray(box.faces),np.tile([8,8,8],(64,1))]
    cfg=Config();n=5;ids=np.arange(n);t0=np.tile([0.,0.,3.],(n,1));rot=np.tile(np.eye(3),(n,1,1))
    q=np.tile([3.,3.,3.],(5840,1));q[0]=[.988,.1,.2]
    hands=np.tile(q,(n,1,1))+t0[:,None]
    updated,diag,initial,final=solve(v,f,rot,1.,hands,t0,ids,pen,cfg)
    expected=-10/11*.014
    assert np.allclose(updated[:,0],expected,atol=1e-8,rtol=0)
    assert diag['final_penetration']['maximum_m']<1e-8
    assert diag['maximum_correction_m']>.01
    # Full-point duplication cannot alter the frozen-normalizer objective.
    double=np.repeat(hands,2,axis=1)
    again,doubled,_,_=solve(v,f,rot,1.,double,t0,ids,pen,cfg)
    assert np.allclose(again,updated,atol=1e-9,rtol=0)
    assert abs(doubled['final_merit']-diag['final_merit'])<1e-12
    # A deeper intersection must improve substantially while respecting20mm.
    deep=hands.copy();deep[:,0,0]=.97
    bounded,bdiag,_,_=solve(v,f,rot,1.,deep,t0,ids,pen,cfg)
    assert abs(bdiag['maximum_correction_m']-.02)<1e-9
    assert bdiag['final_penetration']['maximum_m']<.010001
    # Two opposing contacts are incompatible with translation-only separation.
    opposing=np.tile([3.,3.,3.],(n,5840,1))+t0[:,None]
    opposing[:,0]=np.array([.995,0.,0.])+t0;opposing[:,1]=np.array([-.995,0.,0.])+t0
    unchanged,odiag,_,_=solve(v,f,rot,1.,opposing,t0,ids,pen,cfg)
    assert np.allclose(unchanged,t0,atol=1e-12,rtol=0) and odiag['final_penetration']['maximum_m']>0
    # Sparse analytic Jacobian agrees with central finite differences.
    mesh=SignedMesh(v,f,rot,1.,hands,pen);normalizers,_=frozen_normalizers(initial,cfg)
    d,_=second_difference(ids);a,b,w,_=mesh.linearization(t0,np.zeros_like(t0),initial,normalizers,cfg)
    fun,jac=inner_functions(t0,d,a,b,w,cfg)
    x=np.linspace(-1e-5,1e-5,t0.size);eps=1e-7
    numeric=np.column_stack([(fun(x+eps*np.eye(t0.size)[k])-fun(x-eps*np.eye(t0.size)[k]))/(2*eps) for k in range(t0.size)])
    err=float(np.abs(numeric-jac(x).toarray()).max());assert err<1e-6,err
    # Correct real scene derivative under nonidentity R and nonunit scale.
    r=Rotation.from_rotvec([.3,-.4,.2]).as_matrix();rs=np.tile(r,(n,1,1));scale=2.3
    wh=scale*np.einsum('tpi,ji->tpj',np.tile(q,(n,1,1)),r)+t0[:,None]
    m=SignedMesh(v,f,rs,scale,wh,pen);gradient=[]
    for k in range(3):
        shift=np.zeros_like(t0);shift[0,k]=1e-6
        gradient.append((m.signed(t0+shift,.022)[0,0]-m.signed(t0-shift,.022)[0,0])/2e-6)
    assert np.allclose(gradient,-r[:,0],atol=1e-7,rtol=0),(gradient,-r[:,0])
    # No temporal triple spans a missing source interval.
    _,gap=second_difference(np.array([0,1,2,10,11,12]));assert gap['eligible_triples']==2
    d5,_=second_difference(ids*5);d1,_=second_difference(ids)
    assert np.allclose(d5.toarray(),d1.toarray()/25,atol=0,rtol=0)
    util.dump(util.ROOT/'JOINT_CORE_TEST_V1.json',dict(status='PASS',rare_single_contact_12mm_cleared=True,rare_contact_displacement_m=float(diag['maximum_correction_m']),active_normalizer_not_diluted_by_5840points=True,uniform_point_duplication_objective_invariant=True,deep_contact_20mm_bound=True,opposing_contacts_explicitly_unresolved=True,analytic_jacobian_max_error=err,rotated_scaled_object_sdf_derivative_correct=True,padded_unused_vertex_safe=True,gap_triples_not_bridged=True,native_stride_penalty_scaling_correct=True,source_sha256={str(Path(__file__)):util.sha(__file__),str(Path(__file__).with_name('joint_core_v1.py')):util.sha(Path(__file__).with_name('joint_core_v1.py'))},CPU_job=os.environ['PBS_JOBID']))
    print('JOINT_CORE_TEST_PASS',err,flush=True)
