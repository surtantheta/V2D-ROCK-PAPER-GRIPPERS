"""Fixed-radius perspective circle pose from an image ellipse, retaining both starts."""
import math
import cv2
import numpy as np
from scipy.optimize import least_squares


def rotation(tilt, phi):
    c,s=math.cos(tilt),math.sin(tilt)
    cp,sp=math.cos(phi),math.sin(phi)
    return np.array([[cp,-sp,0],[sp,cp,0],[0,0,1]]) @ np.array([[1,0,0],[0,c,-s],[0,s,c]])


def fit_circle(mask, focal, radius, sign):
    ys,xs=np.nonzero(mask)
    points=np.stack([xs+.5,ys+.5],-1).astype(np.float32)
    (cx,cy),(width,height),angle=cv2.fitEllipse(points[:,None])
    if width>=height:
        a,b=width/2,height/2;phi=math.radians(angle)
    else:
        a,b=height/2,width/2;phi=math.radians(angle+90)
    phi=(phi+math.pi)%(2*math.pi)-math.pi
    tilt=sign*math.acos(np.clip(b/a,.01,1))
    depth=focal*radius/a
    h,w=mask.shape
    initial=np.array([tilt,phi,(cx-w/2)*depth/focal,(cy-h/2)*depth/focal,math.log(depth)])
    theta=np.arange(96)*2*math.pi/96
    circle=np.stack([radius*np.cos(theta),radius*np.sin(theta),np.zeros_like(theta)],-1)
    cp,sp=math.cos(phi),math.sin(phi)
    def residual(parameters):
        rr=rotation(parameters[0],parameters[1])
        tt=np.array([parameters[2],parameters[3],math.exp(parameters[4])])
        camera=circle@rr.T+tt
        if camera[:,2].min()<=.05:return np.full(96,100.)
        uv=camera[:,:2]/camera[:,2:]*focal+np.array([w/2,h/2])
        centered=uv-np.array([cx,cy])
        x=cp*centered[:,0]+sp*centered[:,1]
        y=-sp*centered[:,0]+cp*centered[:,1]
        return x*x/(a*a)+y*y/(b*b)-1
    bounds=([-1.56,-2*math.pi,initial[2]-1,initial[3]-1,math.log(.6)],
            [1.56,2*math.pi,initial[2]+1,initial[3]+1,math.log(20)])
    result=least_squares(residual,initial,bounds=bounds,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=200)
    p=result.x
    return {'angles_xyz':[float(p[0]),0.,float(p[1])],
            'translation_logz':[float(p[2]),float(p[3]),float(p[4])],
            'ellipse_center':[cx,cy],'ellipse_semiaxes':[a,b],
            'normalized_conic_residual_rms':float(np.sqrt(np.mean(residual(p)**2))),
            'converged':bool(result.success),'evaluations':result.nfev,
            'metric_radius_is_fixed_prior':radius}
