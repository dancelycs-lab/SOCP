"""Compiled arithmetic for the unchanged held-input map and interval certificate.
Full-dq integration is used independently in the boundary checks.
"""
import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss
nodes,weights=leggauss(12)
@njit(cache=True)
def _states_at(w,i,ref,t,J,R,L,p,flux,C,Emin,Emax,Imax,Hlim,kt,rho,f,wc=1000.):
 n=len(w);d=i[:n];q=i[n:];ud=ref[:n];uq=ref[n:];a=f/J
 z=np.exp(-wc*t);za=np.exp(-a*t)
 it=ref+(i-ref)*z
 wt=za*w+kt/J*(uq*(-np.expm1(-a*t))/a+(q-uq)*(za-z)/(wc-a))
 return wt,it

@njit(cache=True)
def _advance(y,ref,brake,dt,J,R,L,p,flux,C,Emin,Emax,Imax,Hlim,kt,rho,f,wc=1000.):
 n=(len(y)-1)//4;w=y[:n];i=y[n:3*n];d=i-ref;z=np.exp(-wc*dt)
 wt,it=_states_at(w,i,ref,dt,J,R,L,p,flux,C,Emin,Emax,Imax,Hlim,kt,rho,f,wc)
 H=rho*((ref[:n]**2+ref[n:]**2)*dt+2*(ref[:n]*d[:n]+ref[n:]*d[n:])*(-np.expm1(-wc*dt))/wc+(d[:n]**2+d[n:]**2)*(-np.expm1(-2*wc*dt))/(2*wc))
 # Gauss integration of a known sum of exponentials; 12 nodes at wc*dt<=1.
 integral=np.zeros(n)
 for x,v in zip(nodes,weights):
  ww,_=_states_at(w,i,ref,.5*dt*(x+1),J,R,L,p,flux,C,Emin,Emax,Imax,Hlim,kt,rho,f,wc);integral+=v*ww*ww*.5*dt
 energy=y[3*n]+.5*J*(np.sum(w*w)-np.sum(wt*wt))+.75*L*(np.sum(i*i)-np.sum(it*it))-np.sum(H)-f*np.sum(integral)-brake*dt
 out=y.copy();out[:n]=wt;out[n:3*n]=it;out[3*n]=energy;out[3*n+1:]+=H
 return out

@njit(cache=True)
def _guard(y,ref,brake,dt,J,R,L,p,flux,C,Emin,Emax,Imax,Hlim,kt,rho,friction_bounds,wc=1000.):
 n=(len(y)-1)//4;w=y[:n];i=y[n:3*n];z=np.exp(-wc*dt);ie=ref+(i-ref)*z
 norm0=i[:n]**2+i[n:]**2;norm1=ie[:n]**2+ie[n:]**2
 imax2=np.maximum(norm0,norm1)
 # Exact minimum of ||ref+(i-ref)*z||^2 on z in [exp(-wc dt),1].
 err=i-ref;dot=ref[:n]*err[:n]+ref[n:]*err[n:];den=err[:n]**2+err[n:]**2
 zstar=np.clip(-dot/np.maximum(den,1e-30),z,1.)
 imin2=(ref[:n]+err[:n]*zstar)**2+(ref[n:]+err[n:]*zstar)**2
 qbar=np.maximum(np.abs(i[n:]),np.abs(ref[n:]));wabs=np.abs(w)+kt*qbar*dt/J
 dw=(kt*qbar+friction_bounds[1]*wabs)*dt/J;wl=w-dw;wh=w+dw
 # Refine the coarse tube using signed torque and the friction-product range.
 fw=np.vstack((friction_bounds[0]*wl,friction_bounds[0]*wh,friction_bounds[1]*wl,friction_bounds[1]*wh))
 fwlo=np.minimum(np.minimum(fw[0],fw[1]),np.minimum(fw[2],fw[3]))
 fwhi=np.maximum(np.maximum(fw[0],fw[1]),np.maximum(fw[2],fw[3]))
 alo=(kt*np.minimum(i[n:],ref[n:])-fwhi)/J
 ahi=(kt*np.maximum(i[n:],ref[n:])-fwlo)/J
 wl=np.maximum(wl,w+np.minimum(0.,alo)*dt);wh=np.minimum(wh,w+np.maximum(0.,ahi)*dt)

 wmax=np.maximum(np.abs(wl),np.abs(wh));wmin=np.where(wl*wh<=0,0,np.minimum(np.abs(wl),np.abs(wh)))
 H=rho*((ref[:n]**2+ref[n:]**2)*dt+2*dot*(-np.expm1(-wc*dt))/wc+den*(-np.expm1(-2*wc*dt))/(2*wc))
 Elo=y[3*n]+.5*J*np.sum(w*w-wmax*wmax)+.75*L*np.sum(norm0-imax2)-sum(H)-brake*dt-friction_bounds[1]*sum(wmax*wmax)*dt
 Ehi=y[3*n]+.5*J*np.sum(w*w-wmin*wmin)+.75*L*np.sum(norm0-imin2)
 vmax=np.zeros(n)
 for zz in [z,1.]:
  ii=ref+err*zz
  for ww in [wl,wh]:
   vd=R*ii[:n]-p*ww*L*ii[n:]+L*wc*(ref[:n]-ii[:n])
   vq=R*ii[n:]+p*ww*(L*ii[:n]+flux)+L*wc*(ref[n:]-ii[n:])
   vmax=np.maximum(vmax,np.hypot(vd,vq))
 limit=np.sqrt(max(0,2*Elo/C))/np.sqrt(3.)
 thermal=y[3*n+1:]+H-Hlim
 return (Elo>0 and np.max(vmax)<=limit,Elo>=Emin and Ehi<=Emax,imax2.max()<=Imax**2+1e-8,thermal.max()<=1e-8,Elo,Ehi,limit-float(vmax.max()),float(thermal.max()))

def parameters(c):
 return (c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance,c.Emin,c.Emax,c.current_limit,np.array(c.heat_budget),c.kt,c.Rc)

def advance(y,ref,brake,dt,c,f,wc=1000.):
 return _advance(y,ref,brake,dt,*parameters(c),f,wc)

def guard(y,ref,brake,dt,c,friction_bounds,wc=1000.):
 a=_guard(y,ref,brake,dt,*parameters(c),friction_bounds,wc)
 keys=['unsaturated','energy_safe','current_safe','heat_safe','Elo','Ehi','voltage_demand_margin','heat_excess']
 return {k:bool(v) if j<4 else float(v) for j,(k,v) in enumerate(zip(keys,a))}
