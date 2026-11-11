"""Compiled arithmetic for the same point-state terminal certificate.
The NumPy implementation in backup.py remains the independent reference.
"""
import numpy as np
from numba import njit
from collections import namedtuple
from backup import ReserveFilter as ReferenceFilter, guard, certificate
import time
Parameters=namedtuple('Parameters','omega J Rs L poles flux capacitance Emin Emax current_limit heat_budget kt Rc')
def parameters(c):
 return Parameters(np.asarray(c.omega),c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance,c.Emin,c.Emax,c.current_limit,np.asarray(c.heat_budget),c.kt,c.Rc)
@njit(cache=True)
def reserves(w,i,c,wc=1000.):
 n=len(w);r2=i[:n]**2+i[n:]**2;q=i[n:];gamma=.75*c.L;rho=c.Rc
 lower=max(rho/(2*wc)-gamma,0.)*r2+c.kt/wc*np.maximum(w*q,0.)+c.kt**2*q*q/(2*c.J*wc**2)
 upper=max(gamma-rho/(2*wc),0.)*r2+c.kt/wc*np.maximum(-w*q,0.)
 wp=np.maximum(w,0)+c.kt*np.maximum(q,0)/(c.J*wc)
 wm=np.maximum(-w,0)+c.kt*np.maximum(-q,0)/(c.J*wc)
 voltage=np.zeros(n)
 for ww in [-wm,wp]:
  # Affine in decaying-current fraction z: the norm maximum is at z=0 or 1.
  for z in [0.,1.]:
   d=i[:n]*z;qq=q*z
   vd=(c.Rs-c.L*wc)*d-c.poles*ww*c.L*qq
   vq=(c.Rs-c.L*wc)*qq+c.poles*ww*(c.L*d+c.flux)
   voltage=np.maximum(voltage,np.hypot(vd,vq))
 return lower,upper,voltage


@njit(cache=True)
def speed_box(y,ref,h,c,bounds,wc=1000.):
 n=len(c.omega);w=y[:n];q=y[2*n:3*n];uq=ref[n:];qb=np.maximum(np.abs(q),np.abs(uq));wb=np.abs(w)+c.kt*qb*h/c.J;D=(c.kt*qb+bounds[1]*wb)*h/c.J;lo=w-D;hi=w+D
 ps=np.vstack((bounds[0]*lo,bounds[0]*hi,bounds[1]*lo,bounds[1]*hi));al=(c.kt*np.minimum(q,uq)-np.maximum(np.maximum(ps[0],ps[1]),np.maximum(ps[2],ps[3])))/c.J;au=(c.kt*np.maximum(q,uq)-np.minimum(np.minimum(ps[0],ps[1]),np.minimum(ps[2],ps[3])))/c.J
 return np.maximum(lo,w+np.minimum(0,al)*h),np.minimum(hi,w+np.maximum(0,au)*h)


@njit(cache=True)
def endpoint_box(y,ref,h,c,bounds,wc=1000.):
 n=len(c.omega);w=y[:n];cur=y[n:3*n];d=cur-ref;ie=ref+d*np.exp(-wc*h)
 wl,wh=speed_box(y,ref,h,c,bounds,wc)
 products=np.vstack((bounds[0]*wl,bounds[0]*wh,bounds[1]*wl,bounds[1]*wh))
 qintegral=ref[n:]*h+d[n:]*(-np.expm1(-wc*h))/wc
 wlo=w+c.kt*qintegral/c.J-np.maximum(np.maximum(products[0],products[1]),np.maximum(products[2],products[3]))*h/c.J
 whi=w+c.kt*qintegral/c.J-np.minimum(np.minimum(products[0],products[1]),np.minimum(products[2],products[3]))*h/c.J
 wlo=np.maximum(wlo,wl);whi=np.minimum(whi,wh)
 mag=.75*c.L*(np.sum(cur*cur)-np.sum(ie*ie))
 dh=c.Rc*((ref[:n]**2+ref[n:]**2)*h+2*(ref[:n]*d[:n]+ref[n:]*d[n:])*(-np.expm1(-wc*h))/wc+(d[:n]**2+d[n:]**2)*(-np.expm1(-2*wc*h))/(2*wc))
 maxend=np.maximum(np.abs(wlo),np.abs(whi));minend=np.where(wlo*whi<=0,0,np.minimum(np.abs(wlo),np.abs(whi)))
 maxhold=np.maximum(np.abs(wl),np.abs(wh));minhold=np.where(wl*wh<=0,0,np.minimum(np.abs(wl),np.abs(wh)))
 elo=y[3*n]+.5*c.J*np.sum(w*w-maxend*maxend)+mag-dh.sum()-bounds[1]*np.sum(maxhold*maxhold)*h
 ehi=y[3*n]+.5*c.J*np.sum(w*w-minend*minend)+mag-dh.sum()-bounds[0]*np.sum(minhold*minhold)*h
 return wlo,whi,float(elo),float(ehi),ie,dh


@njit(cache=True)
def terminal_certificate(y,ref,h,c,bounds,wc=1000.,brake=0.):
 n=len(c.omega);wl,wh,elo,ehi,ie,dh=endpoint_box(y,ref,h,c,bounds,wc)
 L,H,V=reserves(wl,ie,c,wc);ll,hh,vv=reserves(wh,ie,c,wc)
 elo=elo-brake*h-np.maximum(L,ll).sum();ehi=ehi-brake*h+np.maximum(H,hh).sum();v=np.maximum(V,vv)
 r2=ie[:n]**2+ie[n:]**2
 return bool(elo>=c.Emin and ehi<=c.Emax and np.max(v)<=np.sqrt(max(0,2*elo/(3*c.capacitance))) and np.max(y[3*n+1:]+dh+c.Rc*r2/(2*wc)-c.heat_budget)<=1e-10)

class CompiledReserveFilter(ReferenceFilter):
 def __call__(self,y,zeta,ref,h,c,brake=0.):
  if zeta is not None:return super().__call__(y,zeta,ref,h,c,brake)
  start=time.perf_counter();self.calls+=1;n=len(c.omega);cp=parameters(c)
  candidates=[ref]+[np.r_[ref[:n]*s,ref[n:]]for s in [.75,.5,.25,0.]]+[np.r_[ref[:n],ref[n:]*s]for s in [.75,.5,.25,0.]]+[ref*s for s in [.75,.5,.25]]
  for idx,u in enumerate(candidates):
   g=guard(y,u,brake,h,c,self.bounds);g['i2t_safe']=True
   if all(g[k]for k in ['unsaturated','energy_safe','current_safe','heat_safe']) and terminal_certificate(y,u,h,cp,self.bounds,brake=brake):
    self.changed+=int(idx>0);self.times.append(time.perf_counter()-start);g['backup']=False;g['applied_brake']=brake;return u,g,idx>0
  cert=certificate(y,None,c,self.tau,self.inom);self.times.append(time.perf_counter()-start)
  if cert['valid']:
   self.zero+=1;self.changed+=1
   return np.zeros(2*n),dict(unsaturated=True,energy_safe=True,current_safe=True,heat_safe=True,i2t_safe=True,backup=True,applied_brake=0.,**cert),True
  return None,dict(backup_certificate=cert,i2t_safe=False),True
