"""Invariant backup set and reference execution filter."""
import numpy as np
import time
from fast_hold import guard,advance
from thermal import thermal_step,TAU,INOM,F_EQ

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

def certificate(y,zeta,c,tau=TAU,inom=INOM,wc=1000.):
 n=len(c.omega);lo,hi,v=reserves(y[:n],y[n:3*n],c,wc);r2=y[n:2*n]**2+y[2*n:3*n]**2
 elo=float(y[3*n]-lo.sum());ehi=float(y[3*n]+hi.sum());thermal=np.zeros(n) if zeta is None else zeta+r2/(2*wc*tau*inom**2)
 m=dict(lower_margin=elo-c.Emin,upper_margin=c.Emax-ehi,voltage_margin=np.sqrt(max(0,2*elo/(3*c.capacitance)))-float(v.max()),current_margin=c.current_limit**2-float(r2.max()),i2t_margin=1-float(thermal.max()),heat_margin=float(np.min(np.array(c.heat_budget)-y[3*n+1:]-c.Rc*r2/(2*wc))),Elo=elo,Ehi=ehi)
 m['valid']=bool(all(m[k]>=-1e-12 for k in ['lower_margin','upper_margin','voltage_margin','current_margin','i2t_margin','heat_margin']))
 return m

def speed_box(y,ref,h,c,bounds,wc=1000.):
 n=len(c.omega);w=y[:n];q=y[2*n:3*n];uq=ref[n:];qb=np.maximum(abs(q),abs(uq));wb=abs(w)+c.kt*qb*h/c.J;D=(c.kt*qb+bounds[1]*wb)*h/c.J;lo=w-D;hi=w+D
 ps=np.array([bounds[0]*lo,bounds[0]*hi,bounds[1]*lo,bounds[1]*hi]);al=(c.kt*np.minimum(q,uq)-ps.max(axis=0))/c.J;au=(c.kt*np.maximum(q,uq)-ps.min(axis=0))/c.J
 return np.maximum(lo,w+np.minimum(0,al)*h),np.minimum(hi,w+np.maximum(0,au)*h)

def endpoint_box(y,ref,h,c,bounds,wc=1000.):
 n=len(c.omega);w=y[:n];cur=y[n:3*n];d=cur-ref;ie=ref+d*np.exp(-wc*h)
 wl,wh=speed_box(y,ref,h,c,bounds,wc)
 products=np.array([bounds[0]*wl,bounds[0]*wh,bounds[1]*wl,bounds[1]*wh])
 qintegral=ref[n:]*h+d[n:]*(-np.expm1(-wc*h))/wc
 wlo=w+c.kt*qintegral/c.J-products.max(axis=0)*h/c.J
 whi=w+c.kt*qintegral/c.J-products.min(axis=0)*h/c.J
 wlo=np.maximum(wlo,wl);whi=np.minimum(whi,wh)
 mag=.75*c.L*(np.sum(cur*cur)-np.sum(ie*ie))
 dh=c.Rc*((ref[:n]**2+ref[n:]**2)*h+2*(ref[:n]*d[:n]+ref[n:]*d[n:])*(-np.expm1(-wc*h))/wc+(d[:n]**2+d[n:]**2)*(-np.expm1(-2*wc*h))/(2*wc))
 maxend=np.maximum(abs(wlo),abs(whi));minend=np.where(wlo*whi<=0,0,np.minimum(abs(wlo),abs(whi)))
 maxhold=np.maximum(abs(wl),abs(wh));minhold=np.where(wl*wh<=0,0,np.minimum(abs(wl),abs(wh)))
 elo=y[3*n]+.5*c.J*np.sum(w*w-maxend*maxend)+mag-dh.sum()-bounds[1]*np.sum(maxhold*maxhold)*h
 ehi=y[3*n]+.5*c.J*np.sum(w*w-minend*minend)+mag-dh.sum()-bounds[0]*np.sum(minhold*minhold)*h
 return wlo,whi,float(elo),float(ehi),ie,dh

def terminal_certificate(y,zeta,ref,h,c,bounds,g,tau=TAU,inom=INOM,wc=1000.,brake=0.):
 n=len(c.omega);wl,wh,elo,ehi,ie,dh=endpoint_box(y,ref,h,c,bounds,wc)
 L,H,V=reserves(wl,ie,c,wc);ll,hh,vv=reserves(wh,ie,c,wc)
 elo=elo-brake*h-np.maximum(L,ll).sum();ehi=ehi-brake*h+np.maximum(H,hh).sum();v=np.maximum(V,vv)
 ze=np.zeros(n) if zeta is None else thermal_step(zeta,y[n:3*n],ref,h,tau,inom,wc);r2=ie[:n]**2+ie[n:]**2
 return bool(elo>=c.Emin and ehi<=c.Emax and np.max(v)<=np.sqrt(max(0,2*elo/(3*c.capacitance))) and (zeta is None or np.max(ze+r2/(2*wc*tau*inom**2))<=1) and np.max(y[3*n+1:]+dh+c.Rc*r2/(2*wc)-np.array(c.heat_budget))<=1e-10)

class ReserveFilter:
 def __init__(self,bounds=(.7*F_EQ,1.3*F_EQ),tau=TAU,inom=INOM):
  self.bounds=bounds;self.tau=tau;self.inom=inom;self.calls=0;self.zero=0;self.changed=0;self.times=[]
 def __call__(self,y,zeta,ref,h,c,brake=0.):
  start=time.perf_counter();self.calls+=1;n=len(c.omega)
  candidates=[ref]+[np.r_[ref[:n]*s,ref[n:]]for s in [.75,.5,.25,0.]]+[np.r_[ref[:n],ref[n:]*s]for s in [.75,.5,.25,0.]]+[ref*s for s in [.75,.5,.25]]
  for idx,u in enumerate(candidates):
   g=guard(y,u,brake,h,c,self.bounds)
   im=np.maximum(y[n:2*n]**2+y[2*n:3*n]**2,u[:n]**2+u[n:]**2)
   g['i2t_safe']=bool(zeta is None or np.max(zeta+h*im/(self.tau*self.inom**2))<=1.)
   if all(g[k]for k in ['unsaturated','energy_safe','current_safe','heat_safe','i2t_safe']) and terminal_certificate(y,zeta,u,h,c,self.bounds,g,self.tau,self.inom,brake=brake):
    self.changed+=int(idx>0);self.times.append(time.perf_counter()-start);g['backup']=False;g['applied_brake']=brake;return u,g,idx>0
  cert=certificate(y,zeta,c,self.tau,self.inom)
  self.times.append(time.perf_counter()-start)
  if cert['valid']:
   self.zero+=1;self.changed+=1
   return np.zeros(2*n),dict(unsaturated=True,energy_safe=True,current_safe=True,heat_safe=True,i2t_safe=True,backup=True,applied_brake=0.,**cert),True
  return None,dict(backup_certificate=cert,i2t_safe=False),True
