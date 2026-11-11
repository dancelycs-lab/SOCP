"""Full dq plant, integration and schedule precomputation."""
import time
from dataclasses import replace
import numpy as np
from numba import njit
import baselines as pub
from backup import ReserveFilter,certificate,guard
from model import Case,metrics

@njit(cache=True)
def rhs(t,y,u,b,nom,act,mode,T,aux):
 n=3;J,R,L,p,psi,C=act;Jn,Rn,Ln,pn,psin,Cn=nom
 f=np.empty(n)
 for k in range(n):
  if mode==1:f[k]=.0089*(1+.3*np.sin(2*np.pi*3*t+2*np.pi*k/3))
  elif mode==2:f[k]=.0089*(1.3 if t<(k+1)*.2*T else .7)
  else:f[k]=.0089
 out=np.zeros_like(y);V=np.sqrt(max(1e-9,2*y[9]/C));lim=V/np.sqrt(3);sat=0.
 for k in range(n):
  w=y[k];d=y[3+k];q=y[6+k]
  vd=Rn*d-pn*w*Ln*q+Ln*1000*(u[k]-d)
  vq=Rn*q+pn*w*(Ln*d+psin)+Ln*1000*(u[3+k]-q)
  vn=np.hypot(vd,vq)
  if vn>lim:vd*=lim/vn;vq*=lim/vn;sat=1.
  out[k]=(1.5*p*psi*q-f[k]*w)/J
  out[3+k]=(vd+p*w*L*q-R*d)/L
  out[6+k]=(vq-p*w*(L*d+psi)-R*q)/L
  out[9]-=1.5*(vd*d+vq*q)
  out[10+k]=1.5*R*(d*d+q*q)
  out[13]+=f[k]*w*w
 out[9]-=b;out[14]=b
 return out,sat

@njit(cache=True)
def integrate(y,u,b,t,h,nom,act,mode,T,dtmax):
 ns=int(np.ceil(h/dtmax));dt=h/ns;out=np.zeros((ns,len(y)+2));satmax=0.
 for j in range(ns):
  a,sa=rhs(t,y,u,b,nom,act,mode,T,0)
  bb,sb=rhs(t+dt/2,y+a*dt/2,u,b,nom,act,mode,T,0)
  cc,sc=rhs(t+dt/2,y+bb*dt/2,u,b,nom,act,mode,T,0)
  dd,sd=rhs(t+dt,y+cc*dt,u,b,nom,act,mode,T,0)
  y=y+dt*(a+2*bb+2*cc+dd)/6;t+=dt
  out[j,0]=t;out[j,1:-1]=y;out[j,-1]=max(sa,sb,sc,sd);satmax=max(satmax,out[j,-1])
 return y,out

def plan(c,bounds):
 n=3;T=c.N*c.h;N=c.N-2;hh=(T-.025-2*c.h)/N;ff=.0089
 profile=np.where(.025+(np.arange(N)+1)*hh<=.5+1e-12,c.brake_limit,min(c.brake_limit,100.))
 pc=replace(c,N=N,h=hh,voltage_max=c.voltage_max-10,heat_budget=tuple(np.array(c.heat_budget)-5*c.Rc*c.current_limit**2/1000.))
 cache=pub.solve(pc,ff,profile,prepare_only=True,friction_bounds=bounds)
 fy=np.zeros(13);fy[:3]=c.omega;fy[9]=c.E0;filt=ReserveFilter(bounds);start=time.perf_counter()
 for j in range(25):
  u,b=pub.mcl(fy,j*.001,c,ff);u,g,_=filt(fy,None,u,.001,c,b)
  if u is None:return None
  fy=pub.advance(fy,u,g['applied_brake'],.001,c,ff)
 pc=replace(pc,omega=tuple(np.maximum(0,fy[:3])),voltage0=float(np.sqrt(2*fy[9]/c.capacitance)),initial_magnetic=float(.75*c.L*np.sum(fy[3:9]**2)),heat_budget=tuple(np.array(pc.heat_budget)-fy[10:]))
 r=pub.solve(pc,ff,profile,prepared=cache,friction_bounds=bounds)
 if 'K'not in r:return None
 check,pol=pub.audit(pc,r,ff)
 if not check['valid']:return None
 return dict(h=hh,N=N,omega=pol['omega'].tolist(),qacc=pol['qacc'].tolist(),loss=pol['loss'].tolist(),brake=r['brake'].tolist(),compute_ms=1000*(time.perf_counter()-start))

class HoldOnly:
 def __init__(self,bounds):self.bounds=bounds;self.changed=0;self.zero=0;self.calls=0
 def __call__(self,y,zeta,ref,h,c,brake):
  self.calls+=1;n=3;opts=[ref]+[np.r_[ref[:n]*s,ref[n:]]for s in [.75,.5,.25,0]]+[np.r_[ref[:n],ref[n:]*s]for s in [.75,.5,.25,0]]+[ref*s for s in [.75,.5,.25]]+[np.zeros(6)]
  for j,u in enumerate(opts):
   b=0 if j==len(opts)-1 else brake;g=guard(y,u,b,h,c,self.bounds)
   if all(g[k]for k in ['unsaturated','energy_safe','current_safe','heat_safe']):
    self.changed+=int(j>0);self.zero+=int(j==len(opts)-1);g['applied_brake']=b;return u,g,j>0
  return None,{},True
