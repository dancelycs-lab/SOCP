"""Published control principles adapted to the common shared-bus plant."""
import numpy as np
from scipy import sparse
import clarabel
from shared_loss import command
from conic import solve,audit
from fast_hold import advance

class Dual:
 def __init__(self):self.z=0.
 def __call__(self,y,t,c,ff,h=.001):
  n=len(c.omega);wc=1000.;P=c.brake_limit if t<.5 else min(100.,c.brake_limit);V=np.sqrt(max(1.,2*y[3*n]/c.capacitance));target=c.voltage_max-10
  rem=np.maximum(0.,np.array(c.heat_budget)-y[3*n+1:]-5*c.Rc*c.current_limit**2/wc);im=np.sqrt(np.minimum(c.Rc*c.current_limit**2,rem/max(.005,c.N*c.h-t))/c.Rc)
  dm=np.minimum(im,np.maximum(0.,40*np.pi*c.J*y[:n]/c.kt-ff*y[:n]/c.kt));a=max(1.,c.kt*np.dot(np.maximum(y[:n],0),dm)/(c.capacitance*max(V,1.)));cross=160.;phi=np.deg2rad(75);Ti=1/wc;tn=(np.tan(phi)+Ti*cross)/(cross*(1-np.tan(phi)*Ti*cross));ki=cross**2*np.sqrt(1+(Ti*cross)**2)/(a*np.sqrt(1+(tn*cross)**2));kp=ki*tn;e=target-V;raw=kp*e+self.z;frac=np.clip(raw,0,1)
  if 0<raw<1 or raw>=1 and e<0 or raw<=0 and e>0:self.z+=h*ki*e
  self.z=np.clip(self.z,-1,1);alpha=np.clip((y[3*n]-c.Emin)/(.5*(c.Emax-c.Emin)),0,1);q=-dm*frac;d=-np.sqrt(np.maximum(0,im*im-q*q))if V>805 and np.max(abs(y[:n]))>.01 else np.zeros(n)
  return np.r_[d*np.sqrt(alpha),q],min(P,max(0,wc*(y[3*n]-c.E0)))*alpha

def mcl(y,t,c,ff):
 cap=c.brake_limit if t<.5 else min(100.,c.brake_limit)
 u,b=command(y,np.array(c.heat_budget),c.N*c.h-t,c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance,c.Emin,c.Emax,c.current_limit,cap,1000.,ff,False)
 alpha=np.clip((y[3*len(c.omega)]-c.Emin)/(.5*(c.Emax-c.Emin)),0,1)
 return u,b*alpha
class CVO:
 def __init__(self,kp=.0023,ki=1.):
  self.kp=kp;self.ki=ki;self.z=0.;self.error=0.;self.theta=0.
 def __call__(self,y,t,c,ff,h):
  n=3;w=y[:n];I=c.current_limit;remaining=np.maximum(0,np.array(c.heat_budget)-y[10:13]-5*c.Rc*I**2/1000)
  im=np.sqrt(np.minimum(I**2,remaining/(max(.005,c.N*c.h-t)*c.Rc)))
  if np.max(abs(w))<.01:return np.zeros(6),0.
  raw=self.z+self.kp*self.error;self.theta=np.clip(raw,0,np.pi/2)
  if 0<raw<np.pi/2 or raw<=0 and self.error>0 or raw>=np.pi/2 and self.error<0:self.z+=h*self.ki*self.error
  self.z=np.clip(self.z,0,np.pi/2)
  q=-np.sign(w)*np.minimum(im*np.sin(self.theta),np.maximum(0,(40*np.pi*c.J-ff)*abs(w)/c.kt))
  alpha=np.clip((y[9]-c.Emin)/(.5*(c.Emax-c.Emin)),0,1)
  d=-np.sqrt(np.maximum(0,im**2-q**2))*np.sqrt(alpha)
  cap=c.brake_limit if t<.5 else min(100.,c.brake_limit)
  b=min(cap,max(0,1000*(y[9]-c.E0)))*alpha
  return np.r_[d,q],b
 def observe(self,samples,voltage,brake):
  # Current-vector projection, averaged over the completed hold. Correct
  # feedback polarity for id=-I cos(theta), iq=-I sin(theta).
  cur=samples[:,4:10];norm=np.sum(np.hypot(cur[:,:3],cur[:,3:]),axis=1)
  power=1.5*np.sum(cur*voltage,axis=1)+brake
  valid=norm>1e-6
  self.error=float(np.mean(power[valid]/(1.5*norm[valid])))if valid.any()else 0.

class HOCBF:
 def __init__(self,kE=100.,kH=20.):
  self.kE=kE;self.kH=kH;self.kI=500.;self.solver=None;self.changed=0;self.calls=0;self.zero=0;self.statuses={};self.min_psi1=np.inf;self.max_residual=0.;self.t=0.;self.limit=16.5
 def set_context(self,t,limit):self.t=t;self.limit=limit
 def __call__(self,y,zeta,ref,h,c,brake):
  self.calls+=1;n=3;dim=7;nl=10;rows=28;I=c.current_limit;wc=1000.;Pu=max(100.,c.brake_limit);Punit=n*c.Rc*I*I;Vunit=500.
  cur=y[3:9];r2=cur[:3]**2+cur[3:]**2;rem=np.array(c.heat_budget)-y[10:13]
  self.min_psi1=min(self.min_psi1,float(np.min(self.kH*rem-c.Rc*r2)))
  A=np.zeros((rows,dim));b=np.zeros(rows);rad=1.5*c.L*wc*I*cur
  base=c.Rc*r2.sum()+c.kt*np.dot(y[:3],cur[3:])-1.5*c.L*wc*r2.sum()
  A[0,:6]=-rad/Punit;A[0,-1]=-Pu/Punit;b[0]=(self.kE*(c.Emax-y[9])+base)/Punit
  A[1,:6]=rad/Punit;A[1,-1]=Pu/Punit;b[1]=(self.kE*(y[9]-c.Emin)-base)/Punit
  for i in range(n):
   A[2+i,i]=cur[i]/I;A[2+i,3+i]=cur[3+i]/I;b[2+i]=(r2[i]+self.kI/(2*wc)*(I*I-r2[i]))/(I*I)
   A[5+i,i]=cur[i]/I;A[5+i,3+i]=cur[3+i]/I;b[5+i]=((1-self.kH/wc)*r2[i]+self.kH**2*rem[i]/(2*c.Rc*wc))/(I*I)
  cap=c.brake_limit if self.t<.5 else min(100.,c.brake_limit)
  A[8,-1]=-1;A[9,-1]=1;b[9]=cap/Pu
  for i in range(n):
   row=nl+3*i;b[row]=self.limit/I;A[row+1,i]=-1;A[row+2,3+i]=-1
   row=nl+9+3*i;w=y[i];d=cur[i];q=cur[3+i]
   b[row]=np.sqrt(max(1.,2*y[9]/c.capacitance))/(np.sqrt(3)*Vunit)
   b[row+1]=(c.Rs*d-c.poles*w*c.L*q-c.L*wc*d)/Vunit
   b[row+2]=(c.Rs*q+c.poles*w*(c.L*d+c.flux)-c.L*wc*q)/Vunit
   A[row+1,i]=-c.L*wc*I/Vunit;A[row+2,3+i]=-c.L*wc*I/Vunit
  scale=np.maximum(1e-12,np.maximum(np.max(abs(A[:nl]),axis=1),abs(b[:nl])))
  A[:nl]/=scale[:,None];b[:nl]/=scale
  qvec=np.r_[-ref/I,-.01*brake/Pu]
  aa=sparse.csc_matrix((A.T.ravel(),np.tile(np.arange(rows),dim),np.arange(dim+1)*rows),shape=(rows,dim))
  if self.solver is None:
   settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=80;settings.max_threads=1;settings.tol_gap_abs=1e-9;settings.tol_feas=1e-9;settings.tol_gap_rel=1e-9
   self.solver=clarabel.DefaultSolver(sparse.diags(np.r_[np.ones(6),.01],format='csc'),qvec,aa,b,[clarabel.NonnegativeConeT(nl)]+[clarabel.SecondOrderConeT(3)for _ in range(6)],settings)
  else:self.solver.update(q=qvec,A=aa,b=b)
  sol=self.solver.solve();key=str(sol.status);self.statuses[key]=self.statuses.get(key,0)+1
  if key not in ['Solved','AlmostSolved']:return None,{'solver_status':key},False
  x=np.array(sol.x);slack=b-A@x
  residual=max(0.,-min(slack[:nl]),max(np.linalg.norm(slack[j+1:j+3])-slack[j]for j in range(nl,rows,3)))
  self.max_residual=max(self.max_residual,float(residual))
  if residual>1e-6:return None,{'solver_status':'residual_rejected'},False
  u=x[:6]*I;bb=float(np.clip(x[-1]*Pu,0,cap));changed=np.max(abs(u-ref))>1e-5 or abs(bb-brake)>1e-5;self.changed+=int(changed)
  return u,{'applied_brake':bb,'solver_status':key},changed
