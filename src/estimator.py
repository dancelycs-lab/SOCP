"""Linear integral regression from measured currents, speeds and applied voltages."""
import numpy as np
from scipy.integrate import simpson

def applied_voltage(samples,u,nom,bus_voltage):
 # Reconstruct applied voltage from the known command law and measured bus
 # voltage. Actual motor parameters are not inputs.
 J,R,L,p,psi,_=nom;w=samples[:,1:4];d=samples[:,4:7];q=samples[:,7:10]
 vd=R*d-p*w*L*q+L*1000*(u[:3]-d)
 vq=R*q+p*w*(L*d+psi)+L*1000*(u[3:]-q)
 lim=bus_voltage/np.sqrt(3);scale=np.minimum(1.,lim[:,None]/np.maximum(np.hypot(vd,vq),1e-30))
 return np.c_[vd*scale,vq*scale]

class ElectricalEstimator:
 def __init__(self,c,weight_floor=1e-7):
  self.weight_floor=weight_floor
  self.scale=np.array([c.Rs,c.L,c.flux]);self.value=self.scale.copy();self.p=c.poles;self.qr=np.zeros((0,3));self.qt=np.zeros(0);self.accepted=False;self.rank=0;self.residual=0.;self.max_condition=0.
 def update(self,samples,voltage,noise=0.):
  if noise:
   samples=samples.copy();voltage=voltage.copy();tt=samples[:,0]
   samples[:,1:4]+=noise*.1*np.sin(2*np.pi*17*tt[:,None]+np.arange(3)[None,:]*1.1)
   samples[:,4:10]+=noise*.01*np.sin(2*np.pi*31*tt[:,None]+np.arange(6)[None,:]*.7)
   voltage+=noise*.1*np.sin(2*np.pi*23*tt[:,None]+np.arange(6)[None,:]*.9)
  t=samples[:,0];w=samples[:,1:4];d=samples[:,4:7];q=samples[:,7:10]
  integ=lambda x:simpson(x,x=t,axis=0)
  Id,Iq,Iw,Iwd,Iwq=map(integ,[d,q,w,w*d,w*q]);Vd,Vq=integ(voltage[:,:3]),integ(voltage[:,3:])
  A=np.zeros((6,3));A[:3,0]=Id;A[:3,1]=d[-1]-d[0]-self.p*Iwq
  A[3:,0]=Iq;A[3:,1]=q[-1]-q[0]+self.p*Iwd;A[3:,2]=self.p*Iw
  # Rows are normalized in voltage-integral units; parameters in nominal units.
  A=A*self.scale;rhs=np.r_[Vd,Vq];weights=np.maximum(np.linalg.norm(A,axis=1),self.weight_floor)
  newA=A/weights[:,None];newb=rhs/weights
  mat=np.vstack([self.qr,newA]);target=np.r_[self.qt,newb]
  Q,R=np.linalg.qr(mat,mode='reduced');self.qr=R;self.qt=Q.T@target
  sol,_,rank,svals=np.linalg.lstsq(self.qr,self.qt,rcond=1e-9)
  self.rank=int(rank);self.max_condition=max(self.max_condition,float(svals[0]/max(svals[-1],1e-30)));self.residual=float(np.max(abs(newA@sol-newb)))
  if rank==3 and svals[-1]/svals[0]>1e-5 and np.all(sol>np.array([.8,.8,.9])) and np.all(sol<np.array([1.2,1.2,1.1])):
   self.value=sol*self.scale;self.accepted=True
