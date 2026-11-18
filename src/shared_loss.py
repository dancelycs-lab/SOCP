"""Shared-bus loss-maximizing braking allocation."""
import numpy as np
from numba import njit

@njit(cache=True)
def command(y,Hlim,Tleft,J,R,L,p,flux,C,Emin,Emax,Imax,Pmax,wc,f,energy_to_go=False):
 n=len(Hlim);rho=1.5*R;kt=1.5*p*flux;ell=np.zeros(n);demand=np.zeros(n);ref=np.zeros(2*n)
 remaining=np.maximum(0.,Hlim-y[3*n+1:]-5*rho*Imax**2/wc)
 active=np.max(np.abs(y[:n]))>.01;tau=max(.005,Tleft)
 Etarget=.5*C*(np.sqrt(2*Emax/C)-10.)**2
 if energy_to_go and active:
  K=.5*J*np.sum(y[:n]**2);required=max(0.,K-max(0.,Etarget-y[3*n]));lo=0.;hi=tau
  for _ in range(24):
   mid=(lo+hi)/2
   capacity=np.sum(np.minimum(remaining,rho*Imax**2*mid))+Pmax*mid+2*f*K*mid/(3*J)
   if capacity>=required:hi=mid
   else:lo=mid
  tau=max(.005,hi)
 power=0.
 for i in range(n):
  ell[i]=min(rho*Imax**2,remaining[i]/tau) if active else 0.
  im=np.sqrt(ell[i]/rho);demand[i]=max(-im,min(im,-40*np.pi*J*y[i]/kt+f*y[i]/kt));power-=kt*y[i]*demand[i]
 aug=y[3*n]+.75*L*np.sum(y[n:3*n]**2);target=Etarget+L/(2*R)*np.sum(ell)
 available=np.sum(ell)+Pmax+100.*(target-aug);scale=min(1.,max(0.,available/max(1e-12,power)))
 alpha=min(1.,max(0.,(y[3*n]-Emin)/(.5*(Emax-Emin))))
 for i in range(n):
  q=demand[i]*scale;ref[n+i]=q;ref[i]=-np.sqrt(max(0.,ell[i]/rho-q*q))*np.sqrt(alpha)
 brake=min(Pmax,max(0.,power*scale-np.sum(ell)+100.*(aug-target)))
 return ref,brake
