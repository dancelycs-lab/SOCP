"""Optional exponentially weighted current-energy state."""
import numpy as np
TAU=28.5
INOM=2/np.sqrt(3)*2.47
F_EQ=.0315*.192/(10700*2*np.pi/60)

def thermal_step(zeta,i,ref,dt,tau=TAU,inom=INOM,wc=1000.):
 n=len(zeta);d=i-ref;a=1/tau;ea=np.exp(-a*dt)
 def g(b):return (np.exp(-b*dt)-ea)/(a-b)
 u2=ref[:n]**2+ref[n:]**2;ud=ref[:n]*d[:n]+ref[n:]*d[n:];d2=d[:n]**2+d[n:]**2
 return ea*zeta+(u2*g(0)+2*ud*g(wc)+d2*g(2*wc))/(tau*inom**2)
