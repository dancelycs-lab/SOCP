"""Drive parameters and trajectory metrics."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Case:
    omega: tuple=(150.,120.,0.)
    heat_budget: tuple=(10.,40.,80.)
    h: float=.02
    N: int=20
    J: float=.00774
    Rs: float=.25
    L: float=.0048
    poles: int=4
    flux: float=.32
    current_limit: float=16.5
    capacitance: float=.00047
    voltage0: float=800.
    voltage_min: float=650.
    voltage_max: float=840.
    brake_limit: float=100.
    initial_magnetic: float=0.
    @property
    def kt(self):return 1.5*self.poles*self.flux
    @property
    def Rc(self):return 1.5*self.Rs
    @property
    def mag_alpha(self):return self.L/(2*self.Rs)
    @property
    def initial_K(self):return .5*self.J*np.array(self.omega)**2
    @property
    def E0(self):return .5*self.capacitance*self.voltage0**2+self.initial_magnetic
    @property
    def Emin(self):return .5*self.capacitance*self.voltage_min**2
    @property
    def Emax(self):return .5*self.capacitance*self.voltage_max**2
def metrics(c,trace):
 n=len(c.omega);t=trace[:,0];y=trace[:,1:-1];V=np.sqrt(np.maximum(0,2*y[:,3*n]/c.capacitance));speed=np.max(abs(y[:,:n]),axis=1)
 idx=np.flatnonzero(speed>1.)
 settled=float(t[idx[-1]+1]) if len(idx) and idx[-1]<len(t)-1 else (0. if not len(idx) else None)
 m=dict(max_voltage=float(V.max()),min_voltage=float(V.min()),max_current=float(np.sqrt(y[:,n:2*n]**2+y[:,2*n:3*n]**2).max()),max_heat_excess=float(np.max(y[-1,3*n+1:]-np.array(c.heat_budget))),deadline_speed=float(speed[-1]),minimum_speed=float(y[:,:n].min()),settling_time=settled,kinetic_integral=float(np.trapezoid(.5*c.J*np.sum(y[:,:n]**2,axis=1),t)),angular_travel_max=float(np.max(np.trapezoid(abs(y[:,:n]),t,axis=0))),voltage_saturation_fraction=float(trace[:,-1].mean()),copper_energy_total=float(np.sum(y[-1,3*n+1:])))
 m['physical_safe']=bool(m['max_voltage']<=c.voltage_max+1e-3 and m['min_voltage']>=c.voltage_min-1e-3 and m['max_current']<=c.current_limit+1e-3 and m['max_heat_excess']<=1e-4)
 m['success']=m['physical_safe'] and m['deadline_speed']<1.
 return m
