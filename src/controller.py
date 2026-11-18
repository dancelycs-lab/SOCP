"""Final integrated controller and closed-loop experiment."""
import time
from dataclasses import replace
import numpy as np
from scipy.integrate import simpson
import plant
from plant import integrate,pub,ReserveFilter,HoldOnly,Case,metrics
from estimator import ElectricalEstimator,applied_voltage
from compiled_backup import CompiledReserveFilter as ReserveFilter

def run(c,pl,method='conic',mode=1,stress='none',ablation='none',dtmax=5e-5,actual_scale=None,timing_ns=None,adapt=False,reserve=0.,id_noise=0.,weight_floor=1e-7,startup_reserve=False,component_times=None,nominal_policy=None,execution_filter=None):
 nom=np.array([c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance]);act=nom.copy();amp=0.
 if stress=='mismatch_pos':act*=np.array([1.05,1.05,.95,1.,.98,.95])
 if stress=='mismatch_neg':act*=np.array([.95,.95,1.05,1.,1.02,1.05])
 if actual_scale is not None:act=nom*np.asarray(actual_scale)
 if stress=='sensor_1':amp=1.
 if stress=='sensor_2':amp=2.
 actual=replace(c,J=act[0],Rs=act[1],L=act[2],poles=int(act[3]),flux=act[4],capacitance=act[5])
 base=c;estimate=ElectricalEstimator(c,weight_floor);estimate_history=[];cloop=nom.copy()
 bounds=(.0089,.0089)if ablation=='nominal_friction'else(.00623,.01157)
 filt=HoldOnly(bounds)if ablation=='hold_only'else ReserveFilter(bounds);dual=pub.Dual();observer=None;enclosure_error=0.
 if execution_filter is not None:filt=execution_filter
 y=np.zeros(15);y[:3]=c.omega;y[9]=.5*act[5]*c.voltage0**2;t=0.;T=c.N*c.h;rows=[np.r_[0.,y,0.][None,:]];failed=None
 while t<T-1e-10:
  control_start=time.perf_counter_ns() if component_times is not None else 0
  c=replace(base,Rs=cloop[1],L=cloop[2],flux=cloop[4])
  obs=y[:13].copy();voltage=np.sqrt(2*y[9]/act[5]);obs[9]=.5*c.capacitance*voltage**2
  if amp:
   obs[:3]+=amp*.1*np.sin(2*np.pi*17*t+np.arange(3)*1.1)
   obs[3:9]+=amp*.01*np.sin(2*np.pi*31*t+np.arange(6)*.7)
   obs[9]=.5*c.capacitance*(voltage+amp*.1*np.sin(2*np.pi*23*t+.2))**2
  observation_ns=0
  h=min(.001,T-t)
  if nominal_policy is not None:u,b=nominal_policy(obs,t,c,bounds[0],h)
  elif method=='conic'and pl is not None and t>=.025-1e-10:
   tt=max(0,t-.025);k=min(pl['N'],int((tt+1e-10)/pl['h']));local=tt-k*pl['h']
   if k<pl['N']:
    h=min(h,pl['h']-local);w0=np.array(pl['omega'])[:,k];acc=np.array(pl['qacc'])[:,k];ell=np.array(pl['loss'])[:,k];b=pl['brake'][k];wr=w0+base.kt*acc*local/base.J;acc=acc*base.kt/c.kt
   else:acc=np.zeros(3);ell=np.zeros(3);wr=np.zeros(3);b=0.
   q=np.clip(acc+.0089*obs[:3]/c.kt+50*c.J*(wr-obs[:3])/c.kt,-c.current_limit,c.current_limit);im=np.minimum(c.current_limit,np.sqrt(np.maximum(0,ell/c.Rc)));d=-np.sqrt(np.maximum(0,im*im-q*q));alpha=np.clip((obs[9]-c.Emin)/(.5*(c.Emax-c.Emin)),0,1);u=np.r_[d*np.sqrt(alpha),q];b*=alpha
  elif method=='dual':u,b=dual(obs,t,c,bounds[0],h)
  else:u,b=pub.mcl(obs,t,c,.0089 if method=='conic'else bounds[0])
  if t<.025-1e-10:h=min(h,.025-t)
  if ablation=='no_auxiliary':u[2]=0.;u[5]=0.
  clock=time.perf_counter_ns() if timing_ns is not None else 0
  limit=base.current_limit*(.90 if (adapt or startup_reserve) and t<.005 else 1-reserve)
  norms=np.hypot(u[:3],u[3:]);scale=np.minimum(1.,limit/np.maximum(norms,1e-30));u=np.r_[u[:3]*scale,u[3:]*scale]
  if hasattr(filt,'set_context'):filt.set_context(t,limit)
  u,g,_=filt(obs,None,u,h,c,b)
  if timing_ns is not None:timing_ns.append(observation_ns+time.perf_counter_ns()-clock)
  if u is None:failed='certificate_rejected';break
  control_ns=time.perf_counter_ns()-control_start if component_times is not None else 0
  y0=y.copy();start_t=t
  y,tr=integrate(y,u,g['applied_brake'],t,h,cloop,act,mode,T,dtmax);t+=h;rows.append(tr)
  identification_start=time.perf_counter_ns() if component_times is not None else 0
  if nominal_policy is not None and hasattr(nominal_policy,'observe'):
   samples=np.vstack([np.r_[start_t,y0,0.],tr]);voltage_samples=np.sqrt(2*samples[:,10]/act[5])
   nominal_policy.observe(samples,applied_voltage(samples,u,cloop,voltage_samples),g['applied_brake'])
  if adapt:
   samples=np.vstack([np.r_[start_t,y0,0.],tr]);voltage_samples=np.sqrt(2*samples[:,10]/act[5]) # physical voltage sensor
   volts=applied_voltage(samples,u,cloop,voltage_samples)
   estimate.update(samples,volts,noise=id_noise)
   if estimate.accepted:cloop[[1,2,4]]=estimate.value
   estimate_history.append([t,*estimate.value,estimate.rank,estimate.residual])
  if component_times is not None:component_times.append([control_ns,time.perf_counter_ns()-identification_start])
  if observer is not None:enclosure_error=max(enclosure_error,float(np.max(observer.lo-y[:3])),float(np.max(y[:3]-observer.hi)),float(observer.elo-y[9]),float(y[9]-observer.ehi),float(np.max(abs(observer.cur-y[3:9]))))
 tr=np.vstack(rows);base=np.c_[tr[:,:14],tr[:,-1]];m=metrics(actual,base)
 total=tr[:,10]+.5*act[0]*np.sum(tr[:,1:4]**2,axis=1)+.75*act[2]*np.sum(tr[:,4:10]**2,axis=1)+np.sum(tr[:,11:14],axis=1)+tr[:,14]+tr[:,15]
 m.update(completed=failed is None,status=failed or 'completed',duration=t,success=bool(m['success']and failed is None),filter_changes=filt.changed,filter_calls=filt.calls,backup_calls=filt.zero,energy_balance_error=float(np.max(abs(total-total[0]))),plan_available=pl is not None,enclosure_error=enclosure_error)
 if failed:m['kinetic_integral_complete']=None
 else:m['kinetic_integral_complete']=m['kinetic_integral']
 m['adaptation_updates']=len(estimate_history);m['parameter_estimate']=estimate.value.tolist();m['last_regression_residual']=float(estimate.residual);m['max_estimation_condition']=float(estimate.max_condition)
 return m,tr,np.asarray(estimate_history)
