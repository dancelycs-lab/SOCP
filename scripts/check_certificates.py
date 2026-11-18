"""Independent full-dq ODE checks and adversarial commands for the final filter."""
from context import mod,CASES,OUT
import numpy as np,json,time
from scipy.integrate import solve_ivp
from backup import certificate,endpoint_box
from compiled_backup import CompiledReserveFilter as ReserveFilter
rng=np.random.default_rng(5110902);c=mod.Case(**CASES['h0_p0']);limits=np.array(c.heat_budget);bounds=(.00623,.01157);results=[];fail=[];times=[]
keys=['lower_margin','upper_margin','voltage_margin','current_margin','heat_margin']
# Warm numerical kernels before timed calls.
y=np.zeros(13);y[9]=c.E0;ReserveFilter(bounds)(y,None,np.zeros(6),.001,c,0.)
while len(results)<240:
 y=np.zeros(13);y[:3]=rng.uniform(-210,210,3);rad=rng.uniform(0,.9*c.current_limit,3);angle=rng.uniform(-np.pi,np.pi,3);y[3:9]=np.r_[rad*np.cos(angle),rad*np.sin(angle)];y[9]=rng.uniform(c.Emin,c.Emax);y[10:]=limits-c.Rc*rad**2/2000-rng.uniform(.0001,.1,3)
 if not certificate(y,None,c)['valid']:continue
 j=len(results);rref=rng.uniform(.8,.995,3)*c.current_limit;aref=rng.uniform(-np.pi,np.pi,3);ref=np.r_[rref*np.cos(aref),rref*np.sin(aref)];h=.001;b=0.;filt=ReserveFilter(bounds);tic=time.perf_counter_ns();u,g,changed=filt(y,None,ref,h,c,b);times.append((time.perf_counter_ns()-tic)/1e6);assert u is not None
 phase=rng.uniform(0,2*np.pi,3)
 def rhs(t,x):
  w=x[:3];d=x[3:6];q=x[6:9];V=np.sqrt(max(1e-12,2*x[9]/c.capacitance));vd=c.Rs*d-c.poles*w*c.L*q+c.L*1000*(u[:3]-d);vq=c.Rs*q+c.poles*w*(c.L*d+c.flux)+c.L*1000*(u[3:]-q);scale=np.minimum(1.,V/np.sqrt(3)/np.maximum(np.hypot(vd,vq),1e-30));vd*=scale;vq*=scale;f=.0089*(1+.3*np.sin(2*np.pi*71*t+phase))
  return np.r_[(c.kt*q-f*w)/c.J,(vd+c.poles*w*c.L*q-c.Rs*d)/c.L,(vq-c.poles*w*(c.L*d+c.flux)-c.Rs*q)/c.L,-1.5*np.sum(vd*d+vq*q)-g['applied_brake'],c.Rc*(d*d+q*q)]
 ts=np.linspace(0,h,41);sol=solve_ivp(rhs,(0,h),y,method='DOP853',t_eval=ts,rtol=1e-10,atol=1e-11);assert sol.success;x=sol.y.T;end=certificate(x[-1],None,c);wl,wh,el,eh,ie,dh=endpoint_box(y,u,h,c,bounds);el-=g['applied_brake']*h;eh-=g['applied_brake']*h
 violation=max(0,c.Emin-x[:,9].min(),x[:,9].max()-c.Emax,np.hypot(x[:,3:6],x[:,6:9]).max()-c.current_limit,(x[:,10:]-limits).max(),-min(end[k]for k in keys),max(wl-x[-1,:3]),max(x[-1,:3]-wh),el-x[-1,9],x[-1,9]-eh)
 if violation>1e-7:fail.append(dict(case=j,violation=float(violation)))
 results.append(dict(case=j,changed=bool(changed),backup=g['backup'],max_violation=float(violation),minimum_endpoint_margin=min(end[k]for k in keys)))
report=dict(seed=5110902,description='Admissible current references, tight copper reserves, matched full dq plant; zero external absorption.',cases=len(results),modified=sum(r['changed']for r in results),backup=sum(r['backup']for r in results),failures=fail,max_violation=max(r['max_violation']for r in results),filter_time_ms=dict(median=float(np.median(times)),p99=float(np.percentile(times,99)),maximum=max(times),over_1ms=sum(t>1 for t in times)))
(OUT/'certificate_check.json').write_text(json.dumps(report,indent=2));assert not fail,fail;print(json.dumps(report,indent=2))
