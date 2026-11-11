"""Serial per-method online timing, including CVO completed-hold observation.
No plant integration, precomputed planning, or compilation in timed intervals.
"""
from context import OUT,ROOT,PLANS
import sampled_drive as C
from baselines import CVO,HOCBF
from pathlib import Path
import json,platform
import numpy as np
selected=json.loads((ROOT/'config/baseline_gains.json').read_text())
cases=json.loads((ROOT/'config/cases.json').read_text());plans=PLANS
out=OUT/'timing';out.mkdir(exist_ok=True)
def run(item,method,record):
 kwargs={}
 if method=='cvo':kwargs['nominal_policy']=CVO(**selected['cvo'])
 if method=='hocbf':kwargs['execution_filter']=HOCBF(**selected['hocbf'])
 pg=selected['pdo_smpcc'];kwargs['gains']=[pg[k] for k in ['Tc','l','T1','T2','rho']];kwargs['reference_tau']=pg['reference_tau']
 samples=[]
 inner='pdo_smpcc' if method in ['mcl_pdo','pdo_smpcc'] else 'dpcc' if method=='dpcc' else 'sampled_proposed'
 outer='mcl' if method=='mcl_pdo' else method if method in ['dual','mcl'] else 'conic'
 r,_,_,_=C.run(C.Case(**item['case']),plans[item['id']+'_interval'],method=inner,outer_policy=outer,mode=1,id_noise=1,component_times=samples if record else None,**kwargs)
 r.pop('method',None)
 return r,np.asarray(samples,dtype=np.int64)
methods=['conic','dual','mcl','cvo','mcl_pdo','hocbf','pdo_smpcc','dpcc']
for m in methods:run(cases[0],m,False)
raw={m:[]for m in methods};metrics=[]
for i,c in enumerate(cases):
 # Rotate order to reduce a fixed method/host-load order effect.
 order=methods[i%len(methods):]+methods[:i%len(methods)]
 for m in order:
  r,a=run(c,m,True);raw[m].append(a);metrics.append(dict(case_id=c['id'],method=m,**r));np.save(out/f"{c['id']}_{m}_ns.npy",a)
 print(i+1,len(cases),'cases complete',flush=True)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2));report={}
for m,chunks in raw.items():
 a=np.vstack(chunks)/1e6;report[m]={}
 for key,v in [('control_filter',a[:,0]),('post_hold',a[:,1]),('total',a.sum(axis=1))]:report[m][key]=dict(calls=len(v),median_ms=float(np.median(v)),p99_ms=float(np.percentile(v,99)),max_ms=float(np.max(v)),over_1ms=int(np.sum(v>1)))
report['cpu']=next((x.split(':',1)[1].strip()for x in (Path('/proc/cpuinfo').read_text().splitlines() if Path('/proc/cpuinfo').exists() else [])if x.startswith('model name')),platform.processor());report['python']=platform.python_version();(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

# Inner command, observer and reference update only; no plant integration.
import time
c=C.Case(**cases[0]['case']);nom=np.array([c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance])
y=np.zeros(15);y[:3]=[120,90,0];y[9]=.5*c.capacitance*800**2
ref=np.array([-8.,-7.,-10.,-5.,-4.,0.]);gains=np.array([.008,1500.,.008,.008,.5])
@C.njit(cache=True)
def inner_cycle(y,ref,nom,method,state,gains):
 dt=5e-5
 v=C.controller_voltage(y,ref,nom,method,state,gains,dt,.001)
 limit=.99*np.sqrt(2*y[9]/nom[5]/3)
 for k in range(3):
  norm=np.hypot(v[k],v[k+3])
  if norm>limit:v[k]*=limit/norm;v[k+3]*=limit/norm
 if method==1:
  g=C.reaching(y[3:9]-state[:6],gains[0],.5)
  state[:6]+=dt*(v/nom[2]+state[6:12]+g);state[6:12]+=dt*gains[1]*g
  desired=state[18:24].copy()
  state[12:18]+=dt*C.reaching(desired-y[3:9],gains[2],gains[4])
  state[18:24]=ref+(desired-ref)*np.exp(-dt/.001)
 return v
report['inner_loop']={}
for name,code in [('sampled_regulator',0),('pdo_smpcc',1),('dpcc',2)]:
 state=np.zeros(24);state[6:12]=C.nominal_disturbance(y,nom)
 inner_cycle(y,ref,nom,code,state.copy(),gains)
 samples=[]
 for _ in range(20000):
  z=state.copy();tic=time.perf_counter_ns();inner_cycle(y,ref,nom,code,z,gains);samples.append((time.perf_counter_ns()-tic)/1000)
 report['inner_loop'][name]=dict(calls=len(samples),median_us=float(np.median(samples)),p99_us=float(np.percentile(samples,99)),maximum_us=float(max(samples)))
(out/'summary.json').write_text(json.dumps(report,indent=2))
