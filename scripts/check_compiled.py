from context import mod,CASES,OUT
from backup import ReserveFilter,endpoint_box
from compiled_backup import CompiledReserveFilter,endpoint_box as compiled_endpoint,parameters
import numpy as np,json
rng=np.random.default_rng(5110903);c=mod.Case(**CASES['h0_p0']);bounds=(.00623,.01157);worst=0.;changes=0;backups=0
for j in range(2000):
 y=np.zeros(13);y[:3]=rng.uniform(-210,210,3);rad=rng.uniform(0,.99*c.current_limit,3);ang=rng.uniform(-np.pi,np.pi,3);y[3:9]=np.r_[rad*np.cos(ang),rad*np.sin(ang)];y[9]=rng.uniform(c.Emin,c.Emax);y[10:]=rng.uniform(0,1,3)*np.array(c.heat_budget)
 ref=rng.uniform(-1.5,1.5,6)*c.current_limit;h=rng.uniform(.00005,.001);b=0.;a=ReserveFilter(bounds);cc=CompiledReserveFilter(bounds);u,g,z=a(y,None,ref,h,c,b);v,gg,zz=cc(y,None,ref,h,c,b)
 assert (u is None)==(v is None) and z==zz,(j,u,v)
 if u is not None:assert np.array_equal(u,v)and g['backup']==gg['backup'];changes+=int(z);backups+=int(g['backup'])
 e=endpoint_box(y,ref,h,c,bounds);ee=compiled_endpoint(y,ref,h,parameters(c),bounds)
 for x,xx in zip(e,ee):worst=max(worst,float(np.max(abs(np.asarray(x)-np.asarray(xx)))))
assert worst<1e-10,worst
out=dict(seed=5110903,cases=2000,decision_mismatches=0,maximum_endpoint_difference=worst,changed=changes,backups=backups);(OUT/'compiled_check.json').write_text(json.dumps(out,indent=2));print(out)
