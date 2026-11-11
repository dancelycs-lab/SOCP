"""Refine the voltage-driven plant while keeping both controller periods fixed."""
from context import OUT,ROOT,PLANS
import sampled_drive as C
import json
scenarios=json.loads((ROOT/'config/cases.json').read_text())
rows=json.loads((OUT/'parameters/metrics.json').read_text())
tests={'R_minus':[1,.95,1,1,1,1],'R_plus':[1,1.05,1,1,1,1],
       'psi_minus':[1,1,1,1,.98,1],'psi_plus':[1,1,1,1,1.02,1],
       'combined_pos':[1.05,1.05,.95,1,.98,.95],'combined_neg':[.95,.95,1.05,1,1.02,1.05]}
results=[]
for item in [scenarios[0],scenarios[-1]]:
 for condition,scale in tests.items():
  for method in ['conic','mcl_pdo','pdo_smpcc','dpcc']:
   inner='sampled_proposed' if method=='conic' else 'pdo_smpcc' if method=='mcl_pdo' else method
   r,_,_,_=C.run(C.Case(**item['case']),PLANS[item['id']+'_interval'],scale=scale,method=inner,
                outer_policy='mcl' if method=='mcl_pdo' else 'conic',plant_dt=5e-6)
   old=next(x for x in rows if x['case_id']==item['id'] and x['condition']==condition and x['method']==method)
   results.append(dict(case_id=item['id'],condition=condition,method=method,success=r['success'],
                       physical_safe=r['physical_safe'],same_outcome=r['success']==old['success'],
                       J_difference=abs(old['kinetic_integral']-r['kinetic_integral'])))
assert all(x['same_outcome'] and x['physical_safe'] for x in results)
out=dict(plant_steps_us=[10,5],voltage_hold_us=50,runs=results,max_J_difference=max(x['J_difference'] for x in results))
(OUT/'refinement.json').write_text(json.dumps(out,indent=2));print(out)
