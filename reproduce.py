"""Run the shared-bus braking experiments and write numerical results."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import sampled_drive as controller
from baselines import CVO, HOCBF

SCALES = dict(R_minus=[1,.95,1,1,1,1], R_plus=[1,1.05,1,1,1,1],
              psi_minus=[1,1,1,1,.98,1], psi_plus=[1,1,1,1,1.02,1],
              combined_pos=[1.05,1.05,.95,1,.98,.95],
              combined_neg=[.95,.95,1.05,1,1.02,1.05])
METHODS = ['conic', 'dual', 'mcl', 'cvo', 'mcl_pdo', 'hocbf', 'pdo_smpcc', 'dpcc']
def load(name):
    return json.loads((ROOT / 'config' / name).read_text())
def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')
def execute(job):
    suite, item, plan, method, mode, condition, out = job
    gains = load('baseline_gains.json')
    kw = dict(adapt=suite!='no_identification', mode=mode,
              id_noise=1 if suite=='noisy' else 0,
              ablation='no_auxiliary' if suite=='auxiliary' else 'none')
    if condition: kw['scale'] = SCALES[condition]
    if method=='cvo': kw['nominal_policy'] = CVO(**gains['cvo'])
    if method=='hocbf': kw['execution_filter'] = HOCBF(**gains['hocbf'])
    pg=gains['pdo_smpcc']
    kw['gains']=[pg[k] for k in ['Tc','l','T1','T2','rho']]
    kw['reference_tau']=pg['reference_tau']
    kw['outer_policy']='mcl' if method=='mcl_pdo' else method if method in ['dual','mcl'] else 'conic'
    inner='pdo_smpcc' if method=='mcl_pdo' else method if method in ['pdo_smpcc','dpcc'] else 'sampled_proposed'
    m, tr, ref, est = controller.run(controller.Case(**item['case']), plan, method=inner, **kw)
    m.pop('method',None)
    row = dict(suite=suite, case_id=item['id'], method=method, friction_mode=mode, **m)
    if condition: row['condition']=condition
    if method=='hocbf':
        filt=kw['execution_filter']
        row.update(solver_status_counts=filt.statuses, min_psi1=filt.min_psi1,
                   max_solver_residual=filt.max_residual)
    if item['id'] in ['h0_p0','h11_p100','n0_p0','n7_p100']:
        tag=f"{item['id']}_{condition}_{method}" if condition else f"{item['id']}_{method}_{mode}"
        folder=Path(out)/suite; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(tag+'.npz'),trace=tr,reference=ref,estimate=est)
    return row

def experiments(out, workers, smoke=False, schedules=None):
    base=load('cases.json'); extra=load('additional_cases.json'); plans=load('schedules.json')
    if schedules: plans=json.loads(Path(schedules).read_text())
    if smoke: base=base[:1]; extra=extra[:1]
    jobs=[]
    for suite in ['friction','parameters','noisy','newstates','auxiliary','no_identification']:
        for item in extra if suite=='newstates' else base:
            methods=['conic'] if suite in ['auxiliary','no_identification'] else METHODS
            conditions=[None] if suite in ['friction','auxiliary'] else (['combined_pos'] if smoke else list(SCALES))
            modes=[0,1,2] if suite=='friction' and not smoke else [1]
            for condition in conditions:
                for mode in modes:
                    for method in methods:
                        jobs.append((suite,item,plans[item['id']+'_interval'],method,mode,condition,str(out)))
    rows={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(execute,jobs),1):
            suite=row['suite']; rows.setdefault(suite,[]).append(row)
            if i%24==0 or i==len(jobs):
                for name,data in rows.items():save(out/name/'metrics.json',data)
                print(f'Closed-loop runs: {i}/{len(jobs)}',flush=True)
    summarize(out)

def summarize(out):
    summary=[]; raw={}
    for suite in ['friction','parameters','noisy','newstates','auxiliary','no_identification']:
        path=out/suite/'metrics.json'
        if not path.exists():continue
        rows=json.loads(path.read_text()); raw[suite]=rows
        conditions=sorted({str(r.get('condition',r['friction_mode'])) for r in rows})
        for condition in ['pooled']+conditions:
            for method in METHODS:
                a=[r for r in rows if r['method']==method and (condition=='pooled' or str(r.get('condition',r['friction_mode']))==condition)]
                if not a:continue
                complete=[r for r in a if r['completed']]
                summary.append(dict(suite=suite,condition=condition,method=method,runs=len(a),
                    completed=len(complete),successful=sum(r['success'] for r in a),
                    physically_safe_completed=sum(r['completed'] and r['physical_safe'] for r in a),
                    mean_cost=float(np.mean([r['kinetic_integral'] for r in complete])) if complete else None,
                    mean_travel=float(np.mean([r['angular_travel_max'] for r in complete])) if complete else None,
                    max_voltage=max(r['max_voltage'] for r in a), min_voltage=min(r['min_voltage'] for r in a),
                    max_current=max(r['max_current'] for r in a), min_heat_margin=-max(r['max_heat_excess'] for r in a)))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    if len(raw)<4:return
    rng=np.random.default_rng(5110901); paired={}
    key=lambda r:(r['case_id'],r.get('condition',r['friction_mode']))
    for suite in ['friction','parameters','noisy','newstates']:
        rows=raw[suite]; ours={key(r):r for r in rows if r['method']=='conic'}; paired[suite]={}
        # Identical cluster order and draws across the main paired comparisons.
        keys=sorted(ours); clusters=sorted({k[0].split('_p')[0] for k in keys})
        idx=[np.array([j for j,k in enumerate(keys) if k[0].split('_p')[0]==cid]) for cid in clusters]
        draws=rng.integers(0,len(clusters),size=(20000,len(clusters)))
        for method in METHODS[1:]:
            other={key(r):r for r in rows if r['method']==method}
            pairs=[(ours[k],other[k]) for k in keys if ours[k]['completed'] and other[k]['completed']]
            a=np.array([r['kinetic_integral'] for r,q in pairs]);b=np.array([q['kinetic_integral'] for r,q in pairs])
            result=dict(pairs=len(pairs),mean_proposed_cost=float(a.mean()) if len(a) else None,mean_baseline_cost=float(b.mean()) if len(b) else None,
                        cost_reduction_percent=float(100*(1-a.mean()/b.mean())) if len(a) else None, wins=int(sum(a<b)),
                        baseline_success=sum(r['success'] for r in other.values()),
                        baseline_runs=len(other),
                        excluded_incomplete=[k for k in keys if not (ours[k]['completed'] and other[k]['completed'])])
            if len(pairs)==len(keys):
                ac=np.array([a[i].mean() for i in idx]);bc=np.array([b[i].mean() for i in idx]);boot=100*(1-ac[draws].mean(axis=1)/bc[draws].mean(axis=1))
                result['descriptive_cluster_interval95']=np.percentile(boot,[2.5,97.5]).tolist()
                result['independent_initial_states']=len(clusters)
            paired[suite][method]=result
    save(out/'paired_comparisons.json',paired)
    print('Wrote summary.csv and paired_comparisons.json',flush=True)

def script(name,out):
    env=os.environ.copy();env['RESULTS_DIR']=str(out)
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['all','smoke','experiments','checks','timing','planning','plots'],default='all')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--schedules',type=Path,help='Use regenerated schedules instead of the supplied numerical inputs')
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    out=(args.output or ROOT/('smoke_results' if args.stage=='smoke' else 'results')).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.schedules: os.environ['SCHEDULES_FILE']=str(args.schedules.resolve())
    if args.stage in ['all','smoke','experiments']:experiments(out,args.workers,args.stage=='smoke',args.schedules)
    if args.stage in ['all','checks']:
        for name in ['check_mathematics.py','check_compiled.py','check_certificates.py','check_sampled.py','check_refinement.py']:script(name,out)
    if args.stage in ['all','timing']:script('timing.py',out)
    if args.stage in ['all','planning']:script('planning.py',out)
    if args.stage in ['all','plots']:script('plot_results.py',out)
    print('Results:',out,flush=True)
if __name__=='__main__':main()
