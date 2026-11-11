"""Measure precomputation separately from execution; leave experiment caches intact."""
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np

from context import ROOT,OUT
import plant
cases = json.loads((ROOT/'config/cases.json').read_text())
bounds = (.00623, .01157)
# Load/JIT warm-up is separate from repeated algorithm timings.
start = time.perf_counter()
assert plant.plan(plant.Case(**cases[0]['case']), bounds) is not None
warmup_ms = 1000 * (time.perf_counter() - start)
solve, audit = plant.pub.solve, plant.pub.audit
stages = {}

def timed_solve(*args, **kwargs):
    stage = 'preparation_ms' if kwargs.get('prepare_only') else 'solve_ms'
    start = time.perf_counter()
    result = solve(*args, **kwargs)
    stages[stage] = stages.get(stage, 0.) + 1000 * (time.perf_counter() - start)
    return result

def timed_audit(*args, **kwargs):
    start = time.perf_counter()
    result = audit(*args, **kwargs)
    stages['audit_ms'] = 1000 * (time.perf_counter() - start)
    return result

plant.pub.solve, plant.pub.audit = timed_solve, timed_audit
rows = []
regenerated = {}
try:
    for repeat in range(3):
        for item in cases[repeat:] + cases[:repeat]:
            stages = {}
            start = time.perf_counter()
            plan = plant.plan(plant.Case(**item['case']), bounds)
            total = 1000 * (time.perf_counter() - start)
            assert plan is not None, item['id']
            regenerated[item['id']+'_interval'] = {k:v for k,v in plan.items() if k!='compute_ms'}
            # The remainder includes the forecast, object creation, and serialization.
            stages['forecast_and_other_ms'] = total - sum(stages.values())
            rows.append(dict(case_id=item['id'], repeat=repeat, **stages,
                             total_ms=total, valid=True))
        print('Finished planning repetition', repeat + 1, flush=True)
finally:
    plant.pub.solve, plant.pub.audit = solve, audit

summary = {key: dict(median=float(np.median([r[key] for r in rows])),
                     p99=float(np.percentile([r[key] for r in rows], 99)),
                     maximum=float(max(r[key] for r in rows)))
           for key in ['preparation_ms', 'forecast_and_other_ms', 'solve_ms', 'audit_ms', 'total_ms']}
cpu = next((line.split(':', 1)[1].strip() for line in (Path('/proc/cpuinfo').read_text().splitlines() if Path('/proc/cpuinfo').exists() else [])
            if line.startswith('model name')), platform.processor())
report = dict(cpu=cpu, python=platform.python_version(), warmup_ms=warmup_ms,
              cases=len(cases), repetitions=3, runs=len(rows),
              protocol='Serial execution after one untimed warm-up; full per-case model construction and canonicalization included; process import/JIT warm-up excluded; no saved plan reused.',
              summary=summary, rows=rows)
(OUT / 'planning_timing.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['cpu', 'runs', 'summary']}, indent=2))

for item in json.loads((ROOT/'config/additional_cases.json').read_text()):
    plan=plant.plan(plant.Case(**item['case']),bounds)
    assert plan is not None, item['id']
    regenerated[item['id']+'_interval']={k:v for k,v in plan.items() if k!='compute_ms'}
(OUT/'schedules.json').write_text(json.dumps(regenerated)+'\n')
