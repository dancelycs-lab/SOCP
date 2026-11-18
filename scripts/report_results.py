"""Export numerical comparisons from completed experiment records."""
import csv
import json
import math
from pathlib import Path
from context import OUT

ALLOCATION = ['conic', 'dual', 'mcl', 'cvo', 'mcl_pdo']
SUITES = ['friction', 'parameters', 'noisy', 'newstates',
          'auxiliary', 'no_identification']


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, field):
    values = [r[field] for r in rows if r['completed'] and
              isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
    return sum(values) / len(values) if values else None


def key(row):
    return row['case_id'], row.get('condition', row['friction_mode'])


def main():
    paths = {name: OUT / name / 'metrics.json' for name in SUITES}
    missing = [str(p.relative_to(OUT)) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit('Missing experiment outputs: ' + ', '.join(missing) +
                         '. Run the experiments stage with the same --output directory.')
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    folder = OUT / 'tables'
    folder.mkdir(exist_ok=True)
    raw = [r for rows in data.values() for r in rows]
    # Keep individual outcomes visible, including rejected or unsuccessful runs.
    write_csv(OUT / 'metrics.csv', [
        {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        for row in raw])
    write_csv(folder / 'experiment_counts.csv', [
        dict(suite=name, runs=len(rows), completed=sum(r['completed'] for r in rows),
             successful=sum(r['success'] for r in rows))
        for name, rows in data.items()])

    summary = []
    for suite, rows in data.items():
        conditions = sorted({str(r.get('condition', r['friction_mode'])) for r in rows})
        for condition in ['pooled'] + conditions:
            for method in sorted({r['method'] for r in rows}):
                selected = [r for r in rows if r['method'] == method and
                            (condition == 'pooled' or
                             str(r.get('condition', r['friction_mode'])) == condition)]
                if not selected:
                    continue
                row = dict(suite=suite, condition=condition, method=method,
                           runs=len(selected), completed=sum(r['completed'] for r in selected),
                           successful=sum(r['success'] for r in selected),
                           physically_safe_completed=sum(r['completed'] and r['physical_safe']
                                                         for r in selected))
                for field in ['kinetic_integral', 'angular_travel_max', 'current_rmse',
                              'copper_energy_total']:
                    row['mean_' + field] = mean(selected, field)
                settled = [r for r in selected if r['completed'] and
                           isinstance(r.get('settling_time'), (int, float)) and
                           math.isfinite(r['settling_time'])]
                row['settled_completed'] = len(settled)
                row['mean_settling_time_settled_only'] = mean(settled, 'settling_time')
                row.update(max_voltage=max(r['max_voltage'] for r in selected),
                           min_voltage=min(r['min_voltage'] for r in selected),
                           max_current=max(r['max_current'] for r in selected),
                           min_copper_margin=-max(r['max_heat_excess'] for r in selected),
                           max_energy_balance_error=max(r['energy_balance_error'] for r in selected))
                summary.append(row)
    write_csv(folder / 'performance.csv', summary)
    write_csv(folder / 'allocation.csv', [r for r in summary if
              r['suite'] in SUITES[:4] and r['method'] in ALLOCATION])
    write_csv(folder / 'current_loops.csv', [r for r in summary if
              r['suite'] in SUITES[:4] and r['condition'] == 'pooled' and
              r['method'] in ['conic', 'pdo_smpcc', 'dpcc']])

    ablations = []
    for suite, reference in [('auxiliary', 'friction'), ('no_identification', 'parameters')]:
        base = {key(r): r for r in data[reference] if r['method'] == 'conic'}
        for condition in ['pooled'] + sorted({str(r.get('condition', r['friction_mode']))
                                           for r in data[suite]}):
            removed = [r for r in data[suite] if condition == 'pooled' or
                       str(r.get('condition', r['friction_mode'])) == condition]
            complete = [base[key(r)] for r in removed]
            pairs = [(base[key(r)], r) for r in removed
                     if base[key(r)]['completed'] and r['completed']]
            row = dict(ablation=suite, condition=condition, runs=len(removed),
                       complete_successful=sum(r['success'] for r in complete),
                       ablated_successful=sum(r['success'] for r in removed),
                       ablated_completed=sum(r['completed'] for r in removed),
                       ablated_physically_safe_completed=sum(r['completed'] and r['physical_safe']
                                                           for r in removed),
                       completed_pairs=len(pairs))
            for field in ['kinetic_integral', 'angular_travel_max']:
                row['complete_paired_mean_' + field] = mean([a for a, b in pairs], field)
                row['ablated_paired_mean_' + field] = mean([b for a, b in pairs], field)
            ablations.append(row)
    write_csv(folder / 'ablations.csv', ablations)

    filters = []
    for suite in SUITES[:4]:
        a = {key(r): r for r in data[suite] if r['method'] == 'conic'}
        b = {key(r): r for r in data[suite] if r['method'] == 'hocbf'}
        if a.keys() != b.keys():
            raise ValueError('Unmatched execution-filter cases in ' + suite)
        pairs = [(a[k], b[k]) for k in a if a[k]['completed'] and b[k]['completed']]
        filters.append(dict(suite=suite, runs=len(a),
                            proposed_successful=sum(r['success'] for r in a.values()),
                            hocbf_successful=sum(r['success'] for r in b.values()),
                            completed_pairs=len(pairs),
                            proposed_paired_mean_cost=mean([x for x, y in pairs], 'kinetic_integral'),
                            hocbf_paired_mean_cost=mean([y for x, y in pairs], 'kinetic_integral')))
    write_csv(folder / 'execution_filters.csv', filters)

    timing = OUT / 'timing' / 'summary.json'
    if timing.exists():
        report = json.loads(timing.read_text())
        write_csv(folder / 'supervisory_timing.csv', [
            dict(method=method, **record['total']) for method, record in report.items()
            if isinstance(record, dict) and 'total' in record])
        write_csv(folder / 'inner_loop_timing.csv', [
            dict(method=method, **record) for method, record in report.get('inner_loop', {}).items()])
    planning = OUT / 'planning_timing.json'
    if planning.exists():
        report = json.loads(planning.read_text())
        write_csv(folder / 'planning_timing.csv', [
            dict(component=name, **record) for name, record in report['summary'].items()])
    print(f'Exported {len(raw)} run records and numerical comparisons to {folder}.')


if __name__ == '__main__':
    main()
