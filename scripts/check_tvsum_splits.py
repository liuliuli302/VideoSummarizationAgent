import json

d = json.load(open('archive/8s-deepseek/exam_20260312_160107/split_overview.json'))
for ds, dd in d.get('datasets', {}).items():
    print(f'=== {ds} ===')
    for vn, vd in dd.get('variants', {}).items():
        per_split = [sr.get('mean_f1', sr.get('f1_mean')) for sr in vd['split_results']]
        splits_str = ', '.join(f'{x:.4f}' for x in per_split)
        mean = sum(per_split) / len(per_split)
        print(f'  {vn}: per_split=[{splits_str}] mean={mean:.4f}')
