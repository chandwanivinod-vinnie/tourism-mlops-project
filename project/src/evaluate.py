
from pathlib import Path
import json

metrics_path = Path(__file__).resolve().parents[1] / 'experiments' / 'metrics.json'
if not metrics_path.exists():
    raise FileNotFoundError(f'Metrics file missing: {metrics_path}')

with open(metrics_path, 'r', encoding='utf-8') as f:
    payload = json.load(f)

print('Best Params:', payload.get('best_params', {}))
print('Metrics:', payload.get('metrics', {}))

