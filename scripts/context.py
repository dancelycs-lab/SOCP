from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import controller as mod
OUT=Path(os.environ.get('RESULTS_DIR',ROOT/'results')).resolve()
OUT.mkdir(parents=True,exist_ok=True)
CASES={x['id']:x['case'] for name in ['cases.json','additional_cases.json'] for x in json.loads((ROOT/'config'/name).read_text())}
PLANS=json.loads(Path(os.environ.get('SCHEDULES_FILE',ROOT/'config/schedules.json')).read_text())
