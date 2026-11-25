"""Run the experiment in a new output directory; use --quick for a smoke test."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()
config = json.loads((root / 'config.json').read_text(encoding='utf-8-sig'))
if args.quick:
    config.update(duration_seconds=5, repeats=1, steps=40)
out = root / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
out.mkdir(parents=True, exist_ok=False)
for name in ('LIVE_JOURNALS', 'DIAGNOSTICS'):
    (out / name).mkdir()
(out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
print('Results:', out, flush=True)
raise SystemExit(subprocess.call([sys.executable, str(root / 'crystal_engine.py'), str(out)]))
