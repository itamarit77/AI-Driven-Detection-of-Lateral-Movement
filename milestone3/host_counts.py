"""Per-host class counts (attack / benign rows) for the forensic discussion of §2.1 -> results/host_class_counts.json.
Reads the intermediate event tables written by `run_m3.py --stage ingest` (work/<ds>_events.parquet)."""
import json, sys
import pandas as pd
import config as cfg
out = {}
for ds in cfg.DATASETS:
    ev = pd.read_parquet(cfg.WORK / f"{ds}_events.parquet", columns=['host', 'label'])
    g = ev.groupby('host')['label'].agg(['sum', 'count'])
    out[ds] = {h: {'attack': int(r['sum']), 'benign': int(r['count'] - r['sum'])} for h, r in g.iterrows()}
cfg.RESULTS.mkdir(parents=True, exist_ok=True); json.dump(out, open(cfg.RESULTS / 'host_class_counts.json', 'w'), indent=1); print(out)
