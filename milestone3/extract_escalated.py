"""Dump the escalated rows (events + features + out-of-fold scores) of both datasets to ONE small JSON file for offline
inspection of the LLM prompts and of arbitration baselines, plus the label vector of each dataset (labels_<ds>.npy) and a few per-row facts (rowinfo_<ds>.npz: event category,
LOLBin flag, host event rate, host), all aligned with oof_*.npy / fold_id_*.npy. Writes only these new files into results/.
Usage (from the milestone3 folder, venv active):  python extract_escalated.py   -> results/escalated_rows.json"""
import json, numpy as np, pandas as pd, config as cfg

out = {}
for ds in cfg.DATASETS:
    casc = json.load(open(cfg.RESULTS / f"cascade_{ds}.json"))
    pos = np.array(casc['escalated_positions'], dtype=int)
    feats = pd.read_parquet(cfg.WORK / f"{ds}_features.parquet"); ev = pd.read_parquet(cfg.WORK / f"{ds}_events.parquet").set_index('row_id')
    assert ev.index.is_unique, "row_id not unique in the events table"
    p1 = np.load(cfg.RESULTS / f"oof_{ds}_lgbm.npy"); p2 = np.load(cfg.RESULTS / f"oof_{ds}_rf.npy"); an = np.load(cfg.RESULTS / f"oof_{ds}_if.npy")
    fold = np.load(cfg.RESULTS / f"fold_id_{ds}.npy")
    r1 = json.load(open(cfg.RESULTS / f"cv_{ds}_lgbm.json")); r2 = json.load(open(cfg.RESULTS / f"cv_{ds}_rf.json"))
    rids = feats['row_id'].values[pos]
    e = ev.loc[rids].reset_index(); f = feats.iloc[pos].copy()          # events are looked up by row_id, exactly as the LLM stage does
    for c in e.columns:
        if str(e[c].dtype).startswith('datetime'): e[c] = e[c].astype(str)
    out[ds] = {'positions': pos.tolist(), 'row_ids': feats['row_id'].values[pos].tolist(),
               'topup_positions': casc.get('near_threshold_topup_positions', []), 'sensitivity_positions': casc['sensitivity_positions'],
               'p_lgbm': p1[pos].tolist(), 'p_rf': p2[pos].tolist(), 'anom_pct': an[pos].tolist(), 'fold': fold[pos].tolist(),
               'thr_lgbm': r1['fold_thresholds'], 'thr_rf': r2['fold_thresholds'], 'bands_lgbm': r1['fold_bands'],
               'events': json.loads(e.to_json(orient='records', date_format='iso', default_handler=str)),
               'features': json.loads(f.to_json(orient='records', date_format='iso', default_handler=str))}
    np.save(cfg.RESULTS / f"labels_{ds}.npy", feats['label'].values.astype(np.int8))   # row-aligned with oof_*.npy / fold_id_*.npy
    hosts = feats['host'].astype(str); host_names = sorted(hosts.unique()); host_code = hosts.map({h: i for i, h in enumerate(host_names)}).values.astype(np.int16)
    np.savez_compressed(cfg.RESULTS / f"rowinfo_{ds}.npz", event_category=feats['f_event_category'].values.astype(np.int16), image_is_lolbin=feats['f_image_is_lolbin'].values.astype(np.int8),
                        host_evt_rate_60s=feats['f_host_evt_rate_60s'].values.astype(np.float32), host_code=host_code, host_names=np.array(host_names))   # row-aligned per-row facts for the diagnostics
    print(ds, len(pos), 'rows; labels ->', cfg.RESULTS / f"labels_{ds}.npy")
json.dump(out, open(cfg.RESULTS / 'escalated_rows.json', 'w'))
print('wrote', cfg.RESULTS / 'escalated_rows.json')
