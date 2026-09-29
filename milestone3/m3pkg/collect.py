"""Collect every stage output into results/m3_results.json and draw the report figures."""
import json, platform, sys, subprocess, logging
from pathlib import Path
import numpy as np

log = logging.getLogger("m3.collect")

LITERATURE = [   # verified in Milestone 2 (Table 14 there); used for Part 2.4
    {'study': 'Bai et al. 2021 [1] — LogitBoost (LANL RDP sessions)', 'metric': 'F1 0.998 (0.997 without identity fields); AP 0.95 on unseen sources', 'unit': 'RDP session'},
    {'study': 'Smiliotopoulos et al. 2023 [2] — ExtraTrees best (LMD-2023)', 'metric': 'AUC 99.84% / F1 99.41%; RF 99.68%, LightGBM 99.61% AUC; per-event LSTM AUC 95.82% / F1 95.55%', 'unit': 'Sysmon event'},
    {'study': 'Smiliotopoulos & Kambourakis 2026 [3] — resampling study (LMD-2023)', 'metric': 'resampling adds ≈0.05 percentage points (shallow), ≈3.5 (DNN)', 'unit': 'Sysmon event'},
    {'study': 'Smiliotopoulos et al. 2025 [4] — unsupervised (LMD-2023)', 'metric': 'best shallow ≈94.7% AUC / 93% F1; best deep ≈95.2% / 93.8%', 'unit': 'Sysmon event'},
    {'study': 'Kushwaha et al. 2022 [5] — XGBoost / FCDN (LANL)', 'metric': 'recall 86.51% / FPR 0.0013% (XGBoost); 92.13% / 0.0029% (FCDN)', 'unit': 'user-behaviour window'},
    {'study': 'Le & Zincir-Heywood 2021 [11] — Isolation Forest (CERT r4.2)', 'metric': 'instance-level AUC 0.825 / user-level 0.847 (evaluation includes training period)', 'unit': 'user-week'},
]


def env_info():
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': __import__('os').cpu_count()}
    for m in ('numpy', 'pandas', 'sklearn', 'lightgbm', 'torch', 'requests', 'pyarrow', 'matplotlib'):
        try: info[m] = __import__(m).__version__
        except Exception: info[m] = 'not installed'
    try:
        import torch; info['cuda'] = torch.version.cuda if torch.cuda.is_available() else None
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception: info['cuda'] = None; info['gpu'] = None
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        info['nvidia_smi'] = out.stdout.strip() or None
    except Exception: info['nvidia_smi'] = None
    return info


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


def collect(cfg, MODELS):
    R = cfg.RESULTS
    def _cfgval(v):   # paths are stored by name only (no machine-specific directories in the results file)
        if isinstance(v, Path): return v.name
        if isinstance(v, list): return [x.name if isinstance(x, Path) else x for x in v]
        return v if isinstance(v, (int, float, str, bool, list, dict)) or v is None else str(v)
    out = {'environment': env_info(), 'config': {k: _cfgval(v) for k, v in vars(cfg).items() if k.isupper()}}
    prev = _load(R / 'm3_results.json') or {}   # an earlier results file: its diagnostics are kept when their inputs (OOF arrays, escalated rows) are absent
    out['facts'] = {ds: _load(R / f"facts_{ds}.json") for ds in cfg.DATASETS}
    out['cv'] = {ds: {m: _load(R / f"cv_{ds}_{m}.json") for m in MODELS} for ds in cfg.DATASETS}
    out['cv_reduced'] = {m: _load(R / f"cv_mordor_{m}_reduced.json") for m in ('rf', 'lgbm')}
    out['sensitivity'] = {ds: {m: _load(R / f"sens_{ds}_{m}.json") for m in MODELS} for ds in cfg.DATASETS}
    out['cross'] = {}
    for p in sorted(R.glob('cross_*_to_*_*.json')):
        out['cross'][p.stem] = _load(p)
    out['cascade'] = {ds: _load(R / f"cascade_{ds}.json") for ds in cfg.DATASETS}
    out['llm'] = {}
    for ds in cfg.DATASETS:
        j = _load(R / f"llm_{ds}.json")
        if j:
            j = dict(j); rec = j['records']
            pt = [r['prompt_tokens'] for r in rec if r.get('prompt_tokens')]; ot = [r['output_tokens'] for r in rec if r.get('output_tokens')]
            j['mean_prompt_tokens_all'] = float(np.mean(pt)) if pt else None; j['mean_output_tokens_all'] = float(np.mean(ot)) if ot else None; j['n_records_all'] = len(rec)
            j['records'] = rec[:50]          # keep the JSON small; the full cache is llm_cache.jsonl
        out['llm'][ds] = j
    out['refine'] = {ds: _load(R / f"refine_{ds}.json") for ds in cfg.DATASETS}    # §2.2 diagnostic-driven refinement (None until --stage refine has run)
    out['literature'] = LITERATURE
    # optimisation log (v0 default threshold -> v1 tuned threshold -> v2 cascade stage 2 -> v3 LLM stage 3)
    opt = {}
    for ds in cfg.DATASETS:
        lg = out['cv'][ds].get('lgbm'); ca = out['cascade'].get(ds); ll = out['llm'].get(ds)
        if lg:
            steps = [{'step': 'v0 LightGBM, default threshold 0.5', **{k: lg['oof_metrics_thr05'][k] for k in ('f1', 'fpr', 'fnr')}},
                     {'step': 'v1 LightGBM, inner-validation F1 threshold', **{k: lg['oof_metrics'][k] for k in ('f1', 'fpr', 'fnr')}}]
            if ca: steps.append({'step': 'v2 cascade stage 2 (RF on the ambiguous band)', **{k: ca['stage2'][k] for k in ('f1', 'fpr', 'fnr')}})
            if ll and 'stage3' in ll: steps.append({'step': 'v3 cascade stage 3 (LLM on disputed rows)', **{k: ll['stage3'][k] for k in ('f1', 'fpr', 'fnr')}})
            opt[ds] = steps
    out['optimisation_log'] = opt
    try:
        out['diagnostics'] = diagnostics(cfg, MODELS, out)
    except Exception as e:
        log.exception("diagnostics failed: %s", e); out['diagnostics'] = {}
    try:
        out['baselines'] = arbitration_baselines(cfg, out)
    except Exception as e:
        log.exception("baselines failed: %s", e); out['baselines'] = {}
    for ds in cfg.DATASETS:   # do not degrade an existing results file when a re-run lacks the inputs of a diagnostic
        d = (out['diagnostics'] or {}).get(ds) or {}; pd_ = (prev.get('diagnostics') or {}).get(ds)
        if pd_ and not d.get('error_overlap') and pd_.get('error_overlap'):
            log.warning("diagnostics %s: OOF arrays missing — keeping the diagnostics of the existing m3_results.json (run --stage train to recompute)", ds); out['diagnostics'][ds] = pd_
        if not (out['baselines'] or {}).get(ds) and (prev.get('baselines') or {}).get(ds):
            log.warning("baselines %s: results/escalated_rows.json missing — keeping the baselines of the existing m3_results.json (run extract_escalated.py to recompute)", ds); out['baselines'][ds] = prev['baselines'][ds]
    json.dump(out, open(R / 'm3_results.json', 'w'), indent=1, default=str)
    try:
        figures(cfg, MODELS, out)
    except Exception as e:
        log.exception("figure generation failed: %s", e)
    log.info("wrote %s", R / 'm3_results.json')
    return out


def diagnostics(cfg, MODELS, out):
    """Row-level checks that the report quotes and that no single-model file contains: (1) how far the supervised models'
    out-of-fold errors coincide (row-level intersections, per-fold thresholds); (2) how the cascade band relates to the
    Stage-1 F1 threshold in every fold (rows that lie between the two devices). Needs work/<ds>_features.parquet (labels)."""
    import pandas as pd
    R = cfg.RESULTS; diag = {}
    for ds in cfg.DATASETS:
        fp = cfg.WORK / f"{ds}_features.parquet"; lp = R / f"labels_{ds}.npy"
        if fp.exists(): y = pd.read_parquet(fp, columns=['host', 'ts', 'label']).sort_values(['host', 'ts'])['label'].values.astype(int)
        elif lp.exists(): y = np.load(lp).astype(int)                      # written by extract_escalated.py (row-aligned)
        else: log.warning("diagnostics: neither %s nor %s present, skipped", fp, lp); continue
        fold = np.load(R / f"fold_id_{ds}.npy"); d = {'n': int(len(y)), 'pos': int(y.sum())}
        pred, cov = {}, {}
        for m in MODELS:
            cvj = out['cv'][ds].get(m); sp = R / f"oof_{ds}_{m}.npy"
            if not cvj or not sp.exists(): continue
            s = np.load(sp); ft = cvj['fold_thresholds']; thr = np.array([ft[f] if 0 <= f < len(ft) else cvj['threshold'] for f in fold])
            ok = ~np.isnan(s); pred[m] = np.where(ok, (s >= thr - 1e-7).astype(int), -1); cov[m] = ok
        err = {m: (pred[m] != y) & cov[m] for m in pred}
        ov = {}
        for a, b in (('rf', 'lgbm'), ('rf', 'lstm'), ('lgbm', 'lstm')):
            if a in err and b in err:
                both = cov[a] & cov[b]; ea, eb = err[a] & both, err[b] & both
                ov[f"{a}_{b}"] = {'rows_compared': int(both.sum()), 'errors_' + a: int(ea.sum()), 'errors_' + b: int(eb.sum()), 'errors_both': int((ea & eb).sum()),
                                  'jaccard': float((ea & eb).sum() / max((ea | eb).sum(), 1)),
                                  'share_of_' + a + '_errors_shared': float((ea & eb).sum() / max(ea.sum(), 1)), 'share_of_' + b + '_errors_shared': float((ea & eb).sum() / max(eb.sum(), 1))}
        if all(m in err for m in ('rf', 'lgbm', 'lstm')):
            both = cov['rf'] & cov['lgbm'] & cov['lstm']; e3 = [err[m] & both for m in ('rf', 'lgbm', 'lstm')]
            ov['rf_lgbm_lstm'] = {'rows_compared': int(both.sum()), 'errors_any': int((e3[0] | e3[1] | e3[2]).sum()), 'errors_all_three': int((e3[0] & e3[1] & e3[2]).sum()),
                                  'errors_by_model': {m: int(e.sum()) for m, e in zip(('rf', 'lgbm', 'lstm'), e3)}}
        d['error_overlap'] = ov
        # matched comparison: every model on exactly the rows the LSTM scores (window-end events), per-fold thresholds
        if 'lstm' in cov:
            from sklearn.metrics import roc_auc_score, average_precision_score
            mm = {}
            for m in pred:
                sel = cov['lstm'] & cov[m]; yy = y[sel]; pp = pred[m][sel]; ss = np.load(R / f"oof_{ds}_{m}.npy")[sel]
                tp = int(((pp == 1) & (yy == 1)).sum()); fp_ = int(((pp == 1) & (yy == 0)).sum()); fn = int(((pp == 0) & (yy == 1)).sum()); tn = int(((pp == 0) & (yy == 0)).sum())
                pr = tp / (tp + fp_) if tp + fp_ else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
                mm[m] = {'n': int(sel.sum()), 'pos': int(yy.sum()), 'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0, 'fpr': fp_ / (fp_ + tn) if fp_ + tn else 0.0, 'fnr': fn / (fn + tp) if fn + tp else 0.0,
                         'roc_auc': float(roc_auc_score(yy, ss)) if len(np.unique(yy)) == 2 else None, 'pr_auc': float(average_precision_score(yy, ss)) if len(np.unique(yy)) == 2 else None}
            d['matched_on_lstm_rows'] = mm
        # composition of the rows on which the two tree models disagree, and hosts per fold (from the feature parquet when present, else from rowinfo_<ds>.npz written by extract_escalated.py)
        try:
            rp = R / f"rowinfo_{ds}.npz"
            if fp.exists():
                f = pd.read_parquet(fp, columns=['host', 'ts', 'f_event_category', 'f_image_is_lolbin', 'f_host_evt_rate_60s']).sort_values(['host', 'ts'])
                cat = f['f_event_category'].values; lol = f['f_image_is_lolbin'].values; rate = f['f_host_evt_rate_60s'].values; host = f['host'].astype(str).values
            elif rp.exists():
                z = np.load(rp, allow_pickle=False); cat = z['event_category']; lol = z['image_is_lolbin']; rate = z['host_evt_rate_60s']; host = z['host_names'][z['host_code']]
            else:
                cat = None
            if cat is not None and 'rf' in pred and 'lgbm' in pred:
                from .common_features import CAT_ORDINAL
                CAT_NAME = {v: k for k, v in CAT_ORDINAL.items()}; CAT_NAME[-1] = 'other'
                dis = (pred['rf'] != pred['lgbm']) & cov['rf'] & cov['lgbm']
                def comp(mask):
                    c = pd.Series([CAT_NAME.get(int(x), 'other') for x in cat[mask]]).value_counts(normalize=True).head(6)
                    return {k: round(float(v), 3) for k, v in c.items()}
                d['disagreement'] = {'n': int(dis.sum()), 'attack_share': float(y[dis].mean()) if dis.sum() else None, 'categories': comp(dis), 'categories_all_rows': comp(np.ones(len(y), bool)),
                                     'lolbin_share': float(lol[dis].mean()) if dis.sum() else None, 'lolbin_share_all_rows': float(lol.mean()),
                                     'host_rate_median': float(np.median(rate[dis])) if dis.sum() else None, 'host_rate_median_all_rows': float(np.median(rate)),
                                     'hosts': {str(k): round(float(v), 3) for k, v in pd.Series(host[dis]).value_counts(normalize=True).head(4).items()},
                                     'lgbm_attack_rf_benign': int((dis & (pred['lgbm'] == 1)).sum()), 'rf_attack_lgbm_benign': int((dis & (pred['rf'] == 1)).sum())}
                d['hosts_per_fold'] = {int(k): int(len(set(host[fold == k]))) for k in np.unique(fold[fold >= 0])}; d['n_hosts'] = int(len(set(host)))
        except Exception as e:
            log.warning("disagreement composition skipped: %s", e)
        # cascade routing broken down by label (the same stage_decisions the cascade stage uses)
        try:
            from .cascade import stage_decisions
            lg = out['cv'][ds].get('lgbm'); rf = out['cv'][ds].get('rf')
            if lg and rf and (R / f"oof_{ds}_if.npy").exists():
                p1 = np.load(R / f"oof_{ds}_lgbm.npy"); p2 = np.load(R / f"oof_{ds}_rf.npy"); an = np.load(R / f"oof_{ds}_if.npy")
                dec = stage_decisions(y, p1, p2, an, lg['fold_bands'], lg['fold_thresholds'], rf['fold_thresholds'], fold)
                amb = dec['ambiguous']; dsp = dec['disputed']; res_att = amb & ~dsp & (dec['stage2'] == 1); res_ben = amb & ~dsp & (dec['stage2'] == 0)
                s_or = dec['stage2'].copy(); s_or[dsp] = y[dsp]; v = dec['valid']
                fp_res = (s_or == 1) & (y == 0) & v; fn_res = (s_or == 0) & (y == 1) & v
                out_att = v & ~amb & (dec['stage1'] == 1); out_ben = v & ~amb & (dec['stage1'] == 0)
                # per-fold view of the cascade and the escalation tie-break
                pf = []
                for k in sorted(set(int(f) for f in fold if f >= 0)):
                    mk = (fold == k) & v; f1 = lambda pr: (lambda tp, fp_, fn: 2 * tp / (2 * tp + fp_ + fn) if tp else 0.0)(int(((pr == 1) & (y == 1) & mk).sum()), int(((pr == 1) & (y == 0) & mk).sum()), int(((pr == 0) & (y == 1) & mk).sum()))
                    pf.append({'fold': k, 'n_ambiguous': int((amb & mk).sum()), 'n_disputed': int((dsp & mk).sum()), 'f1_stage1': f1(dec['stage1']), 'f1_stage2': f1(dec['stage2']), 'collapsed_band': bool(lg['fold_bands'][k][1] - lg['fold_bands'][k][0] < 1e-5)})
                tie = None; ca_json = out['cascade'].get(ds)
                if ca_json and ca_json.get('escalated_positions'):
                    kmax = cfg.LLM_MAX_ESCALATIONS; dist = np.abs(p1 - dec['thr1']); dd = np.sort(dist[dsp])
                    if len(dd) > kmax:
                        cut = dd[kmax - 1]; tied = dsp & (np.abs(dist - cut) <= 1e-12); esc_pos = np.array(ca_json['escalated_positions'], dtype=int)
                        tie = {'cutoff_distance': float(cut), 'n_strictly_below_cutoff': int((dist[dsp] < cut - 1e-12).sum()), 'n_tied_at_cutoff': int(tied.sum()), 'n_escalated': int(len(esc_pos)),
                               'tied_folds': sorted(int(f) for f in np.unique(fold[tied])), 'tied_attack_share': float(y[tied].mean()) if tied.sum() else None,
                               'disputed_attack_share': float(y[dsp].mean()), 'escalated_attack_share': float(y[esc_pos].mean())}
                    else: tie = {'cutoff_distance': None, 'n_strictly_below_cutoff': int(dsp.sum()), 'n_tied_at_cutoff': 0, 'n_escalated': int(len(ca_json['escalated_positions']))}
                d['cascade_breakdown'] = {'per_fold': pf, 'escalation_tie': tie, 'band_resolved_attack': {'n': int(res_att.sum()), 'attack': int(y[res_att].sum())}, 'band_resolved_benign': {'n': int(res_ben.sum()), 'attack': int(y[res_ben].sum())},
                                          'outside_band_attack_verdicts': {'n': int(out_att.sum()), 'attack': int(y[out_att].sum())}, 'outside_band_benign_verdicts': {'n': int(out_ben.sum()), 'attack': int(y[out_ben].sum())},
                                          'stage2_attack_verdicts': {'n': int((v & (dec['stage2'] == 1)).sum()), 'attack': int(y[v & (dec['stage2'] == 1)].sum())}, 'stage2_benign_verdicts': {'n': int((v & (dec['stage2'] == 0)).sum()), 'attack': int(y[v & (dec['stage2'] == 0)].sum())},
                                          'disputed': {'n': int(dsp.sum()), 'attack': int(y[dsp].sum()), 'lean_attack': int(dec['lean1'][dsp].sum())},
                                          'oracle_residual_fp_in_band': int((fp_res & amb).sum()), 'oracle_residual_fp_outside_band': int((fp_res & ~amb).sum()),
                                          'oracle_residual_fn_in_band': int((fn_res & amb).sum()), 'oracle_residual_fn_outside_band': int((fn_res & ~amb).sum())}
        except Exception as e:
            log.warning("cascade breakdown skipped: %s", e)
        # cascade band vs Stage-1 threshold, per fold (LightGBM out-of-fold scores)
        lg = out['cv'][ds].get('lgbm'); sp = R / f"oof_{ds}_lgbm.npy"
        if lg and sp.exists():
            s = np.load(sp); rows = []
            for k, (lo, hi) in enumerate(lg['fold_bands']):
                thr = lg['fold_thresholds'][k]; sk = s[fold == k]; raw = (lg.get('fold_bands_raw') or [None] * 99)[k]
                rows.append({'fold': k, 'thr': thr, 'tau_lo': lo, 'tau_hi': hi, 'tau_lo_raw': raw[0] if raw else None, 'tau_hi_raw': raw[1] if raw else None,
                             'n_rows': int(len(sk)), 'n_ambiguous': int(((sk > lo + 1e-7) & (sk < hi - 1e-7)).sum()),
                             'n_attack_verdict_below_tau_lo': int(((sk >= thr - 1e-7) & (sk <= lo + 1e-7)).sum()),
                             'n_benign_verdict_above_tau_hi': int(((sk < thr - 1e-7) & (sk >= hi - 1e-7)).sum())})
            d['band_vs_threshold'] = rows
        diag[ds] = d
    return diag


def _cm(y, v):
    y = np.asarray(y); v = np.asarray(v); tp = int(((v == 1) & (y == 1)).sum()); fp = int(((v == 1) & (y == 0)).sum()); fn = int(((v == 0) & (y == 1)).sum()); tn = int(((v == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {'n': int(len(y)), 'pos': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0,
            'fpr': fp / (fp + tn) if fp + tn else 0.0, 'fnr': fn / (fn + tp) if fn + tp else 0.0, 'accuracy': (tp + tn) / len(y) if len(y) else 0.0}


def arbitration_baselines(cfg, out):
    """Label-free arbitration rules on the escalated rows (results/escalated_rows.json, written by extract_escalated.py),
    evaluated on exactly the rows the LLM answered when llm_<ds>.json exists (else on all escalated rows), so that the
    LLM of Part 4 has a controlled comparison. Every rule uses only what the LLM also saw: the two model scores with
    their thresholds and the anomaly percentile. Each rule is evaluated twice: on the full-precision scores ('rules')
    and on the numbers exactly as the prompt displayed them ('rules_displayed': scores and thresholds to two decimals,
    anomaly percentile to an integer percent, the formatting of llm.models_text), which is what the LLM had to work
    with. At displayed precision a score can coincide with its threshold; such ties count as positive (the >= convention
    of the cascade code) and their number is reported. Also returns the budget-matched oracle: Stage 2 with a perfect
    verdict on the escalated disputed rows only (the room Stage 3 actually had, as opposed to the every-disputed-row bound)."""
    R = cfg.RESULTS; fp = R / 'escalated_rows.json'
    if not fp.exists(): log.warning("baselines: %s missing, skipped", fp); return {}
    ex = json.load(open(fp)); res = {}
    disp2 = lambda a: np.array([float(f"{x:.2f}") for x in a]); disp0 = lambda a: np.array([int(f"{100 * x:.0f}") for x in a])   # exactly llm.models_text's formatting
    def rule_set(p1, p2, an, t1, t2, eps):
        m1 = (p1 >= t1 - eps).astype(int); m2 = (p2 >= t2 - eps).astype(int); d1 = p1 - t1; d2 = p2 - t2
        return m1, m2, {'lgbm': m1, 'rf': m2, 'margin': np.where(np.abs(d1) >= np.abs(d2), m1, m2), 'anomaly_99': (an >= 0.99 - eps).astype(int),
                        'majority_with_anomaly_95': ((m1 + m2 + (an >= 0.95 - eps).astype(int)) >= 2).astype(int), 'always_attack': np.ones_like(m1), 'always_benign': np.zeros_like(m1)}
    for ds, e in ex.items():
        fold = np.array(e['fold']); p1 = np.array(e['p_lgbm']); p2 = np.array(e['p_rf']); an = np.array(e['anom_pct'])
        t1 = np.array([e['thr_lgbm'][f] for f in fold]); t2 = np.array([e['thr_rf'][f] for f in fold]); y = np.array([ev['label'] for ev in e['events']])
        m1, m2, rules = rule_set(p1, p2, an, t1, t2, 1e-7)
        p1d, p2d, t1d, t2d, and_ = disp2(p1), disp2(p2), disp2(t1), disp2(t2), disp0(an) / 100.0
        m1d, m2d, rules_d = rule_set(p1d, p2d, and_, t1d, t2d, 1e-9)
        mask = np.ones(len(y), dtype=bool); ll = out.get('llm', {}).get(ds)
        if ll and ll.get('records'):
            ok = {r['row_id'] for r in _load(R / f"llm_{ds}.json")['records'] if r['verdict'] is not None}
            mask = np.array([rid in ok for rid in e['row_ids']])
        top = set(int(p) for p in e.get('topup_positions', [])); tmask = np.array([int(p) in top for p in e['positions']])
        sub = set(int(p) for p in e.get('sensitivity_positions', [])); smask = np.array([int(p) in sub for p in e['positions']]) & mask
        gb_cross = int(((m1 == 1) & (p2 < t1 - 1e-7) & smask).sum()); rf_cross = int((((p1 >= t2 - 1e-7).astype(int) != m2) & smask).sum())
        gb_cross_d = int(((m1d == 1) & (p2d < t1d - 1e-9) & smask).sum()); rf_cross_d = int((((p1d >= t2d - 1e-9).astype(int) != m2d) & smask).sum())
        d3 = lambda a: np.array([float(f"{x:.3f}") for x in a])
        ties = {'gb_score_equals_threshold': int((np.isclose(p1d, t1d) & mask).sum()), 'rf_score_equals_threshold': int((np.isclose(p2d, t2d) & mask).sum()),
                'gb_score_equals_threshold_3dp': int((np.isclose(d3(p1), d3(t1)) & mask).sum()), 'rf_score_equals_threshold_3dp': int((np.isclose(d3(p2), d3(t2)) & mask).sum()),   # would three decimals have resolved the ties?
                'gb_ties_among_topups': int((np.isclose(p1d, t1d) & mask & tmask).sum()), 'n_topups_evaluated': int((mask & tmask).sum()),
                'equal_margins': int((np.isclose(np.abs(p1d - t1d), np.abs(p2d - t2d)) & mask).sum())}
        # budget-matched oracle: Stage 2 with the truth on the escalated disputed rows (all of them, parsed or not); the Stage-2 verdict on a band row is the RF verdict
        ca = out.get('cascade', {}).get(ds); budget = None
        if ca and ca.get('stage2') and (~tmask).sum():
            tp, fpc, fn, tn = (int(ca['stage2'][k]) for k in ('tp', 'fp', 'fn', 'tn'))
            for v, yy in zip(m2[~tmask], y[~tmask]):
                if v == yy: continue
                if yy == 1: fn -= 1; tp += 1
                else: fpc -= 1; tn += 1
            pr = tp / (tp + fpc) if tp + fpc else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
            budget = {'n_escalated_disputed': int((~tmask).sum()), 'tp': tp, 'fp': fpc, 'fn': fn, 'tn': tn, 'precision': pr, 'recall': rc, 'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                      'fpr': fpc / (fpc + tn) if fpc + tn else 0.0, 'fnr': fn / (fn + tp) if fn + tp else 0.0, 'n_corrected': int((m2[~tmask] != y[~tmask]).sum())}
        res[ds] = {'n_escalated': int(len(y)), 'n_evaluated': int(mask.sum()), 'restricted_to_parsed_llm_rows': bool(ll and ll.get('records')),
                   'rules': {k: _cm(y[mask], v[mask]) for k, v in rules.items()}, 'rules_displayed': {k: _cm(y[mask], v[mask]) for k, v in rules_d.items()},
                   'rules_on_sensitivity_subset': {k: _cm(y[smask], v[smask]) for k, v in rules.items()}, 'rules_on_sensitivity_subset_displayed': {k: _cm(y[smask], v[smask]) for k, v in rules_d.items()},
                   'n_sensitivity_subset': int(smask.sum()), 'displayed_ties': ties, 'margin_follows_rf_share': float(((rules_d['margin'] == m2d) & mask).sum() / max(1, mask.sum())),
                   'swap_effect_on_subset': {'gb_slot_falls_below_its_threshold': gb_cross, 'rf_slot_changes_side': rf_cross, 'gb_slot_falls_below_its_threshold_displayed': gb_cross_d, 'rf_slot_changes_side_displayed': rf_cross_d, 'n': int(smask.sum())},
                   'budget_matched_oracle': budget,
                   'display_format': 'scores and thresholds to two decimals, anomaly percentile to an integer percent (llm.models_text); displayed ties count as positive',
                   'definitions': {'lgbm': 'LightGBM verdict at its threshold', 'rf': 'Random-Forest verdict at its threshold', 'margin': 'verdict of the model further from its own threshold',
                                   'anomaly_99': 'attack iff anomaly percentile >= 0.99', 'majority_with_anomaly_95': 'majority of {LightGBM, RF, anomaly >= 0.95}', 'always_attack': 'constant attack', 'always_benign': 'constant benign'}}
    return res


def figures(cfg, MODELS, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    F = cfg.RESULTS / 'figures'; F.mkdir(exist_ok=True)
    # 0. architecture diagrams (Step 7 / Step 8 deliverables): the modular pipeline and the cascade + LLM-arbitration block diagram
    from .diagrams import draw_pipeline, draw_cascade
    draw_pipeline(F / 'arch_pipeline.png'); draw_cascade(F / 'arch_cascade.png', {ds: c for ds, c in out['cascade'].items() if c}, {ds: l for ds, l in out['llm'].items() if l}, cfg)
    C = {'lmd': '#2a78d6', 'mordor': '#e34948'}; NAME = {'rf': 'Random Forest', 'lgbm': 'LightGBM', 'if': 'Isolation Forest', 'lstm': 'LSTM'}
    # 1. sensitivity curves
    for m in MODELS:
        grids = [out['sensitivity'][ds][m] for ds in cfg.DATASETS if out['sensitivity'][ds].get(m)]
        if not grids: continue
        hps = list(dict.fromkeys(r['hyperparameter'] for g in grids for r in g['rows']))
        ncol = 2 if len(hps) >= 4 else len(hps); nrow = int(np.ceil(len(hps) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
        for ax, hp in zip(axes.ravel(), hps):
            for ds in cfg.DATASETS:
                g = out['sensitivity'][ds].get(m)
                if not g: continue
                rows = [r for r in g['rows'] if r['hyperparameter'] == hp]
                xs = [str(r['value']) for r in rows]; ax.plot(xs, [r['f1'] for r in rows], marker='o', color=C[ds], label=f"{cfg.DATASETS[ds]} F1")
                ax.plot(xs, [r['roc_auc'] or 0 for r in rows], marker='s', ls='--', color=C[ds], alpha=0.6, label=f"{cfg.DATASETS[ds]} AUC")
            ax.set_title(f"{NAME[m]}: {hp}", fontsize=10); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.set_xlabel(hp)
        for ax in axes.ravel()[len(hps):]: ax.axis('off')
        axes.ravel()[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(F / f"sens_{m}.png", dpi=150); plt.close(fig)
    # 2. per-dataset model comparison (OOF F1 / FPR / FNR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, cfg.DATASETS):
        ms = [m for m in MODELS if out['cv'][ds].get(m)]; x = np.arange(len(ms)); w = 0.26
        for j, k in enumerate(('f1', 'fpr', 'fnr')):
            ax.bar(x + (j - 1) * w, [out['cv'][ds][m]['oof_metrics'][k] for m in ms], w, label=k.upper())
        ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ms], fontsize=8); ax.set_title(f"{cfg.DATASETS[ds]}: out-of-fold performance"); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    axes[0].legend(); fig.tight_layout(); fig.savefig(F / 'cv_models.png', dpi=150); plt.close(fig)
    # 3. cross-dataset transfer: target ROC-AUC per model and variant, one panel per direction (0.5 = no ranking power)
    keys = sorted(out['cross'])
    if keys:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True); variants = [('unified', 'raw'), ('unified', 'quantile'), ('reduced', 'raw'), ('reduced', 'quantile')]
        for ax, (src, tgt) in zip(axes, [('lmd', 'mordor'), ('mordor', 'lmd')]):
            ms = [m for m in MODELS if any(k.startswith(f"cross_{src}_to_{tgt}_{m}_") for k in keys)]; x = np.arange(len(ms)); w = 0.2
            for j, (fs, v) in enumerate(variants):
                vals = [out['cross'].get(f"cross_{src}_to_{tgt}_{m}_{fs}_{v}", {}).get('target_metrics', {}).get('roc_auc', np.nan) for m in ms]
                ax.bar(x + (j - 1.5) * w, vals, w, label=f"{fs}, {v}")
            ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ms], fontsize=9)
            ax.set_title(f"{cfg.DATASETS[src]} → {cfg.DATASETS[tgt]}"); ax.set_ylim(0, 1.0); ax.grid(axis='y', alpha=0.3)
        axes[0].set_ylabel('ROC-AUC on the target'); axes[0].legend(fontsize=8, title='features, alignment', title_fontsize=8)
        fig.suptitle('Cross-dataset transfer: ranking quality on the target (dashed = chance)', fontsize=11); fig.tight_layout(); fig.savefig(F / 'cross_dataset.png', dpi=150); plt.close(fig)
    # 4. cascade stages
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, ds in zip(axes, cfg.DATASETS):
        ca = out['cascade'].get(ds); ll = out['llm'].get(ds)
        if not ca: continue
        stages = [('Stage 1\nLightGBM', ca['stage1']), ('Stage 2\n+RF band', ca['stage2'])]
        if ll and 'stage3' in ll and ll.get('n_llm_in_cascade', 1) > 0: stages.append(('Stage 3\n+LLM', ll['stage3']))
        stages.append(('oracle\narbiter', ca['stage3_oracle_upper_bound'])); x = np.arange(len(stages)); w = 0.26
        for j, k in enumerate(('f1', 'fpr', 'fnr')):
            vals = [s[1][k] for s in stages]; bars = ax.bar(x + (j - 1) * w, vals, w, label=k.upper())
            for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}" if k == 'f1' else f"{100 * v:.2f}%", ha='center', va='bottom', fontsize=6.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([s[0] for s in stages], fontsize=8); ax.set_title(f"{cfg.DATASETS[ds]}: cascade"); ax.set_ylim(0, 1.25); ax.grid(axis='y', alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8); fig.tight_layout(); fig.savefig(F / 'cascade.png', dpi=150); plt.close(fig)
    # 5. LLM context sensitivity (only once the LLM stage has run: an empty plot must not reach the report)
    if not any(out['llm'].get(ds) and out['llm'][ds].get('sensitivity') for ds in cfg.DATASETS):
        (F / 'llm_sensitivity.png').unlink(missing_ok=True); return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, ds in zip(axes, cfg.DATASETS):
        ll = out['llm'].get(ds)
        if not ll or not ll.get('sensitivity'): continue
        abl = list(ll['sensitivity']); acc = [ll['sensitivity'][a]['accuracy'] or 0 for a in abl]; flip = [ll['sensitivity'][a]['flip_rate_vs_full'] or 0 for a in abl]
        x = np.arange(len(abl)); ax.bar(x - 0.18, acc, 0.36, label='accuracy vs label'); ax.bar(x + 0.18, flip, 0.36, label='verdict flips vs full context')
        ax.set_xticks(x); ax.set_xticklabels(abl, fontsize=8, rotation=20); ax.set_title(f"{cfg.DATASETS[ds]}: LLM context ablations"); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(F / 'llm_sensitivity.png', dpi=150); plt.close(fig)
