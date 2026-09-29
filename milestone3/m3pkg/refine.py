"""Step 8 — diagnostic-driven pipeline refinement (report §2.2, run by `run_m3.py --stage refine`).

The forensic error analysis of §2.1 (read after the pre-registered run) located the supervised models' errors in
(i) LMD-2023: process-centric windows — benign process-creation / DNS bursts (false positives) and attack rows in
windows with several users and admin-port connections (false negatives) — and (ii) Mordor: DLL-load events whose
trailing windows differ from the attack bursts (quieter windows for the false positives on the densest host).
Two label-free, causal feature families follow from that, computed per host from the event stream only:

  A. window composition  — the share of each event category in the host's trailing 60-s window
                           (count of the category / all events in the window);
  B. burst ratio         — the 60-s event rate and process-creation rate relative to the host's own trailing-hour
                           mean per-minute rate (a per-host normalisation of the velocity features).

They are evaluated as increments (base -> +A -> +A+B) for the two supervised per-event models of the cascade, on the
same outer folds, the same inner hold-outs and the same hyper-parameters as the main run. The candidate families were
fixed from §2.1 before this stage ran; per fold, the 'selected' variant is the one with the best F1 on the inner
hold-out (never on the test fold). That protects the variant choice only: the families themselves were designed after
inspecting the earlier out-of-fold errors and the evaluation reuses the outer folds, so the gains are exploratory
refinement results, not an independent confirmation. The main-run baseline is refitted here as well so that the
before/after comparison comes from one run (and is checked against cv_<ds>_<model>.json).
"""
import logging
import numpy as np
import pandas as pd
from .common_features import EVENT_CATEGORY
from .evaluate import metrics_rowthr

log = logging.getLogger("m3.refine")

COMPOSITION_CATS = ['process_create', 'network_conn', 'process_end', 'image_load', 'process_access', 'file_create', 'registry', 'dns']
REFINE_A = [f"f_share_{c}_60s" for c in COMPOSITION_CATS]                     # window composition (8)
REFINE_B = ['f_evt_burst_1h', 'f_proc_burst_1h']                                # burst ratios (2)
VARIANTS = {'base': [], 'A': REFINE_A, 'AB': REFINE_A + REFINE_B}


def add_refinement_features(feats):
    """feats: the feature table of the shared feature stage (sorted by host, ts; columns host, ts, eventid, f_*).
    Adds the REFINE_A / REFINE_B columns. Everything is a trailing per-host window statistic: causal and label-free."""
    feats = feats.sort_values(['host', 'ts']).reset_index(drop=True)
    cat = feats['eventid'].map(lambda e: EVENT_CATEGORY.get(int(e) if pd.notna(e) else -1, 'other'))
    idx = pd.DatetimeIndex(feats['ts'].values); hosts = feats['host'].values

    def roll(mask, window):
        s = pd.Series(np.where(np.asarray(mask), 1.0, 0.0), index=idx)
        return s.groupby(hosts).rolling(window).sum().reset_index(level=0, drop=True).values

    total60 = roll(np.ones(len(feats), dtype=bool), '60s')                    # == f_host_evt_rate_60s (the event itself included, so >= 1)
    for c in COMPOSITION_CATS:
        feats[f"f_share_{c}_60s"] = roll(cat.values == c, '60s') / np.maximum(total60, 1.0)
    total1h = roll(np.ones(len(feats), dtype=bool), '3600s'); proc1h = roll(feats['eventid'].values == 1, '3600s')
    proc60 = roll(feats['eventid'].values == 1, '60s')
    feats['f_evt_burst_1h'] = total60 / np.maximum(total1h / 60.0, 1e-9)       # current minute vs the host's mean minute over the trailing hour (<= 60)
    feats['f_proc_burst_1h'] = proc60 / np.maximum(proc1h / 60.0, 1.0 / 60.0)  # same for process creations (0 when none in the hour)
    return feats


def error_by_category(feats, oof, fold_id, thr_folds):
    """FP / FN counts per event category at the fold thresholds (what the refinement was meant to suppress)."""
    y = feats['label'].values; valid = fold_id >= 0
    thr = np.array([thr_folds[f] if f >= 0 else 0.5 for f in fold_id]); pred = (oof >= thr - 1e-7).astype(int)
    cat = feats['eventid'].map(lambda e: EVENT_CATEGORY.get(int(e) if pd.notna(e) else -1, 'other')).values
    fp = pd.Series(cat[valid & (pred == 1) & (y == 0)]).value_counts(); fn = pd.Series(cat[valid & (pred == 0) & (y == 1)]).value_counts()
    top_host_fp = pd.Series(feats['host'].values[valid & (pred == 1) & (y == 0)]).value_counts()
    return {'fp': {str(k): int(v) for k, v in fp.items()}, 'fn': {str(k): int(v) for k, v in fn.items()}, 'fp_total': int(fp.sum()), 'fn_total': int(fn.sum()),
            'top_host_fp': ({str(top_host_fp.index[0]): int(top_host_fp.iloc[0])} if len(top_host_fp) else {})}


def select_on_inner(results, y, fold_id, feats=None):
    """Per-fold adoption: take the variant with the best inner-hold-out F1 (ties -> the earlier variant, base first).
    The variant choice never looks at the test fold; note that the candidate families themselves were designed after
    inspecting the earlier out-of-fold errors, so the evaluation reuses the outer folds (an exploratory refinement).
    results: {variant: cv_flat_model result}. With feats, also counts the composite policy's FP / FN per event category."""
    names = list(results); n_folds = len(results[names[0]]['fold_metrics']); chosen = []; oof = np.full(len(y), np.nan); thr_folds = []
    for k in range(n_folds):
        inner = [results[v]['fold_metrics'][k].get('inner_f1', -1.0) for v in names]; best = names[int(np.argmax(inner))]; chosen.append(best)
        m = fold_id == k; oof[m] = results[best]['oof_scores'][m]; thr_folds.append(results[best]['fold_thresholds'][k])
    thr = np.array([thr_folds[f] if f >= 0 else 0.5 for f in fold_id]); valid = ~np.isnan(oof)
    out = {'chosen_per_fold': chosen, 'oof_metrics': metrics_rowthr(y[valid], oof[valid], thr[valid]), 'fold_thresholds': thr_folds,
           'inner_f1_per_fold': {v: [round(fm.get('inner_f1', float('nan')), 4) for fm in results[v]['fold_metrics']] for v in names}}
    if feats is not None: out['error_by_category'] = error_by_category(feats, oof, fold_id, thr_folds)
    return out
