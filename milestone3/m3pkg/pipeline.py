"""Training / evaluation orchestration (Steps 7-8): grouped stratified CV per model per dataset with inner-validation
thresholds, out-of-fold (OOF) scores, forensic error analysis, hyper-parameter sensitivity, and cross-dataset transfer.
Every function receives feature tables that came out of the shared feature stage, never raw data."""
import json, logging, time
import numpy as np
import pandas as pd
import joblib
from .features import (UNIFIED, REDUCED, design_matrix, outer_folds, inner_holdout, RobustScalerNP, continuous_idx,
                       quantile_align)
from .models import make_model, seed_everything
from .windows import WindowSet
from .evaluate import metrics, metrics_rowthr, best_f1_threshold, error_analysis, fold_summary, cascade_bands

log = logging.getLogger("m3.pipeline")
TREE_MODELS = ('rf', 'lgbm'); FLAT_MODELS = ('rf', 'lgbm', 'if')


def _matrix(feats, cols, model):
    """Trees: integer category code. Isolation Forest / LSTM: one-hot category (distance/gradient based)."""
    return design_matrix(feats, cols, one_hot=(model in ('if', 'lstm')))


def cv_flat_model(name, params, feats, cols, cfg, gm, models_dir=None, tag=''):
    """5-fold StratifiedGroupKFold. Inside each training fold a grouped inner hold-out chooses the F1-optimal
    threshold (and drives LightGBM early stopping). Returns OOF scores, per-fold metrics and the default-threshold
    (0.5) metrics used as the v0 baseline of the optimisation log."""
    seed_everything(cfg.SEED)
    X = _matrix(feats, cols, name); y = feats['label'].values
    folds = outer_folds(feats, cfg.N_FOLDS, gm, cfg.SEED)
    oof = np.full(len(y), np.nan, dtype=np.float64); fold_id = np.full(len(y), -1, dtype=np.int8); thr_folds, fold_m, fold_m05, extra, bands, bands_raw = [], [], [], [], [], []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time(); fit_idx, val_idx = inner_holdout(feats, tr, cfg.INNER_VAL_FRACTION, gm, cfg.SEED + k); fold_id[te] = k
        m = make_model(name, params, early_stopping=cfg.LGBM_EARLY_STOPPING) if name == 'lgbm' else make_model(name, params)
        if name == 'if':
            sc = RobustScalerNP(cols, continuous_idx([c for c in cols if c != 'f_event_category'])).fit(X[fit_idx])
            Xf, Xv, Xt = sc.transform(X[fit_idx]), sc.transform(X[val_idx]), sc.transform(X[te])
            m.fit(Xf)                                             # unsupervised: no labels
        else:
            Xf, Xv, Xt = X[fit_idx], X[val_idx], X[te]; m.fit(Xf, y[fit_idx], Xv, y[val_idx])
        sv = m.score(Xv); thr = best_f1_threshold(y[val_idx], sv); s = m.score(Xt); oof[te] = s
        bands.append(cascade_bands(y[val_idx], sv, cfg.CASCADE_PRECISION_TARGET, cfg.CASCADE_NPV_TARGET))
        bands_raw.append(cascade_bands(y[val_idx], sv, cfg.CASCADE_PRECISION_TARGET, cfg.CASCADE_NPV_TARGET, raw=True))
        fm = metrics(y[te], s, thr); fm['fold'] = k; fm['n_train'] = int(len(fit_idx)); fm['n_val'] = int(len(val_idx)); fm['inner_f1'] = metrics(y[val_idx], sv, thr)['f1']   # inner hold-out F1 at the chosen threshold (selection criterion of the refine stage)
        fm['seconds'] = round(time.time() - t0, 1); fold_m.append(fm); thr_folds.append(thr); fold_m05.append(metrics(y[te], s, 0.5))
        if name == 'lgbm': extra.append({'fold': k, 'best_iteration': m.best_iteration, 'best_inner_auc': m.best_inner_auc, 'rounds_run': m.rounds_run, 'inner_auc_curve': m.inner_auc_curve})
        log.info("%s %s fold %d: F1 %.4f FPR %.4f FNR %.4f AUC %s thr %.3f (%.0fs)", tag, name, k, fm['f1'], fm['fpr'], fm['fnr'], fm['roc_auc'], thr, fm['seconds'])
        if models_dir is not None and k == 0:
            joblib.dump(m.artifact(), models_dir / f"{tag}_{name}_fold0.pkl", compress=3)
    thr_global = float(np.median(thr_folds)); valid = ~np.isnan(oof)
    thr_row = np.array([thr_folds[f] if f >= 0 else thr_global for f in fold_id])
    oof_m = metrics_rowthr(y[valid], oof[valid], thr_row[valid]); oof_m05 = metrics(y[valid], oof[valid], 0.5)
    return {'model': name, 'features': cols, 'n_features': len(cols), 'params': {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in params.items()},
            'fold_metrics': fold_m, 'fold_metrics_thr05': fold_m05, 'fold_thresholds': thr_folds, 'threshold': thr_global,
            'summary': fold_summary(fold_m), 'summary_thr05': fold_summary(fold_m05), 'oof_metrics': oof_m, 'oof_metrics_thr05': oof_m05,
            'lgbm_iterations': extra, 'fold_bands': bands, 'fold_bands_raw': bands_raw, 'oof_scores': oof, 'fold_id': fold_id}


def cv_lstm(params, feats, cols, cfg, gm, stride, n_folds, models_dir=None, tag=''):
    """Same outer folds; the LSTM sees T-event per-host windows ending at every stride-th event. Robust scaling and the
    one-hot category are fitted/defined on the training rows. Metrics are computed on the window-end events of the
    outer test fold (a deterministic subsample of its rows)."""
    seed_everything(cfg.SEED)
    X = _matrix(feats, cols, 'lstm'); y = feats['label'].values
    cont = continuous_idx([c for c in cols if c != 'f_event_category'])
    folds = outer_folds(feats, cfg.N_FOLDS, gm, cfg.SEED)[:n_folds]
    T = params['T']; ws = WindowSet(X, y, feats, T, stride)
    oof = np.full(len(y), np.nan, dtype=np.float64); fold_id = np.full(len(y), -1, dtype=np.int8); fold_m, fold_m05, thr_folds, hist = [], [], [], []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time(); fit_idx, val_idx = inner_holdout(feats, tr, cfg.INNER_VAL_FRACTION, gm, cfg.SEED + k)
        sc = RobustScalerNP(cols, cont).fit(X[fit_idx]); ws.X = sc.transform(X)          # scaler fitted on training rows only
        w_fit, w_val, w_te = ws.window_index_for_events(fit_idx), ws.window_index_for_events(val_idx), ws.window_index_for_events(te)
        m = make_model('lstm', params, seed=cfg.SEED + k).fit(ws, w_fit, w_val)
        thr = best_f1_threshold(ws.y[w_val], m.score(ws, w_val)); s = m.score(ws, w_te); ends = ws.ends[w_te]; oof[ends] = s; fold_id[ends] = k
        fm = metrics(y[ends], s, thr); fm['fold'] = k; fm['n_windows_train'] = int(len(w_fit)); fm['n_windows_test'] = int(len(w_te))
        fm['seconds'] = round(time.time() - t0, 1); fold_m.append(fm); thr_folds.append(thr); fold_m05.append(metrics(y[ends], s, 0.5)); hist.append(m.history)
        log.info("%s lstm fold %d: F1 %.4f FPR %.4f FNR %.4f AUC %s (%.0fs)", tag, k, fm['f1'], fm['fpr'], fm['fnr'], fm['roc_auc'], fm['seconds'])
        if models_dir is not None and k == 0:
            art = m.artifact()
            try:
                import torch; torch.save(art, models_dir / f"{tag}_lstm_fold0.pt")
            except Exception:
                joblib.dump(art, models_dir / f"{tag}_lstm_fold0.pkl")
    valid = ~np.isnan(oof); thr_global = float(np.median(thr_folds))
    thr_row = np.array([thr_folds[f] if f >= 0 else thr_global for f in fold_id])
    return {'model': 'lstm', 'features': cols, 'n_features_input': int(ws.d), 'params': dict(params), 'stride': stride, 'n_folds_used': len(folds),
            'fold_metrics': fold_m, 'fold_metrics_thr05': fold_m05, 'fold_thresholds': thr_folds, 'threshold': thr_global,
            'summary': fold_summary(fold_m), 'summary_thr05': fold_summary(fold_m05),
            'oof_metrics': metrics_rowthr(y[valid], oof[valid], thr_row[valid]), 'oof_metrics_thr05': metrics(y[valid], oof[valid], 0.5),
            'training_history': hist, 'oof_scores': oof, 'fold_id': fold_id}


def run_error_analysis(feats, res, cols):
    """Error forensics on the OOF decisions (row-wise fold thresholds when available)."""
    valid = ~np.isnan(res['oof_scores']); sc = res['oof_scores'].copy()
    if res.get('fold_id') is not None:
        thr_row = np.array([res['fold_thresholds'][f] if f >= 0 else res['threshold'] for f in res['fold_id']]); sc = sc - thr_row + 0.5; thr = 0.5
    else:
        thr = res['threshold']
    return error_analysis(feats.loc[valid].reset_index(drop=True), feats['label'].values[valid], sc[valid], thr, [c for c in cols if c != 'f_event_category'])


# --------------------------------------------------------------------------------------------- sensitivity
def sensitivity_flat(name, base_params, grid, feats, cols, cfg, gm):
    """One-at-a-time variation of the primary hyper-parameters on outer fold 0 (threshold from the inner hold-out)."""
    X = _matrix(feats, cols, name); y = feats['label'].values
    tr, te = outer_folds(feats, cfg.N_FOLDS, gm, cfg.SEED)[0]
    fit_idx, val_idx = inner_holdout(feats, tr, cfg.INNER_VAL_FRACTION, gm, cfg.SEED)
    sc = RobustScalerNP(cols, continuous_idx([c for c in cols if c != 'f_event_category'])).fit(X[fit_idx]) if name == 'if' else None
    Xf, Xv, Xt = (sc.transform(X[fit_idx]), sc.transform(X[val_idx]), sc.transform(X[te])) if sc else (X[fit_idx], X[val_idx], X[te])
    rows = []
    for hp, values in grid.items():
        for v in values:
            p = dict(base_params); p[hp] = v; t0 = time.time()
            m = make_model(name, p, early_stopping=cfg.LGBM_EARLY_STOPPING) if name == 'lgbm' else make_model(name, p)
            if name == 'if': m.fit(Xf)
            else: m.fit(Xf, y[fit_idx], Xv, y[val_idx])
            thr = best_f1_threshold(y[val_idx], m.score(Xv)); fm = metrics(y[te], m.score(Xt), thr)
            rows.append({'hyperparameter': hp, 'value': v if v is not None else 'None', **{k: fm[k] for k in ('f1', 'precision', 'recall', 'fpr', 'fnr', 'roc_auc', 'pr_auc')}, 'seconds': round(time.time() - t0, 1)})
            log.info("sens %s %s=%s F1 %.4f AUC %s", name, hp, v, fm['f1'], fm['roc_auc'])
    return rows


def sensitivity_lstm(base_params, grid, feats, cols, cfg, gm, stride):
    X = _matrix(feats, cols, 'lstm'); y = feats['label'].values; cont = continuous_idx([c for c in cols if c != 'f_event_category'])
    tr, te = outer_folds(feats, cfg.N_FOLDS, gm, cfg.SEED)[0]
    fit_idx, val_idx = inner_holdout(feats, tr, cfg.INNER_VAL_FRACTION, gm, cfg.SEED)
    Xs = RobustScalerNP(cols, cont).fit(X[fit_idx]).transform(X); rows = []
    for hp, values in grid.items():
        for v in values:
            p = dict(base_params); p[hp] = v; t0 = time.time(); ws = WindowSet(Xs, y, feats, p['T'], stride)
            w_fit, w_val, w_te = ws.window_index_for_events(fit_idx), ws.window_index_for_events(val_idx), ws.window_index_for_events(te)
            m = make_model('lstm', p, seed=cfg.SEED).fit(ws, w_fit, w_val)
            thr = best_f1_threshold(ws.y[w_val], m.score(ws, w_val)); fm = metrics(ws.y[w_te], m.score(ws, w_te), thr)
            rows.append({'hyperparameter': hp, 'value': v, **{k: fm[k] for k in ('f1', 'precision', 'recall', 'fpr', 'fnr', 'roc_auc', 'pr_auc')}, 'seconds': round(time.time() - t0, 1)})
            log.info("sens lstm %s=%s F1 %.4f AUC %s", hp, v, fm['f1'], fm['roc_auc'])
    return rows


# --------------------------------------------------------------------------------------------- cross-dataset
def cross_dataset(name, params, feats_src, feats_tgt, cols, cfg, gm, variant='raw'):
    """Train on ALL rows of the source dataset (inner hold-out for threshold / early stopping), test on ALL rows of the
    target. variant: 'raw' (source-fitted robust scaling only where the model needs it), 'quantile' (within-dataset
    percentile ranks of the continuous columns on both sides)."""
    seed_everything(cfg.SEED)
    Xs = _matrix(feats_src, cols, name); ys = feats_src['label'].values; Xt = _matrix(feats_tgt, cols, name); yt = feats_tgt['label'].values
    cont = continuous_idx([c for c in cols if c != 'f_event_category'])
    if variant == 'quantile':
        Xs, Xt = quantile_align(Xs, Xt, cont)
    all_idx = np.arange(len(ys)); fit_idx, val_idx = inner_holdout(feats_src, all_idx, cfg.INNER_VAL_FRACTION, gm, cfg.SEED)
    if name == 'if':
        sc = RobustScalerNP(cols, cont).fit(Xs[fit_idx]); Xf, Xv, Xtt = sc.transform(Xs[fit_idx]), sc.transform(Xs[val_idx]), sc.transform(Xt)
        m = make_model('if', params).fit(Xf)
    else:
        Xf, Xv, Xtt = Xs[fit_idx], Xs[val_idx], Xt
        m = make_model(name, params, early_stopping=cfg.LGBM_EARLY_STOPPING) if name == 'lgbm' else make_model(name, params)
        m.fit(Xf, ys[fit_idx], Xv, ys[val_idx])
    thr = best_f1_threshold(ys[val_idx], m.score(Xv)); st = m.score(Xtt)
    return {'model': name, 'variant': variant, 'n_features': len(cols), 'threshold_source_val': thr,
            'target_metrics': metrics(yt, st, thr), 'target_metrics_thr05': metrics(yt, st, 0.5),
            'target_metrics_oracle_thr': metrics(yt, st, best_f1_threshold(yt, st)),   # upper bound: NOT a legitimate deployment number
            'scores': st}


def cross_dataset_lstm(params, feats_src, feats_tgt, cols, cfg, gm, stride_src, stride_tgt, variant='raw'):
    seed_everything(cfg.SEED)
    Xs = _matrix(feats_src, cols, 'lstm'); ys = feats_src['label'].values; Xt = _matrix(feats_tgt, cols, 'lstm'); yt = feats_tgt['label'].values
    cont = continuous_idx([c for c in cols if c != 'f_event_category'])
    if variant == 'quantile':
        Xs, Xt = quantile_align(Xs, Xt, cont)
    all_idx = np.arange(len(ys)); fit_idx, val_idx = inner_holdout(feats_src, all_idx, cfg.INNER_VAL_FRACTION, gm, cfg.SEED)
    sc = RobustScalerNP(cols, cont).fit(Xs[fit_idx]); ws_s = WindowSet(sc.transform(Xs), ys, feats_src, params['T'], stride_src)
    ws_t = WindowSet(sc.transform(Xt), yt, feats_tgt, params['T'], stride_tgt)
    w_fit, w_val = ws_s.window_index_for_events(fit_idx), ws_s.window_index_for_events(val_idx)
    m = make_model('lstm', params, seed=cfg.SEED).fit(ws_s, w_fit, w_val)
    thr = best_f1_threshold(ws_s.y[w_val], m.score(ws_s, w_val)); w_all = np.arange(ws_t.n); st = m.score(ws_t, w_all)
    scores = np.full(len(yt), np.nan, dtype=np.float64); scores[ws_t.ends] = st
    return {'model': 'lstm', 'variant': variant, 'threshold_source_val': thr, 'target_metrics': metrics(ws_t.y, st, thr),
            'target_metrics_thr05': metrics(ws_t.y, st, 0.5), 'target_metrics_oracle_thr': metrics(ws_t.y, st, best_f1_threshold(ws_t.y, st)), 'scores': scores}
