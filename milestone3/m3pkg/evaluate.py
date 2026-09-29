"""Stage 5 helpers — metrics, inner-validation threshold selection, confusion matrices and forensic error analysis.
All functions take (labels, scores) arrays, so they are model- and dataset-agnostic."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .common_features import CAT_ORDINAL

CAT_NAME = {v: k for k, v in CAT_ORDINAL.items()}; CAT_NAME[-1] = 'other'


EPS = 1e-7   # scores and thresholds are compared in float64 with a tolerance (a threshold can equal a score value)


def confusion(y, s, thr):
    yhat = (np.asarray(s, dtype=np.float64) >= np.asarray(thr, dtype=np.float64) - EPS).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics(y, s, thr):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64); tp, fp, fn, tn = confusion(y, s, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {'n': int(len(y)), 'pos': int(y.sum()), 'thr': float(thr), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
           'precision': prec, 'recall': rec, 'f1': f1, 'fpr': fp / (fp + tn) if fp + tn else 0.0,
           'fnr': fn / (fn + tp) if fn + tp else 0.0, 'accuracy': (tp + tn) / len(y) if len(y) else 0.0}
    two = len(np.unique(y)) == 2
    out['roc_auc'] = float(roc_auc_score(y, s)) if two else None
    out['pr_auc'] = float(average_precision_score(y, s)) if two else None
    return out


def metrics_rowthr(y, s, thr_row):
    """Metrics when every row is thresholded by ITS OWN fold's threshold (thr_row is an array aligned with y):
    confusion counts from the row-wise decisions, ranking metrics from the raw scores."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64); thr_row = np.asarray(thr_row, dtype=np.float64)
    yhat = (s >= thr_row - EPS).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    out = {'n': int(len(y)), 'pos': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': prec, 'recall': rec,
           'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0, 'fpr': fp / (fp + tn) if fp + tn else 0.0,
           'fnr': fn / (fn + tp) if fn + tp else 0.0, 'accuracy': (tp + tn) / len(y) if len(y) else 0.0}
    two = len(np.unique(y)) == 2
    out['roc_auc'] = float(roc_auc_score(y, s)) if two else None; out['pr_auc'] = float(average_precision_score(y, s)) if two else None
    out['thr'] = 'per-fold'; return out


def best_f1_threshold(y_val, s_val, grid=200):
    """Threshold maximising F1 on the inner validation hold-out (never on the outer test fold)."""
    y_val = np.asarray(y_val); s_val = np.asarray(s_val)
    if len(np.unique(y_val)) < 2:
        return 0.5
    cands = np.unique(np.quantile(s_val, np.linspace(0.0, 1.0, grid)))
    best, best_t = -1.0, 0.5
    for t in cands:
        m = metrics(y_val, s_val, t)
        if m['f1'] > best:
            best, best_t = m['f1'], float(t)
    return best_t


def cascade_bands(y_val, s_val, precision_target, npv_target, grid=400, raw=False):
    """tau_hi = smallest threshold whose precision on inner validation >= target (confident attack);
       tau_lo = largest threshold whose negative predictive value >= target (confident benign)."""
    y_val = np.asarray(y_val); s_val = np.asarray(s_val); cands = np.unique(np.quantile(s_val, np.linspace(0, 1, grid)))
    tau_hi, tau_lo = None, None
    for t in cands:                                        # ascending
        tp, fp, fn, tn = confusion(y_val, s_val, t)
        if tau_hi is None and tp + fp > 0 and tp / (tp + fp) >= precision_target:
            tau_hi = float(t)
        npv = tn / (tn + fn) if tn + fn else 1.0
        if npv >= npv_target:
            tau_lo = float(t)
    if tau_hi is None: tau_hi = float(cands[-1])
    if tau_lo is None: tau_lo = float(cands[0])
    if raw: return tau_lo, tau_hi
    if tau_lo >= tau_hi:                                   # the two criteria cross (precision target reached below the NPV
        mid = (tau_lo + tau_hi) / 2; tau_lo, tau_hi = mid - 1e-6, mid + 1e-6   # limit): collapse the band to a point at the midpoint
    return tau_lo, tau_hi


def error_analysis(feats_test, y, s, thr, cols, top_k=25, examples=20):
    """Characterise false positives / false negatives of one model on one dataset.
    Returns a JSON-serialisable summary: composition by event category and host, feature shifts (median of the error
    group minus median of the correctly classified group of the same true class, in training-IQR units is not
    available here, so raw medians are reported side by side), and the most confident errors (row_ids)."""
    y = np.asarray(y); s = np.asarray(s); yhat = (s >= thr).astype(int)
    fp = (yhat == 1) & (y == 0); fn = (yhat == 0) & (y == 1); tn = (yhat == 0) & (y == 0); tp = (yhat == 1) & (y == 1)
    cat = feats_test['f_event_category'].map(CAT_NAME).fillna('other').values; host = feats_test['host'].astype(str).values
    def comp(mask, ref):
        if mask.sum() == 0: return {}
        c = pd.Series(cat[mask]).value_counts(normalize=True).head(8); r = pd.Series(cat[ref]).value_counts(normalize=True)
        return {k: {'share': round(float(v), 3), 'share_in_reference': round(float(r.get(k, 0.0)), 3)} for k, v in c.items()}
    def hosts(mask):
        return {} if mask.sum() == 0 else pd.Series(host[mask]).value_counts(normalize=True).head(6).round(3).to_dict()
    def feat_shift(err, ok):
        if err.sum() == 0: return {}
        sub_e = feats_test.loc[err, cols].median(); sub_o = feats_test.loc[ok, cols].median() if ok.sum() else sub_e * 0
        d = (sub_e - sub_o); scale = feats_test[cols].quantile(0.75) - feats_test[cols].quantile(0.25); scale[scale == 0] = 1.0
        z = (d / scale).sort_values(key=np.abs, ascending=False).head(top_k)
        return {c: {'error_median': float(sub_e[c]), 'reference_median': float(sub_o[c]), 'shift_iqr': float(z[c])} for c in z.index}
    def top_examples(mask, descending):
        idx = np.flatnonzero(mask)
        if len(idx) == 0: return []
        order = idx[np.argsort(s[idx])[::-1] if descending else np.argsort(s[idx])]
        return [{'row_id': int(feats_test['row_id'].values[i]), 'score': float(s[i]), 'category': str(cat[i]), 'host': str(host[i])} for i in order[:examples]]
    return {'thr': float(thr), 'counts': {'tp': int(tp.sum()), 'fp': int(fp.sum()), 'fn': int(fn.sum()), 'tn': int(tn.sum())},
            'fp_by_category': comp(fp, tn), 'fn_by_category': comp(fn, tp),
            'fp_by_host': hosts(fp), 'fn_by_host': hosts(fn),
            'fp_feature_shift_vs_tn': feat_shift(fp, tn), 'fn_feature_shift_vs_tp': feat_shift(fn, tp),
            'fp_score_quantiles': [float(x) for x in np.quantile(s[fp], [0.5, 0.9, 0.99])] if fp.sum() else [],
            'fn_score_quantiles': [float(x) for x in np.quantile(s[fn], [0.01, 0.1, 0.5])] if fn.sum() else [],
            'most_confident_fp': top_examples(fp, True), 'most_confident_fn': top_examples(fn, False)}


def fold_summary(fold_metrics):
    keys = ['f1', 'precision', 'recall', 'fpr', 'fnr', 'roc_auc', 'pr_auc']
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} if vals else None
    return out
