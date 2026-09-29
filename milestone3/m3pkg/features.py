"""Stage 2 — feature engineering (one shared function for every dataset) and Stage 3 — preprocessing:
feature sets, grouped stratified splits, inner validation hold-out, scaling, class weights, quantile alignment.
Nothing here knows which dataset it is processing.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from .common_features import build_features, FEATURE_COLS, FAMILY, CAT_ORDINAL

log = logging.getLogger("m3.features")

TIME5 = ['f_seconds_in_day', 'f_diurnal_sin', 'f_diurnal_cos', 'f_weekday', 'f_is_business_hours']
EXCLUDED = TIME5 + ['f_cmdline_len', 'f_naive_relational_pid']
UNIFIED = [c for c in FEATURE_COLS if c not in EXCLUDED]                       # 35 prospective model inputs (Milestone 2 Table 9, unified schema)
SEED_FIELD = FAMILY['lineage'] + FAMILY['cmdline']                              # circular with Mordor's provenance labels
REDUCED = [c for c in UNIFIED if c not in SEED_FIELD]                          # Mordor-honest set (Milestone 2 §3.1.1 / Table 10 remedies)
CAT_LEVELS = sorted(CAT_ORDINAL.values()) + [-1]                               # fixed one-hot vocabulary (registry, incl. "other")
CONTINUOUS = [c for c in UNIFIED if c in FAMILY['temporal'] + FAMILY['velocity'] + FAMILY['fanout'] + FAMILY['graph'] + FAMILY['user'] + ['f_cmdline_entropy'] and c != 'f_rare_dst_pair']


def make_features(events, source_name):
    feats = build_features(events, source_name)
    feats['row_id'] = feats['row_id'].astype(np.int64)
    return feats


# --------------------------------------------------------------------------------------- splits
def group_key(feats, minutes):
    return (feats['host'].astype(str) + '|' + feats['ts'].dt.floor(f'{minutes}min').astype(str)).values


def outer_folds(feats, n_folds, minutes, seed):
    """StratifiedGroupKFold: every host x <minutes> block is entirely in one fold; positives balanced across folds."""
    y = feats['label'].values; g = group_key(feats, minutes); n_groups = len(np.unique(g))
    k = min(n_folds, max(2, n_groups // 3))
    if k < n_folds:
        log.warning("only %d groups: using %d folds instead of %d", n_groups, k, n_folds)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in sgkf.split(np.zeros(len(y)), y, g)]
    good = [(tr, te) for tr, te in folds if len(np.unique(y[te])) == 2 and len(np.unique(y[tr])) == 2]
    if len(good) < len(folds):
        log.warning("%d fold(s) dropped because a split had a single class", len(folds) - len(good))
    return good


def inner_holdout(feats, train_idx, frac, minutes, seed):
    """Grouped hold-out INSIDE a training fold: used for the decision threshold, early stopping and cascade bands.
    The outer test fold is never touched by any of these choices."""
    g = group_key(feats.iloc[train_idx], minutes); y = feats['label'].values[train_idx]
    for attempt in range(20):                                   # re-draw until both classes appear on both sides
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed + 1000 * attempt)
        fit_rel, val_rel = next(gss.split(np.zeros(len(train_idx)), groups=g))
        if len(np.unique(y[fit_rel])) == 2 and len(np.unique(y[val_rel])) == 2:
            break
    else:
        log.warning("inner hold-out could not obtain both classes on both sides (using the last draw)")
    return train_idx[fit_rel], train_idx[val_rel]


# --------------------------------------------------------------------------------------- matrices
def one_hot_category(feats):
    cat = feats['f_event_category'].values
    return np.stack([(cat == lv).astype(np.float32) for lv in CAT_LEVELS], axis=1)


def design_matrix(feats, cols, one_hot=False):
    """Tree models use the integer category code (order-invariant splits); distance/gradient models get one-hot."""
    X = feats[cols].fillna(0).astype(np.float32).values
    if one_hot and 'f_event_category' in cols:
        keep = [i for i, c in enumerate(cols) if c != 'f_event_category']
        X = np.concatenate([X[:, keep], one_hot_category(feats)], axis=1)
    return X


class RobustScalerNP:
    """Median / IQR scaling fitted on TRAINING rows only (zero IQR -> scale 1). Applied to continuous columns."""
    def __init__(self, cols, cont_idx):
        self.cols, self.idx = cols, cont_idx
    def fit(self, X):
        sub = X[:, self.idx]; self.med = np.median(sub, axis=0); q1, q3 = np.percentile(sub, [25, 75], axis=0)
        self.iqr = q3 - q1; self.iqr[self.iqr == 0] = 1.0; return self
    def transform(self, X):
        X = X.copy(); X[:, self.idx] = (X[:, self.idx] - self.med) / self.iqr; return X


def quantile_align(X_src, X_tgt, cont_idx):
    """Within-dataset percentile ranks for continuous columns (label-free): 'top-5% fan-out in THIS environment'
    becomes the transferable quantity. Each dataset is ranked against its own distribution."""
    def ranks(M):
        M = M.copy()
        for j in cont_idx:                       # ties -> mean percentile rank
            M[:, j] = pd.Series(M[:, j]).rank(method='average', pct=True).values.astype(np.float32)
        return M
    return ranks(X_src), ranks(X_tgt)


def pos_weight(y):
    n_pos = max(int(y.sum()), 1); return float((len(y) - n_pos) / n_pos)


def continuous_idx(cols):
    return [i for i, c in enumerate(cols) if c in CONTINUOUS]


# --------------------------------------------------------------------------------------- LSTM windows
def lstm_windows(feats, T, stride):
    """Window i = the T events of the same host that end at event e_i (pre-padded with zeros + mask when a host has
    fewer than T earlier events). Windows end at every stride-th event of each host, so every window's label is the
    label of its last event and the decision is causal.  Returns (end_positions, start_offsets) as arrays of row
    positions in `feats` (which is sorted by host, ts)."""
    host = feats['host'].values; n = len(feats)
    # position of the first row of each host block (feats is sorted by host then ts)
    change = np.r_[0, np.flatnonzero(host[1:] != host[:-1]) + 1]
    block_start = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(change):
        e = change[i + 1] if i + 1 < len(change) else n
        block_start[s:e] = s
    ends = np.arange(n)[(np.arange(n) - block_start) % stride == 0]
    starts = np.maximum(ends - T + 1, block_start[ends])
    return ends, starts
