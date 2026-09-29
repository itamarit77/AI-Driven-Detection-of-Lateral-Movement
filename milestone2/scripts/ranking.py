"""
Chapter 4: tree-based feature ranking on BOTH datasets, with regularization and
permutation importance (not MDI alone, which is cardinality-biased).
Adds (a) a leakage-aware GROUPED split (host x 10-minute block) alongside the random
split, because rolling-window features make adjacent rows nearly identical; and
(b) a Mordor ablation that removes the features overlapping the provenance-label seeds.
"""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import FEATURE_COLS

RF_KW = dict(n_estimators=300, max_depth=8, min_samples_leaf=50, max_features='sqrt',
             class_weight='balanced_subsample', n_jobs=-1, random_state=0)
SEED_OVERLAP = ['f_image_is_lolbin', 'f_cmdline_susp', 'f_lolbin_parent_child']  # overlap with Mordor label seeds

def load(path, sample=400000):
    df = pd.read_parquet(path, columns=FEATURE_COLS + ['label', 'eventid', 'host', 'ts'])
    y = df['label'].values; rng = np.random.RandomState(42)
    if len(df) > sample:
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        npos = min(len(pos), sample//2); nneg = sample-npos
        idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, min(nneg, len(neg)), replace=False)])
        df = df.iloc[idx].reset_index(drop=True)
    df['group'] = df['host'].astype(str) + '|' + df['ts'].dt.floor('10min').astype(str)
    return df

def fit_eval(df, cols, split='random'):
    X = df[cols].fillna(0.0).values; y = df['label'].values
    if split == 'random':
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    else:  # grouped by host x 10-min block: no window straddles train/test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
        tr, te = next(gss.split(X, y, groups=df['group'].values))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)
    pr = rf.predict_proba(Xte)[:, 1]
    return rf, Xte, yte, roc_auc_score(yte, pr), average_precision_score(yte, pr), yte.mean()

def rank(path, name):
    df = load(path)
    rf, Xte, yte, auc_r, ap_r, pos_r = fit_eval(df, FEATURE_COLS, 'random')
    _, _, _, auc_g, ap_g, pos_g = fit_eval(df, FEATURE_COLS, 'grouped')
    mdi = rf.feature_importances_
    perm = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1, scoring='roc_auc')
    Xtr, _, ytr, _ = train_test_split(df[FEATURE_COLS].fillna(0.0).values, df['label'].values, test_size=0.3, stratify=df['label'].values, random_state=7)
    lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=8, num_leaves=31, learning_rate=0.05,
                              min_child_samples=50, class_weight='balanced', random_state=0, verbose=-1).fit(Xtr, ytr)
    gain = lgbm.booster_.feature_importance(importance_type='gain'); gain = gain / gain.sum()
    res = pd.DataFrame({'feature': FEATURE_COLS, 'rf_mdi': mdi, 'rf_perm_auc_drop': perm.importances_mean,
                        'lgbm_gain': gain}).sort_values('rf_perm_auc_drop', ascending=False)
    res.to_csv(ROOT + f'/artifacts/ranking_{name}.csv', index=False)
    summ = {'auc_random': auc_r, 'prauc_random': ap_r, 'auc_grouped': auc_g, 'prauc_grouped': ap_g,
            'test_pos_rate': pos_r, 'n': len(df)}
    # ablations
    no_leak = [c for c in FEATURE_COLS if c not in ('f_weekday', 'f_seconds_in_day', 'f_diurnal_sin', 'f_diurnal_cos', 'f_is_business_hours')]
    _, _, _, a, p_, _ = fit_eval(df, no_leak, 'grouped'); summ['auc_grouped_no_time'] = a; summ['prauc_grouped_no_time'] = p_
    if name == 'Mordor':
        keep = [c for c in no_leak if c not in SEED_OVERLAP]
        _, _, _, a, p_, _ = fit_eval(df, keep, 'grouped'); summ['auc_grouped_no_time_no_seedoverlap'] = a; summ['prauc_grouped_no_time_no_seedoverlap'] = p_
    print(f"\n===== RANKING {name} (n={len(df)}, test attack rate={pos_r:.3f}) =====")
    print({k: round(v, 5) if isinstance(v, float) else v for k, v in summ.items()})
    print(res.round(4).to_string(index=False))
    return res, summ

def cross_compare(rl, rm):
    """Cross-dataset rank agreement on the COMPOSITE basis used in Tables 7-8 and Figures 6-7-9:
    composite = mean of the RF Gini-importance share and the LightGBM gain share (each sums to 1 over the
    42 pool features). Ranks use average ranks for ties. The permutation basis is kept for transparency."""
    from scipy.stats import spearmanr
    def comp(r, tag):
        r = r.copy(); r[tag] = (r['rf_mdi'].clip(lower=0) + r['lgbm_gain'].clip(lower=0)) / 2
        r[tag + '_perm'] = r['rf_perm_auc_drop']; return r[['feature', tag, tag + '_perm']]
    m = comp(rl, 'LMD').merge(comp(rm, 'Mordor'), on='feature')
    for tag in ('LMD', 'Mordor'):
        m['rank_' + tag] = m[tag].rank(ascending=False, method='average')
        m['rank_' + tag + '_perm'] = m[tag + '_perm'].rank(ascending=False, method='average')
    m['rank_gap'] = (m['rank_LMD'] - m['rank_Mordor']).abs()
    m = m.sort_values(['rank_gap', 'feature'], ascending=[False, True])
    m.to_csv(ROOT + '/artifacts/ranking_cross.csv', index=False)
    rho, p = spearmanr(m['LMD'], m['Mordor']); rho_p, p_p = spearmanr(m['LMD_perm'], m['Mordor_perm'])
    print("\n===== CROSS-DATASET RANK COMPARISON (composite basis) ====="); print(m.round(4).head(12).to_string(index=False))
    print(f"Spearman rank corr (composite): rho={rho:.3f} p={p:.3g};  (permutation basis): rho={rho_p:.3f} p={p_p:.3g}")
    return {'cross_rank_spearman': float(rho), 'cross_rank_spearman_p': float(p),
            'cross_rank_spearman_perm': float(rho_p), 'cross_rank_basis': 'composite = mean(RF MDI share, LightGBM gain share)'}

if __name__ == '__main__':
    if '--cross-only' in sys.argv:   # recompute the cross-dataset comparison from the saved per-dataset CSVs
        rl = pd.read_csv(ROOT + '/artifacts/ranking_LMD.csv'); rm = pd.read_csv(ROOT + '/artifacts/ranking_Mordor.csv')
        summ = json.load(open(ROOT + '/artifacts/ranking_summary.json')); summ.update(cross_compare(rl, rm))
        json.dump(summ, open(ROOT + '/artifacts/ranking_summary.json', 'w'), indent=2); sys.exit(0)
    rl, sl = rank(ROOT + '/features/lmd_features.parquet', 'LMD')
    rm, sm = rank(ROOT + '/features/mordor_features.parquet', 'Mordor')
    summ = {'LMD': sl, 'Mordor': sm}; summ.update(cross_compare(rl, rm))
    json.dump(summ, open(ROOT + '/artifacts/ranking_summary.json', 'w'), indent=2)
