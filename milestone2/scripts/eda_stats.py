"""
Chapter 3 EDA: per-feature class-divergence statistics computed on ALL records of each dataset.

Analysis unit: one event row. Rows are NOT independent (rolling-window features make adjacent rows
correlated), so p-values are anti-conservative and are reported for completeness only; effect sizes
(attack-positive AUC / Cliff's delta, Cramer's V, mutual information, JS divergence) are the evidence.

Mutual information is a plug-in (histogram) estimator in BITS: continuous features are quantile-binned
into <=32 bins, integer/binary features use their exact values. An automated check asserts
0 <= MI <= H(label) for every feature; constant features get MI = 0 by construction.
"""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import rel_entr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import FEATURE_COLS

DISCRETE = {'f_is_business_hours','f_lolbin_parent_child','f_remote_exec_parent','f_remote_exec_after_inbound',
            'f_image_is_lolbin','f_exec_from_staging','f_cmdline_susp','f_integrity_high','f_admin_port_flag',
            'f_initiated','f_is_ipv6','f_lsass_access','f_cred_access_mask','f_exec_file_drop','f_file_drop_staging',
            'f_naive_relational_pid','f_rare_dst_pair','f_explicit_cred_4648','f_network_logon_t3','f_kerberos_tgs_4769',
            'f_admin_share_access','f_service_install','f_sec_observed','f_net_observed','f_event_category','f_weekday'}
BINARY = DISCRETE - {'f_event_category', 'f_weekday'}

def entropy_bits(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def mi_bits(x, y, discrete):
    """Plug-in MI(X;Y) in bits. x: 1-D array, y: binary labels."""
    x = np.asarray(x, float); x = np.where(np.isfinite(x), x, np.nan)
    if np.nanstd(x) == 0 or np.all(np.isnan(x)):
        return 0.0
    if discrete:
        codes, _ = pd.factorize(pd.Series(x).fillna(-999), sort=True)
    else:
        q = np.unique(np.nanquantile(x, np.linspace(0, 1, 33)))
        codes = np.searchsorted(q, np.nan_to_num(x, nan=q[0]), side='right')
    ct = pd.crosstab(codes, y).values.astype(float)
    pxy = ct / ct.sum(); px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])).sum())

def js_bits(a, b, bins=40):
    hi = np.nanquantile(np.concatenate([a, b]), 0.995); lo = float(min(np.nanmin(a), np.nanmin(b)))
    if hi <= lo: return 0.0
    e = np.linspace(lo, hi, bins)
    pa, _ = np.histogram(np.clip(a, lo, hi), bins=e); pb, _ = np.histogram(np.clip(b, lo, hi), bins=e)
    pa = (pa + 1e-9) / (pa.sum() + 1e-9 * bins); pb = (pb + 1e-9) / (pb.sum() + 1e-9 * bins); m = 0.5 * (pa + pb)
    return float((0.5 * rel_entr(pa, m).sum() + 0.5 * rel_entr(pb, m).sum()) / np.log(2))

def cramers_v(feat, label):
    ct = pd.crosstab(feat, label)
    if ct.shape[0] < 2 or ct.shape[1] < 2: return 0.0, 1.0, 'constant'
    chi2, p, _, expected = stats.chi2_contingency(ct)
    n = ct.values.sum(); r, k = ct.shape
    v = float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))
    test = 'chi2'
    if ct.shape == (2, 2) and expected.min() < 5:
        _, p = stats.fisher_exact(ct.values); test = 'fisher'
    return v, float(p), test

def analyze(path, name):
    df = pd.read_parquet(path, columns=FEATURE_COLS + ['label'])
    y = df['label'].values.astype(int); n = len(df); prev = y.mean()
    Hy = entropy_bits([prev, 1 - prev])
    rows = []
    for c in FEATURE_COLS:
        x = df[c].astype(float).values
        b = x[y == 0]; a = x[y == 1]
        rec = {'feature': c, 'n': n, 'benign_mean': float(np.nanmean(b)), 'attack_mean': float(np.nanmean(a)),
               'benign_std': float(np.nanstd(b)), 'attack_std': float(np.nanstd(a)),
               'constant': bool(np.nanstd(x) == 0)}
        rec['mi_bits'] = mi_bits(x, y, c in DISCRETE)
        assert -1e-9 <= rec['mi_bits'] <= Hy + 1e-9, (name, c, rec['mi_bits'], Hy)   # automated MI sanity check
        rec['mi_over_Hy'] = rec['mi_bits'] / Hy
        xf = np.nan_to_num(x, nan=0.0)
        rec['auc_attack_positive'] = float(roc_auc_score(y, xf)) if not rec['constant'] else 0.5
        rec['cliffs_delta'] = 2 * rec['auc_attack_positive'] - 1          # sign: + means attack has larger values
        rec['separability'] = abs(rec['cliffs_delta'])
        if c in DISCRETE:
            v, p, test = cramers_v((df[c] > 0.5) if c in BINARY else df[c].fillna(-1).astype(int), df['label'])
            rec.update(effect_type='CramersV', effect_size=v, p_value=p, test=test)
        else:
            try:
                _, p = stats.mannwhitneyu(a[~np.isnan(a)], b[~np.isnan(b)], alternative='two-sided')
            except Exception:
                p = 1.0
            ks, _ = stats.ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)])
            rec.update(effect_type='CliffsDelta', effect_size=rec['cliffs_delta'], p_value=float(p), test='mannwhitney',
                       ks_stat=float(ks), js_bits=js_bits(a[~np.isnan(a)], b[~np.isnan(b)]))
        rows.append(rec)
    res = pd.DataFrame(rows)
    # Holm correction across all features
    m = len(res); order = np.argsort(res['p_value'].values); adj = np.empty(m)
    running = 0.0
    for rank_i, idx in enumerate(order):
        val = min(1.0, (m - rank_i) * res['p_value'].values[idx]); running = max(running, val); adj[idx] = running
    res['p_holm'] = adj
    res = res.sort_values('mi_bits', ascending=False)
    res.to_csv(ROOT + f'/artifacts/eda_stats_{name}.csv', index=False)
    meta = {'n_rows': int(n), 'attack_prevalence': float(prev), 'label_entropy_bits': Hy,
            'max_mi_bits': float(res['mi_bits'].max()), 'n_features': int(m)}
    json.dump(meta, open(ROOT + f'/artifacts/eda_meta_{name}.json', 'w'), indent=2)
    print(f"\n===== {name}: n={n:,} prevalence={prev:.4f} H(Y)={Hy:.3f} bits; max MI={res['mi_bits'].max():.3f} bits =====")
    print(res[['feature','benign_mean','attack_mean','mi_bits','auc_attack_positive','effect_type','effect_size','test','p_holm']]
          .round(4).to_string(index=False))
    return df, res

if __name__ == '__main__':
    os.makedirs(ROOT + '/artifacts', exist_ok=True)
    dl, rl = analyze(ROOT + '/features/lmd_features.parquet', 'LMD')
    dm, rm = analyze(ROOT + '/features/mordor_features.parquet', 'Mordor')
    corr = dl[FEATURE_COLS].fillna(0).sample(min(300000, len(dl)), random_state=1).corr(method='spearman')
    corr.to_csv(ROOT + '/artifacts/corr_lmd_spearman.csv')
    pairs = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= 0.7: pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], round(float(r), 3)))
    print("\nHIGH-REDUNDANCY PAIRS (|Spearman|>=0.7):", pairs)
    nzv = {nm: [c for c in FEATURE_COLS if d[c].astype(float).std() == 0] for nm, d in [('LMD', dl), ('Mordor', dm)]}
    print("ZERO-VARIANCE:", nzv)
    json.dump({'redundant_pairs': pairs, 'zero_variance': nzv}, open(ROOT + '/artifacts/redundancy.json', 'w'), indent=2)
