"""Chapter 5.3 scaling audit: RobustScaler (median / IQR) fitted on the TRAINING rows of a host x day grouped split
and applied unchanged to the held-out rows, for every continuous feature of the unified schema; plus the
cross-dataset check of applying the LMD-fitted transform to Mordor. Writes artifacts/scaling_audit.json."""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import FAMILY
from sklearn.model_selection import GroupShuffleSplit

CONT = FAMILY['temporal'][5:] + FAMILY['velocity'] + FAMILY['fanout'] + ['f_host_out_degree_1h', 'f_distinct_users_60s', 'f_cmdline_entropy']
CONT = [c for c in CONT if c not in ('f_seconds_in_day', 'f_diurnal_sin', 'f_diurnal_cos', 'f_weekday', 'f_is_business_hours')]

def fit(df, cols):
    med = df[cols].median(); q1 = df[cols].quantile(0.25); q3 = df[cols].quantile(0.75); iqr = (q3 - q1).replace(0, 1.0)
    return med, iqr

out = {'continuous_features': CONT, 'split': 'GroupShuffleSplit by host x day, 30% test, seed 7 (as in split_audit.py)'}
fits = {}
for name, path in [('LMD', ROOT + '/features/lmd_features.parquet'), ('Mordor', ROOT + '/features/mordor_features.parquet')]:
    df = pd.read_parquet(path, columns=['host', 'ts', 'label'] + CONT)
    g = (df['host'].astype(str) + '|' + df['ts'].dt.floor('D').astype(str)).values
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=7).split(df, df['label'], groups=g))
    med, iqr = fit(df.iloc[tr], CONT); fits[name] = (med, iqr)
    rec = {'n_train': int(len(tr)), 'n_test': int(len(te)), 'groups_train': int(len(set(g[tr]))), 'groups_test': int(len(set(g[te]))), 'features': {}}
    for c in CONT:
        te_raw = df[c].iloc[te]; te_scaled = (te_raw - med[c]) / iqr[c]
        rec['features'][c] = {'train_median': float(med[c]), 'train_iqr': float(iqr[c]), 'test_median_raw': float(te_raw.median()),
                              'test_median_scaled': float(te_scaled.median()), 'test_p90_scaled': float(te_scaled.quantile(0.9))}
    out[name] = rec
    print(name, rec['n_train'], rec['n_test'], rec['groups_train'], rec['groups_test'])
# cross-dataset: LMD-fitted transform applied to all Mordor rows (and the reverse) for the headline feature
dm = pd.read_parquet(ROOT + '/features/mordor_features.parquet', columns=['f_host_evt_rate_60s', 'f_distinct_dstip_300s'])
dl = pd.read_parquet(ROOT + '/features/lmd_features.parquet', columns=['f_host_evt_rate_60s', 'f_distinct_dstip_300s'])
cross = {}
for c in ['f_host_evt_rate_60s', 'f_distinct_dstip_300s']:
    ml, il = fits['LMD']; mm, im = fits['Mordor']
    cross[c] = {'mordor_median_under_LMD_fit': float(((dm[c] - ml[c]) / il[c]).median()),
                'lmd_median_under_Mordor_fit': float(((dl[c] - mm[c]) / im[c]).median())}
out['cross_fit'] = cross
print(cross)
json.dump(out, open(ROOT + '/artifacts/scaling_audit.json', 'w'), indent=2)
