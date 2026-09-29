"""Chapter 4 split audit + Mordor reduced-feature consistency check.

Sampling scheme (both datasets): a 400,000-row sample that keeps EVERY positive row and draws the
remaining rows uniformly at random from the negatives (seed 42). Test prevalence therefore exceeds the
full-data prevalence (LMD 8.06% -> 35.3%; Mordor 3.77% -> 7.4%) and is reported next to PR-AUC,
whose chance baseline equals the prevalence.

Splits (all without the five absolute-time features): random stratified; grouped by host x 10-minute
block; grouped by host x day; chronological 70/30.

Mordor reduced-feature check: additionally removes the whole lineage and command-line families - every
feature computed from the EID-1 fields that the weak-label seeds read (Image, ParentImage, CommandLine), plus
integrity_high (IntegrityLevel, not read by the seeds but in the same family) - and the PID control, so that no
remaining feature shares its inputs with the labelling rule.
"""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import FEATURE_COLS, FAMILY
from ranking import load, RF_KW
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

TIME = FAMILY['temporal'][:5]            # seconds_in_day, diurnal_sin, diurnal_cos, weekday, is_business_hours
NOTIME = [c for c in FEATURE_COLS if c not in TIME]
SEED_FIELD_FEATURES = FAMILY['lineage'] + FAMILY['cmdline']   # whole families: every feature reading Image/ParentImage/CommandLine (+ integrity_high)
REDUCED = [c for c in NOTIME if c not in SEED_FIELD_FEATURES and c != 'f_naive_relational_pid']

def splits_for(df):
    y = df['label'].values; X = df[NOTIME].fillna(0).values; out = {}
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=7); out['random'] = (tr, te)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=7).split(X, y, groups=df['group'].values)); out['grouped_host10min'] = (tr, te)
    gday = (df['host'].astype(str) + '|' + df['ts'].dt.floor('D').astype(str)).values
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=7).split(X, y, groups=gday)); out['grouped_host_day'] = (tr, te)
    order = np.argsort(df['ts'].values); cut = int(0.7 * len(order)); out['chronological_70_30'] = (order[:cut], order[cut:])
    return out

def evaluate(df, cols, tr, te):
    y = df['label'].values; X = df[cols].fillna(0).values
    rec = {'n_train': int(len(tr)), 'n_test': int(len(te)), 'pos_train': int(y[tr].sum()), 'pos_test': int(y[te].sum()),
           'prev_test': round(float(y[te].mean()), 4), 'auc': None, 'prauc': None, 'n_features': len(cols)}
    if 0 < y[te].sum() < len(te) and y[tr].sum() > 0:
        rf = RandomForestClassifier(**RF_KW).fit(X[tr], y[tr]); pr = rf.predict_proba(X[te])[:, 1]
        rec['auc'] = round(float(roc_auc_score(y[te], pr)), 4); rec['prauc'] = round(float(average_precision_score(y[te], pr)), 4)
    return rec

out = {'sampling': 'all positives kept; negatives sampled uniformly to 400,000 rows (seed 42)',
       'features_notime': NOTIME, 'features_reduced_mordor': REDUCED, 'features_removed_reduced': SEED_FIELD_FEATURES + TIME + ['f_naive_relational_pid']}
for name, path in [('LMD', ROOT + '/features/lmd_features.parquet'), ('Mordor', ROOT + '/features/mordor_features.parquet')]:
    df = load(path); sp = splits_for(df); res = {}
    for k, (tr, te) in sp.items():
        res[k] = evaluate(df, NOTIME, tr, te)
    if name == 'Mordor':
        for k in ('grouped_host10min', 'grouped_host_day'):
            tr, te = sp[k]; res[k + '_reduced'] = evaluate(df, REDUCED, tr, te)
    out[name] = res
    print(name); [print('  ', k, v) for k, v in res.items()]
json.dump(out, open(ROOT + '/artifacts/split_audit.json', 'w'), indent=2)
print('reduced features (%d):' % len(REDUCED), REDUCED)
