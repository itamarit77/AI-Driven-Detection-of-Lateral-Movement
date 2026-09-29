"""Within-modality control (Chapter 4, Surprise 3): class means and attack-positive AUC of fan-out /
inbound features restricted to network rows (EID 3) vs host rows, LMD-2023, all rows."""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import json, pandas as pd
from sklearn.metrics import roc_auc_score
df = pd.read_parquet(ROOT + '/features/lmd_features.parquet', columns=['label','eventid','f_distinct_dstip_300s','f_host_out_degree_1h','f_inbound_adminport_60s'])
net = df[df.eventid == 3]; host = df[df.eventid != 3]; d = {}
for f in ['f_distinct_dstip_300s','f_host_out_degree_1h','f_inbound_adminport_60s']:
    d[f] = {'net_benign_mean': round(float(net.loc[net.label==0,f].mean()),2), 'net_attack_mean': round(float(net.loc[net.label==1,f].mean()),2),
            'net_auc_attack_pos': round(float(roc_auc_score(net.label, net[f])),4),
            'host_benign_mean': round(float(host.loc[host.label==0,f].mean()),2), 'host_attack_mean': round(float(host.loc[host.label==1,f].mean()),2)}
    print(f, d[f])
json.dump(d, open(ROOT + '/artifacts/confound_diag.json','w'), indent=2)

# ---- Scale shift (Chapter 5.2): medians / 90th percentiles / maxima of three magnitude features on both datasets ----
ss = {}
for name, path in [('LMD', ROOT + '/features/lmd_features.parquet'), ('Mordor', ROOT + '/features/mordor_features.parquet')]:
    x = pd.read_parquet(path, columns=['f_host_evt_rate_60s', 'f_host_out_degree_1h', 'f_distinct_dstip_300s'])
    ss[name] = {c: {'median': float(x[c].median()), 'p90': float(x[c].quantile(0.9)), 'max': float(x[c].max())} for c in x.columns}
    print(name, ss[name])
json.dump(ss, open(ROOT + '/artifacts/scale_shift.json', 'w'), indent=2)
