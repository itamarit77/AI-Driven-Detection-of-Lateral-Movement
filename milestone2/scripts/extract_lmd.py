import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}
import sys, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import build_features, FEATURE_COLS

RAW = ROOT + "/data/LMD-2023 [1.75M Elements]/LMD-2023 [1.75M Elements] Checked/Labelled LMD-2023/LMD-2023 [1.75M Elements][Labelled]checked.csv"
OUT = ROOT + "/features/lmd_features.parquet"

usecols = ['SystemTime','Label','EventID','Computer','Execution_ProcessID','ProcessId',
           'Image','CommandLine','ParentImage','ParentCommandLine','IntegrityLevel',
           'Protocol','Initiated','SourceIsIpv6','SourceIp','DestinationIp','DestinationPort',
           'DestinationPortName','User','LogonId','TargetFilename','SourceImage',
           'TargetImage','GrantedAccess']

t0 = time.time()
print("reading raw LMD ...", flush=True)
df = pd.read_csv(RAW, usecols=lambda c: c in usecols, low_memory=False,
                 dtype=str, na_values=['-','?',''], keep_default_na=True)
print("rows", len(df), "cols", len(df.columns), "t", round(time.time()-t0,1), flush=True)

norm = pd.DataFrame()
norm['ts'] = pd.to_datetime(df['SystemTime'], errors='coerce')
norm['host'] = df['Computer'].fillna('UNK')
norm['eventid'] = pd.to_numeric(df['EventID'], errors='coerce')
lab = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
norm['label'] = (lab > 0).astype(int)              # binary: 0 normal, 1 = EoRS or EoHT
norm['label_multi'] = lab
norm['image'] = df['Image']
norm['parent_image'] = df['ParentImage']
norm['cmdline'] = df['CommandLine']
norm['parent_cmdline'] = df['ParentCommandLine']
norm['process_id'] = pd.to_numeric(df['ProcessId'], errors='coerce')
norm['exec_process_id'] = pd.to_numeric(df['Execution_ProcessID'], errors='coerce')
norm['dest_ip'] = df['DestinationIp']
norm['src_ip'] = df['SourceIp']
norm['dest_port'] = pd.to_numeric(df['DestinationPort'], errors='coerce')
norm['dest_port_name'] = df['DestinationPortName']
norm['initiated'] = df['Initiated'].astype(str).str.lower().eq('true')
norm['is_ipv6'] = df['SourceIsIpv6'].astype(str).str.lower().eq('true')
norm['user'] = df['User']
norm['logon_id'] = df['LogonId']
norm['target_filename'] = df['TargetFilename']
norm['source_image'] = df['SourceImage']
norm['target_image'] = df['TargetImage']
norm['granted_access'] = df['GrantedAccess']
norm['integrity_level'] = df['IntegrityLevel']

norm = norm.dropna(subset=['ts']).reset_index(drop=True)
print("after ts-parse", len(norm), "t", round(time.time()-t0,1), flush=True)
del df

import json
from common_features import address_kinds
json.dump(address_kinds(norm), open(ROOT + '/artifacts/lmd_address_kinds.json', 'w'), indent=2)
feats = build_features(norm, 'LMD-2023')     # label_multi is carried through build_features (row-aligned after its sort)
print("features built", feats.shape, "t", round(time.time()-t0,1), flush=True)

feats.to_parquet(OUT, index=False)
print("saved", OUT, "t", round(time.time()-t0,1), flush=True)
# quick sanity
print(feats[FEATURE_COLS].describe().T[['mean','std','min','max']].round(3).to_string())
print("label balance", feats['label'].value_counts().to_dict())
print("multi", feats['label_multi'].value_counts().to_dict())
