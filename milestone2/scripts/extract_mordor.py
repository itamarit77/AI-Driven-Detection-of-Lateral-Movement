"""
Parse Mordor APT29 (day1+day2) JSON -> intermediate schema -> features.
Labels are derived by PROCESS-PROVENANCE tracing (a standard method for
adversary-emulation captures that ship without turnkey per-event labels):
  1. Build the process tree from Sysmon EID1 (ProcessGuid -> ParentProcessGuid).
  2. Seed the malicious set with processes matching published LOLBin-abuse /
     remote-execution detection heuristics (defensive IOCs).
  3. Propagate malicious to all descendants (fixpoint over the tree).
  4. Label every event whose ProcessGuid (or, for EID 8/10, SourceProcessGUID) is in
     the malicious lineage. No authentication-event rule is used.
This yields per-event weak labels grounded in execution lineage; features that read
the seed fields (Image, ParentImage, CommandLine) are circular with them (report 3.1.1).
"""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json, time, re
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import build_features, FEATURE_COLS, SUSP_TOKENS, SUSP_PATTERNS, _basename

FILES = [
    ROOT + "/data/mordor/apt29_evals_day1_manual_2020-05-01225525.json",
    ROOT + "/data/mordor/apt29_evals_day2_manual_2020-05-02035409.json",
]
OUT = ROOT + "/features/mordor_features.parquet"

# High-confidence attack seeds (defensive detection heuristics for this emulation)
SEED_IMAGES = {
    'sdclt.exe', 'psexec.exe', 'psexec64.exe', 'paexec.exe', 'm.exe', 'mimikatz.exe',
    'rar.exe', 'seaduke', 'python.exe', 'wsmprovhost.exe', 'mshta.exe',
}
SEED_TOKENS = [t for t in SUSP_TOKENS if t not in ('\\\\',)] + [
    'monkey.png', 'draft.zip', 'working.zip', 'sysinternalssuite', 'accesschk', 'hostui', '-verb runas',
    'invoke-', 'get-keystrokes', 'get-privatekeys', 'seaduke', 'kerberos::',
    'lsadump', 'sekurlsa', 'golden', '/inject', 'krbtgt', 'psexesvc',
    'timestomp', 'wmidump', 'reflectivepe',
]
def is_seed(image, cmd, parent):
    b = _basename(image); pb = _basename(parent)
    if b in SEED_IMAGES: return True
    c = (cmd or '').lower()
    if any(t in c for t in SEED_TOKENS) or any(r.search(c) for r in SUSP_PATTERNS): return True
    # UAC-bypass / remote-exec lineage: interpreter spawned by sdclt/psexesvc/wmiprvse/wsmprovhost
    if pb in {'sdclt.exe', 'psexesvc.exe', 'wmiprvse.exe', 'wsmprovhost.exe'} and \
       b in {'cmd.exe', 'powershell.exe', 'rundll32.exe'}: return True
    return False


def parse():
    rows = []
    proc = {}   # guid -> dict(parent, image, cmd, host, seed)
    t0 = time.time()
    for fp in FILES:
        with open(fp) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ch = d.get('Channel'); eid = d.get('EventID')
                ts = d.get('@timestamp') or d.get('EventTime')
                host = d.get('Hostname') or d.get('host')
                # Event IDs are provider-specific: keep the numeric ID only for the Sysmon and Security channels and for
                # the System-channel service-install event 7045; every other channel (PowerShell, System 1/3/6,
                # TerminalServices 21-25, WMI-Activity, ...) is mapped to -1 so that it cannot collide with Sysmon IDs.
                chl = (ch or '').lower()
                if chl == 'microsoft-windows-sysmon/operational' or chl == 'security' or (chl == 'system' and eid == 7045):
                    eid_q = eid
                else:
                    eid_q = -1
                rec = {
                    'ts': ts, 'host': host, 'eventid': eid_q, 'channel': ch,
                    'image': None, 'parent_image': None, 'cmdline': None,
                    'parent_cmdline': None, 'process_guid': None, 'parent_process_guid': None,
                    'process_id': None, 'exec_process_id': d.get('ExecutionProcessID'),
                    'dest_ip': None, 'src_ip': None, 'dest_port': None, 'dest_port_name': None,
                    'initiated': None, 'is_ipv6': None, 'user': None, 'logon_id': None,
                    'logon_type': None, 'target_filename': None, 'source_image': None,
                    'target_image': None, 'granted_access': None, 'integrity_level': None,
                    'src_guid': None,
                }
                if ch == 'Microsoft-Windows-Sysmon/Operational':
                    if eid == 1:
                        g = d.get('ProcessGuid'); pg = d.get('ParentProcessGuid')
                        rec.update(image=d.get('Image'), parent_image=d.get('ParentImage'),
                                   cmdline=d.get('CommandLine'), parent_cmdline=d.get('ParentCommandLine'),
                                   process_guid=g, parent_process_guid=pg,
                                   process_id=d.get('ProcessId'), integrity_level=d.get('IntegrityLevel'),
                                   user=d.get('User'), logon_id=d.get('LogonId'))
                        proc[g] = {'parent': pg, 'image': d.get('Image'), 'cmd': d.get('CommandLine'),
                                   'host': host, 'seed': is_seed(d.get('Image'), d.get('CommandLine'), d.get('ParentImage'))}
                    elif eid == 3:
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'),
                                   process_id=d.get('ProcessId'), dest_ip=d.get('DestinationIp'), src_ip=d.get('SourceIp'),
                                   dest_port=d.get('DestinationPort'), dest_port_name=d.get('DestinationPortName'),
                                   initiated=str(d.get('Initiated')).lower() == 'true',
                                   is_ipv6=str(d.get('SourceIsIpv6')).lower() == 'true', user=d.get('User'))
                    elif eid == 10:
                        rec.update(source_image=d.get('SourceImage'), target_image=d.get('TargetImage'),
                                   granted_access=d.get('GrantedAccess'), process_guid=d.get('SourceProcessGUID'),
                                   src_guid=d.get('SourceProcessGUID'), process_id=d.get('SourceProcessId'))
                    elif eid == 11:
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'),
                                   process_id=d.get('ProcessId'), target_filename=d.get('TargetFilename'))
                    elif eid in (12, 13, 14, 17, 18, 22, 23, 8, 7, 9, 5, 2):
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'),
                                   process_id=d.get('ProcessId'), target_filename=d.get('TargetFilename'))
                        if eid == 8:  # remote thread
                            rec.update(source_image=d.get('SourceImage'), target_image=d.get('TargetImage'),
                                       src_guid=d.get('SourceProcessGuid'))
                elif ch in ('Security', 'security'):
                    if eid == 4624:
                        rec.update(logon_type=d.get('LogonType'), user=d.get('TargetUserName'),
                                   src_ip=d.get('IpAddress'))
                    elif eid == 4648:
                        rec.update(user=d.get('SubjectUserName'), target_image=d.get('TargetServerName'),
                                   src_ip=d.get('IpAddress'))
                    elif eid == 4688:
                        rec.update(image=d.get('NewProcessName'), parent_image=d.get('ParentProcessName'),
                                   cmdline=d.get('CommandLine'), user=d.get('SubjectUserName'),
                                   process_id=d.get('NewProcessId'))
                    elif eid in (5140, 5145):
                        rec.update(target_filename=(str(d.get('ShareName') or '') + '\\' + str(d.get('RelativeTargetName') or '')),
                                   src_ip=d.get('IpAddress'), user=d.get('SubjectUserName'))
                rows.append(rec)
    print("parsed", len(rows), "events;", len(proc), "processes; t", round(time.time()-t0, 1), flush=True)

    # ---- provenance propagation to fixpoint ----
    mal = {g for g, v in proc.items() if v['seed']}
    print("seed malicious processes:", len(mal), flush=True)
    for _ in range(50):
        added = 0
        for g, v in proc.items():
            if g not in mal and v['parent'] in mal:
                mal.add(g); added += 1
        if added == 0:
            break
    print("malicious processes after propagation:", len(mal), flush=True)

    df = pd.DataFrame(rows)
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce', utc=True).dt.tz_localize(None)
    df = df.dropna(subset=['ts', 'host']).reset_index(drop=True)
    df['eventid'] = pd.to_numeric(df['eventid'], errors='coerce')
    df['dest_port'] = pd.to_numeric(df['dest_port'], errors='coerce')
    df['logon_type'] = pd.to_numeric(df['logon_type'], errors='coerce')

    # ---- event-level label from provenance ----
    guid_mal = df['process_guid'].isin(mal)
    src_mal = df['src_guid'].isin(mal)
    # Labels are PROVENANCE-ONLY: an event is positive iff its ProcessGuid, or its SourceProcessGUID (EID 8/10),
    # belongs to the malicious lineage. No authentication-event rule is used, so Security-log features (4624 type 3, 4648)
    # are not label-defining and can be evaluated against the weak labels.
    df['label'] = (guid_mal | src_mal).astype(int)
    print("event label balance:", df['label'].value_counts().to_dict(), flush=True)
    json.dump({'seed_processes': int(sum(v['seed'] for v in proc.values())), 'malicious_processes': int(len(mal)),
               'hosts_with_malicious': int(len({v['host'] for g, v in proc.items() if g in mal})),
               'attack_rows': int(df['label'].sum()), 'rows': int(len(df)), 'attack_rate': float(df['label'].mean())},
              open(ROOT + '/artifacts/mordor_labels.json', 'w'), indent=2)

    from common_features import address_kinds
    json.dump(address_kinds(df), open(ROOT + '/artifacts/mordor_address_kinds.json', 'w'), indent=2)
    feats = build_features(df, 'Mordor-APT29')
    feats['label_multi'] = feats['label'].values  # binary only for Mordor (same frame, no re-sort in between)
    feats.to_parquet(OUT, index=False)
    print("saved", OUT, feats.shape, flush=True)
    print(feats[FEATURE_COLS].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
    print("attack rate", round(feats['label'].mean(), 5))


if __name__ == '__main__':
    parse()
