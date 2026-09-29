"""Stage 1 — dataset adapters.  Each adapter maps ONE raw source into the common intermediate event schema
(ts, host, eventid, label, image, parent_image, cmdline, ..., dest_ip, src_ip, dest_port, initiated, ...).
Everything downstream (features, splits, models, cascade, LLM) reads only this schema and is therefore
100% agnostic to the raw layouts (a 94-column Sysmon CSV vs. multi-channel Windows event JSON).

The logic is the Milestone-2 extraction code with the file paths turned into parameters; labels are identical:
  LMD-2023  : creators' ground truth (Normal / EoRS / EoHT -> binary)
  Mordor    : provenance-only weak labels (seed processes -> descendants -> their events), see Milestone 2 §3.1.1
"""
import json, time, logging
import numpy as np
import pandas as pd
from .common_features import SUSP_TOKENS, SUSP_PATTERNS, _basename, address_kinds

log = logging.getLogger("m3.ingest")

INTERMEDIATE_COLUMNS = ['row_id', 'ts', 'host', 'eventid', 'label', 'label_multi', 'image', 'parent_image', 'cmdline',
                        'parent_cmdline', 'process_id', 'exec_process_id', 'dest_ip', 'src_ip', 'dest_port',
                        'dest_port_name', 'initiated', 'is_ipv6', 'user', 'logon_id', 'logon_type', 'target_filename',
                        'source_image', 'target_image', 'granted_access', 'integrity_level']

# ------------------------------------------------------------------------------------------------- LMD-2023
LMD_USECOLS = ['SystemTime', 'Label', 'EventID', 'Computer', 'Execution_ProcessID', 'ProcessId',
               'Image', 'CommandLine', 'ParentImage', 'ParentCommandLine', 'IntegrityLevel',
               'Protocol', 'Initiated', 'SourceIsIpv6', 'SourceIp', 'DestinationIp', 'DestinationPort',
               'DestinationPortName', 'User', 'LogonId', 'TargetFilename', 'SourceImage',
               'TargetImage', 'GrantedAccess']


def ingest_lmd(csv_path):
    t0 = time.time(); log.info("reading %s", csv_path)
    df = pd.read_csv(csv_path, usecols=lambda c: c in LMD_USECOLS, low_memory=False,
                     dtype=str, na_values=['-', '?', ''], keep_default_na=True)
    norm = pd.DataFrame()
    norm['ts'] = pd.to_datetime(df['SystemTime'], errors='coerce')
    norm['host'] = df['Computer'].fillna('UNK')
    norm['eventid'] = pd.to_numeric(df['EventID'], errors='coerce')
    lab = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
    norm['label'] = (lab > 0).astype(int)              # 0 normal, 1 = EoRS or EoHT
    norm['label_multi'] = lab
    for src, dst in [('Image', 'image'), ('ParentImage', 'parent_image'), ('CommandLine', 'cmdline'),
                     ('ParentCommandLine', 'parent_cmdline'), ('DestinationIp', 'dest_ip'), ('SourceIp', 'src_ip'),
                     ('DestinationPortName', 'dest_port_name'), ('User', 'user'), ('LogonId', 'logon_id'),
                     ('TargetFilename', 'target_filename'), ('SourceImage', 'source_image'),
                     ('TargetImage', 'target_image'), ('GrantedAccess', 'granted_access'),
                     ('IntegrityLevel', 'integrity_level')]:
        norm[dst] = df[src]
    norm['process_id'] = pd.to_numeric(df['ProcessId'], errors='coerce')
    norm['exec_process_id'] = pd.to_numeric(df['Execution_ProcessID'], errors='coerce')
    norm['dest_port'] = pd.to_numeric(df['DestinationPort'], errors='coerce')
    norm['initiated'] = df['Initiated'].astype(str).str.lower().eq('true')
    norm['is_ipv6'] = df['SourceIsIpv6'].astype(str).str.lower().eq('true')
    norm['logon_type'] = np.nan
    norm = norm.dropna(subset=['ts']).reset_index(drop=True)
    norm.insert(0, 'row_id', np.arange(len(norm), dtype=np.int64))
    log.info("LMD rows %d  attack %d  t=%.0fs", len(norm), int(norm['label'].sum()), time.time() - t0)
    return norm[INTERMEDIATE_COLUMNS]


# ------------------------------------------------------------------------------------------------- Mordor APT29
SEED_IMAGES = {'sdclt.exe', 'psexec.exe', 'psexec64.exe', 'paexec.exe', 'm.exe', 'mimikatz.exe',
               'rar.exe', 'seaduke', 'python.exe', 'wsmprovhost.exe', 'mshta.exe'}
SEED_TOKENS = [t for t in SUSP_TOKENS if t not in ('\\\\',)] + [
    'monkey.png', 'draft.zip', 'working.zip', 'sysinternalssuite', 'accesschk', 'hostui', '-verb runas',
    'invoke-', 'get-keystrokes', 'get-privatekeys', 'seaduke', 'kerberos::',
    'lsadump', 'sekurlsa', 'golden', '/inject', 'krbtgt', 'psexesvc',
    'timestomp', 'wmidump', 'reflectivepe']


def is_seed(image, cmd, parent):
    b = _basename(image); pb = _basename(parent)
    if b in SEED_IMAGES:
        return True
    c = (cmd or '').lower()
    if any(t in c for t in SEED_TOKENS) or any(r.search(c) for r in SUSP_PATTERNS):
        return True
    if pb in {'sdclt.exe', 'psexesvc.exe', 'wmiprvse.exe', 'wsmprovhost.exe'} and \
       b in {'cmd.exe', 'powershell.exe', 'rundll32.exe'}:
        return True
    return False


def ingest_mordor(json_paths):
    t0 = time.time(); rows = []; proc = {}
    for fp in json_paths:
        log.info("reading %s", fp)
        with open(fp) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ch = d.get('Channel'); eid = d.get('EventID')
                ts = d.get('@timestamp') or d.get('EventTime'); host = d.get('Hostname') or d.get('host')
                chl = (ch or '').lower()   # provider-specific event numbers: keep only Sysmon / Security / System 7045
                eid_q = eid if (chl == 'microsoft-windows-sysmon/operational' or chl == 'security' or (chl == 'system' and eid == 7045)) else -1
                rec = {'ts': ts, 'host': host, 'eventid': eid_q, 'image': None, 'parent_image': None, 'cmdline': None,
                       'parent_cmdline': None, 'process_guid': None, 'parent_process_guid': None, 'process_id': None,
                       'exec_process_id': d.get('ExecutionProcessID'), 'dest_ip': None, 'src_ip': None, 'dest_port': None,
                       'dest_port_name': None, 'initiated': None, 'is_ipv6': None, 'user': None, 'logon_id': None,
                       'logon_type': None, 'target_filename': None, 'source_image': None, 'target_image': None,
                       'granted_access': None, 'integrity_level': None, 'src_guid': None}
                if ch == 'Microsoft-Windows-Sysmon/Operational':
                    if eid == 1:
                        g = d.get('ProcessGuid'); pg = d.get('ParentProcessGuid')
                        rec.update(image=d.get('Image'), parent_image=d.get('ParentImage'), cmdline=d.get('CommandLine'),
                                   parent_cmdline=d.get('ParentCommandLine'), process_guid=g, parent_process_guid=pg,
                                   process_id=d.get('ProcessId'), integrity_level=d.get('IntegrityLevel'),
                                   user=d.get('User'), logon_id=d.get('LogonId'))
                        proc[g] = {'parent': pg, 'host': host, 'seed': is_seed(d.get('Image'), d.get('CommandLine'), d.get('ParentImage'))}
                    elif eid == 3:
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'), process_id=d.get('ProcessId'),
                                   dest_ip=d.get('DestinationIp'), src_ip=d.get('SourceIp'), dest_port=d.get('DestinationPort'),
                                   dest_port_name=d.get('DestinationPortName'),
                                   initiated=str(d.get('Initiated')).lower() == 'true',
                                   is_ipv6=str(d.get('SourceIsIpv6')).lower() == 'true', user=d.get('User'))
                    elif eid == 10:
                        rec.update(source_image=d.get('SourceImage'), target_image=d.get('TargetImage'),
                                   granted_access=d.get('GrantedAccess'), process_guid=d.get('SourceProcessGUID'),
                                   src_guid=d.get('SourceProcessGUID'), process_id=d.get('SourceProcessId'))
                    elif eid == 11:
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'), process_id=d.get('ProcessId'),
                                   target_filename=d.get('TargetFilename'))
                    elif eid in (12, 13, 14, 17, 18, 22, 23, 8, 7, 9, 5, 2):
                        rec.update(image=d.get('Image'), process_guid=d.get('ProcessGuid'), process_id=d.get('ProcessId'),
                                   target_filename=d.get('TargetFilename'))
                        if eid == 8:
                            rec.update(source_image=d.get('SourceImage'), target_image=d.get('TargetImage'),
                                       src_guid=d.get('SourceProcessGuid'))
                elif chl == 'security':
                    if eid == 4624:
                        rec.update(logon_type=d.get('LogonType'), user=d.get('TargetUserName'), src_ip=d.get('IpAddress'))
                    elif eid == 4648:
                        rec.update(user=d.get('SubjectUserName'), target_image=d.get('TargetServerName'), src_ip=d.get('IpAddress'))
                    elif eid == 4688:
                        rec.update(image=d.get('NewProcessName'), parent_image=d.get('ParentProcessName'),
                                   cmdline=d.get('CommandLine'), user=d.get('SubjectUserName'), process_id=d.get('NewProcessId'))
                    elif eid in (5140, 5145):
                        rec.update(target_filename=(str(d.get('ShareName') or '') + '\\' + str(d.get('RelativeTargetName') or '')),
                                   src_ip=d.get('IpAddress'), user=d.get('SubjectUserName'))
                rows.append(rec)
    mal = {g for g, v in proc.items() if v['seed']}
    for _ in range(50):                                   # propagate to descendants until fixpoint
        added = 0
        for g, v in proc.items():
            if g not in mal and v['parent'] in mal:
                mal.add(g); added += 1
        if added == 0:
            break
    df = pd.DataFrame(rows)
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce', utc=True).dt.tz_localize(None)
    df = df.dropna(subset=['ts', 'host']).reset_index(drop=True)
    for c in ('eventid', 'dest_port', 'logon_type'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['label'] = (df['process_guid'].isin(mal) | df['src_guid'].isin(mal)).astype(int)   # provenance-only
    df['label_multi'] = df['label']
    df.insert(0, 'row_id', np.arange(len(df), dtype=np.int64))
    meta = {'seed_processes': int(sum(v['seed'] for v in proc.values())), 'malicious_processes': len(mal),
            'attack_rows': int(df['label'].sum()), 'rows': int(len(df)), 'attack_rate': float(df['label'].mean())}
    log.info("Mordor %s  t=%.0fs", meta, time.time() - t0)
    return df[INTERMEDIATE_COLUMNS], meta


def dataset_facts(events):
    """Label counts and address kinds used by the report (dataset agnostic)."""
    return {'rows': int(len(events)), 'attack_rows': int(events['label'].sum()), 'attack_rate': float(events['label'].mean()),
            'hosts': int(events['host'].nunique()), 'start': str(events['ts'].min()), 'end': str(events['ts'].max()),
            'address_kinds': address_kinds(events)}
