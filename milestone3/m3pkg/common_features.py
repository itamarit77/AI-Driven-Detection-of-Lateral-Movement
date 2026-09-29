"""
[Milestone 3 copy of the Milestone 2 module: identical feature definitions; only a row_id passthrough was added.]

Shared feature-engineering for LMD-2023 and Mordor APT29.
Defensive intrusion-detection feature engineering for Lateral Movement (MITRE TA0008).

Design: raw source events are normalised into ONE intermediate schema, then a single
engineering function derives the candidate feature pool. This guarantees that the
"unified feature schema" (Chapter 5) is computed identically on both datasets.
"""
import re
import numpy as np
import pandas as pd
import math, re

# ----------------------------------------------------------------------------
# Detection reference sets (defensive signatures)
# ----------------------------------------------------------------------------
ADMIN_PORTS = {445, 139, 135, 3389, 5985, 5986, 88, 389, 636, 5722, 464, 47001}
PORT_CATEGORY = {
    445: 'SMB', 139: 'SMB', 135: 'RPC', 3389: 'RDP', 5985: 'WinRM', 5986: 'WinRM',
    88: 'Kerberos', 464: 'Kerberos', 389: 'LDAP', 636: 'LDAP', 5722: 'RPC',
    80: 'Web', 443: 'Web', 53: 'DNS', 123: 'NTP', 137: 'NetBIOS', 138: 'NetBIOS',
}
# Host processes that legitimately host REMOTE execution (parents in lateral-movement lineage)
REMOTE_EXEC_HOSTS = {
    'services.exe', 'wmiprvse.exe', 'wsmprovhost.exe', 'mmc.exe',
    'psexesvc.exe', 'winrshost.exe', 'wininit.exe', 'taskeng.exe', 'schtasks.exe',
}
# Interpreters / tools frequently spawned during remote execution & tool transfer
CHILD_INTERPRETERS = {
    'cmd.exe', 'powershell.exe', 'powershell_ise.exe', 'pwsh.exe', 'wscript.exe',
    'cscript.exe', 'rundll32.exe', 'regsvr32.exe', 'mshta.exe', 'wmic.exe',
    'net.exe', 'net1.exe', 'sc.exe', 'at.exe', 'bitsadmin.exe', 'certutil.exe',
    'reg.exe', 'installutil.exe', 'msbuild.exe',
}
LOLBINS = CHILD_INTERPRETERS | {
    'psexec.exe', 'paexec.exe', 'psexec64.exe', 'wsmprovhost.exe', 'winrs.exe',
    'vssadmin.exe', 'ntdsutil.exe', 'esentutl.exe', 'dsquery.exe',
}
STAGING_PATH_RE = re.compile(
    r'\\(?:windows\\temp|temp|tmp|users\\public|perflogs|appdata|programdata|'
    r'downloads|\$recycle\.bin|windows\\debug|windows\\tasks)\\', re.IGNORECASE)
EXEC_FILE_RE = re.compile(r'\.(exe|dll|bat|ps1|vbs|scr|cmd|hta|jar|psm1|com)$', re.IGNORECASE)
# Command-line obfuscation / remote-exec indicators (defensive)
SUSP_TOKENS = [   # substring tokens: PowerShell download/encode/hidden-window idioms and admin-share paths
    '-enc', '-encodedcommand', '-e ', 'frombase64string', 'downloadstring', 'downloadfile',
    'invoke-expression', '-w hidden', '-windowstyle hidden', '-ep bypass', '-exec bypass',
    '-executionpolicy bypass', 'net.webclient', 'invoke-webrequest', 'reflection.assembly', '-decode',
    '\\\\', 'admin$', 'c$', 'ipc$',
]
# 'iex' only as a stand-alone token (word boundary), so that iexplore.exe does not match
SUSP_PATTERNS = [re.compile(r"""(^|[\s|;('"])iex($|[\s;)('"])""")]
def _susp(s):
    return int(any(t in s for t in SUSP_TOKENS) or any(r.search(s) for r in SUSP_PATTERNS))
# LSASS credential-access GrantedAccess masks (Sysmon EID10) — EoHT signature
CRED_ACCESS_MASKS = {'0x1010', '0x1410', '0x1438', '0x143a', '0x1fffff', '0x1f1fff', '0x1fffff'}

EVENT_CATEGORY = {  # semantic grouping of Sysmon/Windows event ids
    1: 'process_create', 3: 'network_conn', 5: 'process_end', 7: 'image_load',
    8: 'remote_thread', 9: 'raw_access', 10: 'process_access', 11: 'file_create',
    12: 'registry', 13: 'registry', 14: 'registry', 15: 'file_stream',
    17: 'pipe', 18: 'pipe', 22: 'dns', 23: 'file_delete', 26: 'file_delete',
    6: 'driver_load', 4: 'sysmon_state', 16: 'sysmon_config', 25: 'process_tamper',
    4624: 'logon', 4625: 'logon_fail', 4634: 'logoff', 4648: 'explicit_cred',
    4688: 'process_create', 4697: 'service_install', 5140: 'share_access',
    5145: 'share_access', 7045: 'service_install', 4769: 'kerberos_tgs', 4672: 'special_priv',
}
CAT_ORDINAL = {c: i for i, c in enumerate(sorted(set(EVENT_CATEGORY.values())))}


def shannon_entropy(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


NULLS = {'', '0', '0.0', '-', 'nan', 'None', 'null'}

def _basename(img):
    if not isinstance(img, str) or img in NULLS:
        return ''
    return img.replace('/', '\\').split('\\')[-1].strip().lower()

# Field applicability by Sysmon / Security event id. Fields outside their event's
# schema are placeholders ("0" in LMD-2023) or row-misalignment noise and must be masked.
def unicast_mask(ipseries):
    """True for routable unicast peer addresses: excludes empty/placeholder values, IPv4 broadcast (last octet 255 or
    255.255.255.255), multicast (224-239.x.x.x, ff00::/8), loopback (127.x, ::1) and unspecified (0.0.0.0, ::).
    Broadcast/multicast chatter (NetBIOS, SSDP, mDNS) is not a peer that can be pivoted to."""
    ip = ipseries.fillna('').astype(str).str.strip()
    bad = ip.isin(NULLS) | (ip == '') | ip.str.endswith('.255') | (ip == '255.255.255.255') | (ip == '0.0.0.0') | (ip == '::') \
          | ip.str.startswith('127.') | (ip == '::1') | ip.str.lower().str.startswith('ff') \
          | ip.str.match(r'^(22[4-9]|23[0-9])\.')
    return ~bad

def address_kinds(df):
    """Share of outbound DestinationIp / inbound SourceIp (+ Security client IpAddress) by address kind and class,
    for the capture characterisation in Chapter 3 (written by the extract scripts to artifacts/*_address_kinds.json)."""
    def kind(ipseries):
        ip = ipseries.fillna('').astype(str).str.strip(); k = pd.Series('unicast', index=ip.index)
        k[ip.str.match(r'^(22[4-9]|23[0-9])\.') | ip.str.lower().str.startswith('ff')] = 'multicast'
        k[ip.str.endswith('.255') | (ip == '255.255.255.255')] = 'broadcast'
        k[ip.str.startswith('127.') | (ip == '::1')] = 'loopback'
        k[(ip == '0.0.0.0') | (ip == '::')] = 'unspecified'
        k[ip.isin(NULLS) | (ip == '')] = 'null'
        return k
    is_net = df['eventid'] == 3; init = df.get('initiated', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    sec = df['eventid'].isin([4624, 4648, 5140, 5145]); out = {}
    for name, mask, col in [('outbound_dest', is_net & init, 'dest_ip'), ('inbound_src', is_net & ~init, 'src_ip'), ('security_client', sec, 'src_ip')]:
        if col not in df.columns or mask.sum() == 0: continue
        sub = df.loc[mask]; kk = kind(sub[col]); out[name] = {}
        for lab in (0, 1):
            m = sub['label'] == lab
            if m.sum(): out[name]['attack' if lab else 'benign'] = {k: round(float(v), 4) for k, v in kk[m].value_counts(normalize=True).items()}
            out[name]['n_' + ('attack' if lab else 'benign')] = int(m.sum())
    return out

APPLICABLE = {
    'parent_image':   {1, 4688}, 'cmdline': {1, 4688}, 'parent_cmdline': {1, 4688},
    'integrity_level': {1},
    'dest_ip':        {3}, 'src_ip': {3, 4624, 4648, 5140, 5145}, 'dest_port': {3}, 'dest_port_name': {3},
    'initiated':      {3}, 'is_ipv6': {3},
    'target_image':   {10, 8, 4648}, 'granted_access': {10},
    'target_filename': {11, 23, 26, 2, 15, 5140, 5145},
}

def apply_masks(df):
    """Null-out placeholder strings and out-of-schema fields (returns a copy)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].where(~df[c].isin(NULLS))
    for col, eids in APPLICABLE.items():
        if col in df.columns:
            ok = df['eventid'].isin(eids)
            if df[col].dtype == bool or col in ('initiated', 'is_ipv6'):
                df.loc[~ok, col] = False
            else:
                df.loc[~ok, col] = np.nan
    return df


def build_features(df, source_name):
    """
    df: normalised intermediate frame with columns (any missing are tolerated):
      ts(datetime64), host(str), eventid(int), label(int 0/1),
      image, parent_image, cmdline, parent_cmdline, process_id, exec_process_id,
      dest_ip (Sysmon EID 3 DestinationIp), src_ip (EID 3 SourceIp; Security IpAddress = remote client),
      dest_port(float), dest_port_name, initiated(bool), is_ipv6(bool),
      user, logon_id, logon_type(float), target_filename, source_image,
      target_image, granted_access, integrity_level
    Returns a feature DataFrame (one row per event) + the label column.
    """
    df = apply_masks(df)
    df = df.sort_values(['host', 'ts']).reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out['label'] = df['label'].astype(int)
    out['eventid'] = df['eventid']
    out['host'] = df['host']
    out['ts'] = df['ts']

    # ---- image basenames ----
    img = df.get('image', pd.Series([''] * n)).map(_basename)
    pimg = df.get('parent_image', pd.Series([''] * n)).map(_basename)
    timg = df.get('target_image', pd.Series([''] * n)).map(_basename)
    cmd = df.get('cmdline', pd.Series([''] * n)).fillna('').astype(str)

    # ---- temporal ----
    secs = df['ts'].dt.hour * 3600 + df['ts'].dt.minute * 60 + df['ts'].dt.second
    out['f_seconds_in_day'] = secs.astype(float)
    out['f_diurnal_sin'] = np.sin(2 * np.pi * secs / 86400.0)
    out['f_diurnal_cos'] = np.cos(2 * np.pi * secs / 86400.0)
    out['f_weekday'] = df['ts'].dt.dayofweek.astype(float)          # kept to DEMONSTRATE leakage
    out['f_is_business_hours'] = ((df['ts'].dt.hour >= 8) & (df['ts'].dt.hour <= 18)).astype(int)

    # inter-event delta per host
    dt = df.groupby('host')['ts'].diff().dt.total_seconds().fillna(0.0)
    out['f_inter_event_dt_log'] = np.log1p(dt.clip(lower=0))

    # ---- rolling per-host rates (velocity / fan-out) ----
    df_idx = df.set_index('ts')
    def roll_count(mask, window):
        s = pd.Series(np.where(mask.values, 1.0, 0.0), index=df_idx.index)
        r = s.groupby(df['host'].values).rolling(window).sum().reset_index(level=0, drop=True)
        return r.values
    ones = pd.Series(True, index=df.index)
    out['f_host_evt_rate_60s'] = roll_count(ones, '60s')
    out['f_host_proc_create_60s'] = roll_count(df['eventid'] == 1, '60s')
    out['f_host_netconn_60s'] = roll_count(df['eventid'] == 3, '60s')

    # fan-out: distinct destinations in rolling 300s (approx via nunique on window)
    def roll_nunique(valseries, window):
        tmp = pd.DataFrame({'ts': df['ts'].values, 'host': df['host'].values, 'v': valseries.values})
        res = np.zeros(n)
        for h, g in tmp.groupby('host'):
            gi = g.set_index('ts')['v']
            # rolling distinct count via expanding dict over window using a simple two-pointer
            vals = gi.values.astype(object); times = gi.index.values.astype('datetime64[ns]')
            from collections import defaultdict
            cnt = defaultdict(int); left = 0; wnd = np.timedelta64(window, 's')
            outv = np.zeros(len(vals))
            for right in range(len(vals)):
                v = vals[right]
                if v is not None and v == v and v != 0 and v not in NULLS:
                    cnt[v] += 1
                while times[right] - times[left] > wnd:
                    lv = vals[left]
                    if lv is not None and lv == lv and lv != 0 and lv not in NULLS:
                        cnt[lv] -= 1
                        if cnt[lv] <= 0:
                            cnt.pop(lv, None)
                    left += 1
                outv[right] = len(cnt)
            res[g.index.values] = outv
        return res

    dport = df.get('dest_port', pd.Series([np.nan] * n))
    init_b = df.get('initiated', pd.Series([False] * n)).fillna(False).astype(bool)
    is_net = (df['eventid'] == 3)
    # Direction-aware peer addresses. Sysmon records an inbound connection with DestinationIp = the monitored
    # host itself, so DestinationIp is a remote destination only when Initiated = true; the remote peer of an
    # inbound connection is SourceIp, and of a Security logon/share event the client IpAddress (src_ip column).
    dip_all = df.get('dest_ip', pd.Series([''] * n)).fillna('').astype(str)
    sip_all = df.get('src_ip', pd.Series([''] * n)).fillna('').astype(str)
    out_ip = dip_all.where(is_net & init_b & unicast_mask(dip_all), '')                # outbound UNICAST destination
    in_peer = sip_all.where(((is_net & ~init_b) | df['eventid'].isin([4624, 4648, 5140, 5145])) & unicast_mask(sip_all), '')   # inbound unicast remote peer
    dip = out_ip
    out['f_distinct_dstport_300s'] = roll_nunique(dport.where(is_net & init_b).fillna(0), 300)   # outbound destination ports
    out['f_distinct_dstip_300s'] = roll_nunique(out_ip, 300)                            # outbound fan-out
    out['f_inbound_peers_300s'] = roll_nunique(in_peer, 300)                            # inbound peer diversity (destination side)
    out['f_adminport_conn_60s'] = roll_count(is_net & dport.isin(ADMIN_PORTS), '60s')  # all admin-port connections
    out['f_inbound_adminport_60s'] = roll_count(is_net & dport.isin(ADMIN_PORTS) & ~init_b, '60s')   # inbound subset

    # ---- process lineage / execution (real lineage, not PID==ExecPID) ----
    out['f_lolbin_parent_child'] = (pimg.isin(REMOTE_EXEC_HOSTS) & img.isin(CHILD_INTERPRETERS)).astype(int)
    out['f_remote_exec_parent'] = pimg.isin(REMOTE_EXEC_HOSTS).astype(int)
    # destination-side pivot correlation: a remote-exec-hosted child spawned within 60 s of an INBOUND admin-port connection on the same host
    out['f_remote_exec_after_inbound'] = ((out['f_remote_exec_parent'] == 1) & (out['f_inbound_adminport_60s'] > 0)).astype(int)
    out['f_image_is_lolbin'] = img.isin(LOLBINS).astype(int)
    image_full = df.get('image', pd.Series([''] * n)).fillna('').astype(str)
    out['f_exec_from_staging'] = image_full.str.contains(STAGING_PATH_RE).astype(int)

    # command-line structure
    out['f_cmdline_len'] = cmd.str.len().astype(float)
    out['f_cmdline_entropy'] = cmd.map(shannon_entropy)
    low = cmd.str.lower()
    out['f_cmdline_susp'] = low.map(_susp)

    # integrity
    il = df.get('integrity_level', pd.Series([''] * n)).fillna('').astype(str).str.lower()
    out['f_integrity_high'] = il.isin(['high', 'system']).astype(int)

    # ---- network / remote services ----
    out['f_admin_port_flag'] = dport.isin(ADMIN_PORTS).astype(int)
    out['f_initiated'] = df.get('initiated', pd.Series([False] * n)).fillna(False).astype(int)
    out['f_is_ipv6'] = df.get('is_ipv6', pd.Series([False] * n)).fillna(False).astype(int)
    dpn = df.get('dest_port_name', pd.Series([''] * n)).fillna('').astype(str)
    def port_cat(row_port, row_name):
        try:
            p = int(float(row_port))
        except Exception:
            p = 0
        if p in PORT_CATEGORY:
            return PORT_CATEGORY[p]
        nm = str(row_name).lower()
        if 'ds' in nm or 'smb' in nm or 'netbios' in nm: return 'SMB'
        if 'wbt' in nm or 'rdp' in nm: return 'RDP'
        if 'epmap' in nm or 'rpc' in nm: return 'RPC'
        if 'kerb' in nm: return 'Kerberos'
        if 'ldap' in nm or 'gc' in nm: return 'LDAP'
        if 'winrm' in nm or 'wsman' in nm: return 'WinRM'
        return 'Other'
    cats = [port_cat(p, nm) for p, nm in zip(dport.values, dpn.values)]
    out['f_dstport_category'] = cats

    # ---- credential access (EoHT signature; template example) ----
    ga = df.get('granted_access', pd.Series([''] * n)).fillna('').astype(str).str.lower()
    out['f_lsass_access'] = ((df['eventid'] == 10) & (timg == 'lsass.exe')).astype(int)
    out['f_cred_access_mask'] = ((df['eventid'] == 10) & ga.isin([m.lower() for m in CRED_ACCESS_MASKS])).astype(int)

    # ---- file / tool transfer (T1570) ----
    tf = df.get('target_filename', pd.Series([''] * n)).fillna('').astype(str)
    out['f_exec_file_drop'] = ((df['eventid'] == 11) & tf.map(lambda s: bool(EXEC_FILE_RE.search(s)))).astype(int)
    out['f_file_drop_staging'] = ((df['eventid'] == 11) & tf.map(lambda s: bool(STAGING_PATH_RE.search(s)))).astype(int)

    # ---- graph topology / rare pairing (host out-degree & novel pairings) ----
    out['f_host_out_degree_1h'] = roll_nunique(out_ip, 3600)       # distinct OUTBOUND destinations per source host in 1 h
    pair_first = df.assign(_dip=out_ip.values).groupby(['host', '_dip']).cumcount()
    out['f_rare_dst_pair'] = ((pair_first == 0) & (out_ip != '') & ~out_ip.isin(NULLS)).astype(int)   # first (host -> destination) edge

    # ---- user context ----
    usr = df.get('user', pd.Series([''] * n)).fillna('').astype(str)
    out['f_distinct_users_60s'] = roll_nunique(usr, 60)

    # ---- authentication & service-control (Windows Security auditing; populated where collected) ----
    lt = pd.to_numeric(df.get('logon_type', pd.Series([np.nan] * n)), errors='coerce')
    out['f_explicit_cred_4648'] = (df['eventid'] == 4648).astype(int)                 # T1550 alt-cred use
    out['f_network_logon_t3'] = ((df['eventid'] == 4624) & (lt == 3)).astype(int)     # SMB/RPC network logon
    out['f_kerberos_tgs_4769'] = (df['eventid'] == 4769).astype(int)                  # ticket request (PtT)
    out['f_admin_share_access'] = ((df['eventid'].isin([5140, 5145])) &
                                   tf.str.lower().str.contains(r'(?:admin\$|c\$|ipc\$)', regex=True, na=False)).astype(int)
    out['f_service_install'] = (df['eventid'].isin([7045, 4697])).astype(int)         # System 7045 / Security 4697 (NOT Sysmon 6 = driver load)

    # ---- telemetry-modality observed indicators (per host) ----
    # causal ("observed so far"): 1 from the first Security-auditing / EID 3 event seen on the host, never from later rows
    sec_evt = df['eventid'].isin([4624, 4648, 4688, 4769, 5140, 5145, 7045, 4697])
    out['f_sec_observed'] = sec_evt.groupby(df['host']).cummax().astype(int).values
    out['f_net_observed'] = is_net.groupby(df['host']).cummax().astype(int).values

    # ---- event semantics ----
    out['f_event_category'] = df['eventid'].map(lambda e: CAT_ORDINAL.get(EVENT_CATEGORY.get(int(e) if pd.notna(e) else -1, 'other'), -1))

    # ---- naive baselines kept to CRITIQUE (Chapters 2-4) ----
    pid = pd.to_numeric(df.get('process_id', pd.Series([np.nan] * n)), errors='coerce')
    epid = pd.to_numeric(df.get('exec_process_id', pd.Series([np.nan] * n)), errors='coerce')
    # NOTE: deliberately the raw definition used previously (ProcessId == ExecutionProcessID, fills included)
    # so that Chapter 4 can show it is a null-fill artefact of EID 10 rows, not a behavioural signal.
    out['f_naive_relational_pid'] = ((pid == epid) & pid.notna()).astype(int)

    out['source'] = source_name
    if 'row_id' in df.columns:            # M3 passthrough: stable key back to the intermediate event table
        out['row_id'] = df['row_id'].values
    return out


FEATURE_COLS = [
    'f_seconds_in_day', 'f_diurnal_sin', 'f_diurnal_cos', 'f_weekday', 'f_is_business_hours',
    'f_inter_event_dt_log', 'f_host_evt_rate_60s', 'f_host_proc_create_60s', 'f_host_netconn_60s',
    'f_distinct_dstport_300s', 'f_distinct_dstip_300s', 'f_adminport_conn_60s',
    'f_inbound_adminport_60s', 'f_inbound_peers_300s',
    'f_lolbin_parent_child', 'f_remote_exec_parent', 'f_remote_exec_after_inbound', 'f_image_is_lolbin',
    'f_exec_from_staging', 'f_cmdline_len', 'f_cmdline_entropy', 'f_cmdline_susp',
    'f_integrity_high', 'f_admin_port_flag', 'f_initiated', 'f_is_ipv6',
    'f_lsass_access', 'f_cred_access_mask', 'f_exec_file_drop', 'f_file_drop_staging',
    'f_host_out_degree_1h', 'f_rare_dst_pair', 'f_distinct_users_60s',
    'f_explicit_cred_4648', 'f_network_logon_t3', 'f_kerberos_tgs_4769',
    'f_admin_share_access', 'f_service_install', 'f_sec_observed', 'f_net_observed',
    'f_event_category', 'f_naive_relational_pid',
]
# families (used by the report to derive counts from code, not by hand)
FAMILY = {
 'temporal': ['f_seconds_in_day','f_diurnal_sin','f_diurnal_cos','f_weekday','f_is_business_hours','f_inter_event_dt_log'],
 'velocity': ['f_host_evt_rate_60s','f_host_proc_create_60s','f_host_netconn_60s'],
 'fanout':   ['f_distinct_dstport_300s','f_distinct_dstip_300s','f_inbound_peers_300s','f_adminport_conn_60s','f_inbound_adminport_60s'],
 'graph':    ['f_host_out_degree_1h','f_rare_dst_pair'],
 'user':     ['f_distinct_users_60s'],
 'lineage':  ['f_lolbin_parent_child','f_remote_exec_parent','f_remote_exec_after_inbound','f_image_is_lolbin','f_exec_from_staging','f_integrity_high'],
 'cmdline':  ['f_cmdline_len','f_cmdline_entropy','f_cmdline_susp'],
 'network':  ['f_admin_port_flag','f_initiated','f_is_ipv6'],
 'credential':['f_lsass_access','f_cred_access_mask'],
 'tooltransfer':['f_exec_file_drop','f_file_drop_staging'],
 'auth_service':['f_explicit_cred_4648','f_network_logon_t3','f_kerberos_tgs_4769','f_admin_share_access','f_service_install'],
 'modality': ['f_sec_observed','f_net_observed'],
 'structural':['f_event_category'],
 'control':  ['f_naive_relational_pid'],
}
assert sorted(sum(FAMILY.values(), [])) == sorted(FEATURE_COLS), 'FAMILY must partition FEATURE_COLS'
