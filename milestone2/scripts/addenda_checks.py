"""Three data checks quoted in the report (§1.3, §3.1, Table 12) -> artifacts/addenda.json

1. LMD-2023 broadcast rule: every address ending in .255 that the unicast filter (common_features.address_kinds)
   treats as broadcast, with its row counts, and whether its /24 prefix is one in which a monitored host's own
   interface (SourceIp of an outbound EID 3 row) lives.  Establishes that the ".255" rule is exact for this capture
   (192.168.x.0/24 and 10.0.1.0/24 networks) rather than a general test of broadcast status.
4. LMD-2023 admin-port composition: which destination ports make up the inbound admin-port connections (the
   inbound_adminport_60s signal of 3.3) by class, and the share of unicast inbound connections that target an admin port.
2. Mordor channel qualification: rows whose EventID lies in the Sysmon numeric range (1-26) but that another channel
   logged (System, TerminalServices, ...) - the rows that would be mis-categorised without per-channel interpretation.
3. Mordor modality indicators, read from the FEATURE MATRIX itself (features/mordor_features.parquet, i.e. exactly the
   implemented cummax indicators): per host, the rows and seconds during which sec_observed = 0 (before the first
   auth/service event 4624/4648/4688/4769/5140/5145/7045/4697 on that host) and net_observed = 0 (before the first
   Sysmon EID 3) - the window in which a zero is "not yet observed" rather than "recorded zero".
"""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import json, collections, sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import unicast_mask, ADMIN_PORTS

LMD = ROOT + "/data/LMD-2023 [1.75M Elements]/LMD-2023 [1.75M Elements] Checked/Labelled LMD-2023/LMD-2023 [1.75M Elements][Labelled]checked.csv"
MORDOR_FEATURES = ROOT + "/features/mordor_features.parquet"
MORDOR = [ROOT + "/data/mordor/apt29_evals_day1_manual_2020-05-01225525.json",
          ROOT + "/data/mordor/apt29_evals_day2_manual_2020-05-02035409.json"]
OUT = ROOT + "/artifacts/addenda.json"
SYSMON_RANGE = set(range(1, 27))


def p24(ip):
    a = str(ip).split('.')
    return '.'.join(a[:3]) if len(a) == 4 else None


def lmd_broadcast():
    df = pd.read_csv(LMD, usecols=['EventID', 'Initiated', 'SourceIp', 'DestinationIp', 'DestinationPort', 'Label', 'Computer'], dtype=str, low_memory=False)
    df = df[df.EventID == '3']
    out = df[df.Initiated.str.lower() == 'true']; inb = df[df.Initiated.str.lower() == 'false']
    own_prefixes = sorted({p24(ip) for ip in out.SourceIp.dropna().unique() if p24(ip) and not ip.startswith(('169.254.', '127.')) and ':' not in ip})
    b_out = out[out.DestinationIp.str.endswith('.255', na=False)]; b_in = inb[inb.SourceIp.str.endswith('.255', na=False)]
    addrs = sorted(set(b_out.DestinationIp.unique()) | set(b_in.SourceIp.unique()))
    per = {}
    for a in addrs:
        per[a] = {'outbound_rows': int((b_out.DestinationIp == a).sum()), 'inbound_rows': int((b_in.SourceIp == a).sum()),
                  'kind': 'link-local broadcast (169.254.0.0/16)' if a == '169.254.255.255' else ('/24 of a monitored host' if p24(a) in own_prefixes else 'UNEXPLAINED')}
    # admin-port composition of inbound connections (class-conditional)
    inb = inb.copy(); inb['uni'] = unicast_mask(inb.SourceIp).values; inb['port'] = pd.to_numeric(inb.DestinationPort, errors='coerce'); inb['adm'] = inb.port.isin(ADMIN_PORTS); inb['att'] = (inb.Label != '0')
    u = inb[inb.uni]; a = inb[inb.adm]; att = a[a.att]
    adm = {'admin_ports': sorted(ADMIN_PORTS), 'unicast_inbound_rows': {'benign': int((~u.att).sum()), 'attack': int(u.att.sum())},
           'unicast_inbound_on_admin_port_share': {'all': round(float(u.adm.mean()), 3), 'benign': round(float(u[~u.att].adm.mean()), 3), 'attack': round(float(u[u.att].adm.mean()), 3)},
           'admin_port_inbound_rows': {'benign': int((~a.att).sum()), 'attack': int(a.att.sum())},
           'attack_admin_port_inbound_by_port': {str(int(k)): int(v) for k, v in att.port.value_counts().items()},
           'benign_admin_port_inbound_by_port': {str(int(k)): int(v) for k, v in a[~a.att].port.value_counts().items()},
           'attack_share_ldap_kerberos': round(float(att.port.isin([88, 389, 636, 464]).mean()), 3),
           'attack_share_remote_service_ports': round(float(att.port.isin([445, 139, 135, 3389, 5985, 5986, 47001]).mean()), 3),
           'attack_admin_port_inbound_by_host': {str(k): int(v) for k, v in att.Computer.value_counts().head(4).items()}}
    return {'n_eid3_rows': int(len(df)), 'admin_port': adm, 'own_host_prefixes_24': own_prefixes, 'n_255_addresses': len(addrs), 'addresses': per,
            'n_unexplained': sum(1 for v in per.values() if v['kind'] == 'UNEXPLAINED'),
            'outbound_255_rows': int(len(b_out)), 'outbound_255_attack_rows': int((b_out.Label == '1').sum()),
            'inbound_255_rows': int(len(b_in)), 'inbound_255_attack_rows': int((b_in.Label == '1').sum()),
            'n_private_24_networks': len([p for p in own_prefixes])}


def modality_from_features():
    f = pd.read_parquet(MORDOR_FEATURES, columns=['host', 'ts', 'label', 'f_sec_observed', 'f_net_observed'])
    n = len(f); res = {'n_rows': n, 'sec_observed_ids': [4624, 4648, 4688, 4769, 5140, 5145, 7045, 4697], 'net_observed_ids': [3], 'hosts': {}}
    for h, g in f.groupby('host'):
        t0 = g['ts'].min(); d = {'rows': int(len(g))}
        for c, key in (('f_sec_observed', 'sec'), ('f_net_observed', 'net')):
            z = g[g[c] == 0]
            d[f'rows_{key}_zero'] = int(len(z)); d[f'seconds_{key}_zero'] = float((z['ts'].max() - t0).total_seconds()) if len(z) else 0.0
            d[f'attack_rows_in_{key}_zero'] = int(z['label'].sum())
        res['hosts'][h] = d
    for key in ('sec', 'net'):
        res[f'rows_{key}_zero_total'] = int(sum(v[f'rows_{key}_zero'] for v in res['hosts'].values()))
        res[f'pct_{key}_zero'] = round(100 * res[f'rows_{key}_zero_total'] / n, 2)
        res[f'max_seconds_{key}_zero'] = max(v[f'seconds_{key}_zero'] for v in res['hosts'].values())
        res[f'attack_rows_in_{key}_zero_total'] = int(sum(v[f'attack_rows_in_{key}_zero'] for v in res['hosts'].values()))
    return res


def mordor_checks():
    coll = collections.Counter(); n = 0; total = collections.Counter(); bc = collections.Counter(); own = set()
    for fp in MORDOR:
        for line in open(fp):
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1; ch = d.get('Channel') or ''; eid = d.get('EventID'); host = d.get('Hostname') or d.get('host'); ts = d.get('@timestamp')
            chl = ch.lower(); total[host] += 1
            if chl != 'microsoft-windows-sysmon/operational' and eid in SYSMON_RANGE:
                coll[f"{ch} {eid}"] += 1
            if chl == 'microsoft-windows-sysmon/operational' and eid == 3:
                init = str(d.get('Initiated')).lower() == 'true'; addr = (d.get('DestinationIp') if init else d.get('SourceIp')) or ''
                if init and p24(d.get('SourceIp') or '') and ':' not in str(d.get('SourceIp')) and not str(d.get('SourceIp')).startswith(('169.254.', '127.')): own.add(p24(d.get('SourceIp')))
                if addr.endswith('.255'): bc[addr + (' outbound' if init else ' inbound')] += 1
    chan = collections.Counter()
    for k, v in coll.items(): chan[k.rsplit(' ', 1)[0]] += v
    b255 = sorted({k.split(' ')[0] for k in bc}); own = sorted(own)
    broadcast = {'own_host_prefixes_24': own, 'addresses': dict(bc), 'n_255_addresses': len(b255), 'n_unexplained': sum(1 for a in b255 if p24(a) not in own)}
    return {'n_rows': n, 'broadcast': broadcast, 'sysmon_range_ids_from_other_channels': {'n_rows': int(sum(coll.values())), 'by_channel_and_id': dict(sorted(coll.items(), key=lambda kv: -kv[1])),
                                                                  'by_channel': dict(chan), 'pct_rows': round(100 * sum(coll.values()) / n, 3)},
            'modality_ambiguity': modality_from_features()}


if __name__ == '__main__':
    res = {'lmd_broadcast': lmd_broadcast(), 'mordor': mordor_checks()}
    json.dump(res, open(OUT, 'w'), indent=1)
    lb, mo = res['lmd_broadcast'], res['mordor']
    print(f"LMD: {lb['n_255_addresses']} .255 addresses, unexplained {lb['n_unexplained']}, own /24s {lb['own_host_prefixes_24']}")
    print(f"LMD: outbound .255 rows {lb['outbound_255_rows']} (attack {lb['outbound_255_attack_rows']}), inbound {lb['inbound_255_rows']} (attack {lb['inbound_255_attack_rows']})")
    c = mo['sysmon_range_ids_from_other_channels']; print(f"Mordor: {c['n_rows']} of {mo['n_rows']} rows ({c['pct_rows']}%) carry a Sysmon-range ID from another channel: {c['by_channel']}")
    a = mo['modality_ambiguity']; print(f"Mordor: sec_observed = 0 on {a['rows_sec_zero_total']} rows ({a['pct_sec_zero']}%, max {a['max_seconds_sec_zero']:.0f} s); net_observed = 0 on {a['rows_net_zero_total']} rows ({a['pct_net_zero']}%, max {a['max_seconds_net_zero']:.0f} s)")
