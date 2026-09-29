"""Part 4 — local LLM arbitration (Ollama) for disputed events.

The context handed to the model is built from the COMMON intermediate schema and the shared feature vector, so it is
dataset-agnostic. Command lines are truncated and base64 blobs masked; only image basenames are shown.
Every call is cached in results/llm_cache.jsonl (resume-safe). The 'mock' backend is a deterministic rule used only to
test the pipeline without a model server; the report must be produced with the real backend.
"""
import json, hashlib, re, time, logging, random
import numpy as np
import pandas as pd
from .common_features import CAT_ORDINAL, _basename, NULLS, APPLICABLE

log = logging.getLogger("m3.llm")
CAT_NAME = {v: k for k, v in CAT_ORDINAL.items()}; CAT_NAME[-1] = 'other'

SYSTEM_PROMPT = """You are a senior SOC analyst arbitrating between two intrusion-detection models.
Task: decide whether ONE Windows host telemetry event is part of LATERAL MOVEMENT (MITRE ATT&CK TA0008: remote
services such as SMB/RPC/WinRM/RDP, exploitation of remote services, pass-the-hash/ticket, lateral tool transfer)
or BENIGN activity.
Rules:
1. Judge only from the evidence given. Do not invent facts. Rates and counts are trailing per-host windows.
2. Lateral movement typically shows: inbound remote-service connections or logons from new peers, a service/WMI/WinRM
   host process spawning an interpreter, tools written to admin shares or staging paths, credential-store access,
   bursts of distinct destinations/admin ports. Broadcast/multicast chatter, routine service processes and
   single connections to a known destination are usually benign.
3. The two model opinions are advisory; they can be wrong. Weigh them against the telemetry.
4. Answer with ONE JSON object and nothing else:
   {"verdict": "LATERAL_MOVEMENT" or "BENIGN", "confidence": number between 0 and 1, "reason": "one sentence"}"""

USER_TEMPLATE = """### Event
{event}

### Host activity in the trailing windows
{window}

### Model opinions
{models}

### Question
Is this event part of lateral movement? Respond with the JSON object only."""

_B64 = re.compile(r'[A-Za-z0-9+/=]{40,}')


_MISSING = {n.lower() for n in NULLS} | {'n/a'}


def _val(x, field=None, eid=None):
    """Field value as a string, or None when the field is absent or outside the event's schema — the same two masks the
    feature code applies (common_features.apply_masks): the LMD-2023 export stores empty fields as '0' / '0.0' and the
    Mordor export as '-' or null, and a field outside its event id's schema (APPLICABLE) is placeholder or
    row-misalignment noise, so neither may reach the prompt as if it were a value."""
    if field in APPLICABLE and eid not in APPLICABLE[field]: return None
    if x is None: return None
    if isinstance(x, float) and np.isnan(x): return None
    s = str(x).strip()
    return None if s.lower() in _MISSING else s


def _clean_cmd(s, n=160):
    s = _B64.sub('<base64-blob>', s); return s[:n] + ('…' if len(s) > n else '')


def build_context(ev, ft, p1, p2, anom_pct, thr1, thr2=None):
    """ev: intermediate-schema row (Series); ft: feature row (Series). Returns the three prompt sections.
    Only recorded fields are listed: a field the export does not carry for this event type is omitted, not shown as 0."""
    cat = CAT_NAME.get(int(ft['f_event_category']), 'other'); eid = int(ev['eventid']) if pd.notna(ev['eventid']) else None
    lines = [f"- event category: {cat} (event id {eid if eid is not None else 'n/a'})",
             f"- host: {ev['host']}", f"- time: {pd.Timestamp(ev['ts']).strftime('%Y-%m-%d %H:%M:%S')} UTC"]
    g = lambda field: _val(ev.get(field), field, eid)
    v = g('image');           lines.append(f"- process image: {_basename(v)}") if v else None
    v = g('source_image');    lines.append(f"- source process: {_basename(v)}") if v else None
    v = g('parent_image');    lines.append(f"- parent image: {_basename(v)}") if v else None
    v = g('cmdline');         lines.append(f"- command line (truncated): {_clean_cmd(v)}") if v else None
    port = g('dest_port')
    if port is not None and float(port) > 0:
        d = 'outbound' if str(ev.get('initiated')).lower() == 'true' else 'inbound'
        lines.append(f"- network connection: {d}, destination port {int(float(port))} ({g('dest_port_name') or 'unknown service'})")
    v = g('target_image');    lines.append(f"- target process / server: {_basename(v)}") if v else None
    v = g('granted_access');  lines.append(f"- granted access mask: {v}") if v else None
    v = g('target_filename'); lines.append(f"- target file: {v[:120]}") if v else None
    v = g('logon_type');      lines.append(f"- logon type: {int(float(v))}") if v else None
    v = g('user');            lines.append(f"- user: {v.split(chr(92))[-1][:40]}") if v else None
    v = g('integrity_level'); lines.append(f"- integrity level: {v}") if v else None
    window = [f"- events on this host in the last 60 s: {int(ft['f_host_evt_rate_60s'])} (process creations {int(ft['f_host_proc_create_60s'])}, network connections {int(ft['f_host_netconn_60s'])})",
              f"- admin-port connections in the last 60 s: {int(ft['f_adminport_conn_60s'])} (inbound {int(ft['f_inbound_adminport_60s'])})",
              f"- distinct outbound destinations in the last 5 min: {int(ft['f_distinct_dstip_300s'])}; distinct inbound unicast peers: {int(ft['f_inbound_peers_300s'])}",
              f"- distinct outbound destinations in the last hour: {int(ft['f_host_out_degree_1h'])}; distinct users in the last 60 s: {int(ft['f_distinct_users_60s'])}",
              f"- flags: remote-exec parent={int(ft['f_remote_exec_parent'])}, remote-exec after inbound admin-port={int(ft['f_remote_exec_after_inbound'])}, LOLBin image={int(ft['f_image_is_lolbin'])}, suspicious command tokens={int(ft['f_cmdline_susp'])}, LSASS access={int(ft['f_lsass_access'])}, credential-read mask={int(ft['f_cred_access_mask'])}, executable dropped={int(ft['f_exec_file_drop'])}, run from staging path={int(ft['f_exec_from_staging'])}, first time this host contacted this destination={int(ft['f_rare_dst_pair'])}"]
    thr2 = float(thr2) if thr2 is not None else None
    return {'event': '\n'.join(lines), 'window': '\n'.join(window), 'models': models_text(p1, p2, anom_pct, thr1, thr2), '_p': (float(p1), float(p2), float(anom_pct), float(thr1), thr2)}


def models_text(p1, p2, anom_pct, thr1, thr2=None):
    """The supervised scores are class-weighted model outputs on a 0-1 scale, not calibrated probabilities, and are
    therefore shown together with each model's own decision threshold."""
    return '\n'.join([f"- gradient-boosting model: lateral-movement score {p1:.2f} on a 0-1 scale (its decision threshold is {thr1:.2f})",
                       f"- random-forest model: lateral-movement score {p2:.2f} on a 0-1 scale" + (f" (its decision threshold is {thr2:.2f})" if thr2 is not None else ""),
                       "- (the two scores are class-weighted model outputs, not calibrated probabilities)",
                       f"- unsupervised anomaly score: more anomalous than {100*anom_pct:.0f}% of this environment's training traffic"])


def render(ctx, ablation='full', rng=None):
    c = dict(ctx)
    if ablation == 'no_models': c['models'] = '(withheld)'
    elif ablation == 'no_window': c['window'] = '(withheld)'
    elif ablation == 'swapped_models':          # adversarial consistency test: the two model probabilities exchange places
        p1, p2, a, t1, t2 = ctx['_p']; c['models'] = models_text(p2, p1, a, t1, t2)
    elif ablation == 'shuffled':
        lines = c['event'].split('\n'); (rng or random.Random(0)).shuffle(lines); c['event'] = '\n'.join(lines)
        lines = c['window'].split('\n'); (rng or random.Random(1)).shuffle(lines); c['window'] = '\n'.join(lines)
    c.pop('_p', None); return USER_TEMPLATE.format(**c)


def parse_verdict(text):
    m = re.search(r'\{.*?\}', text, re.S)
    if m:
        try:
            j = json.loads(m.group(0)); v = str(j.get('verdict', '')).upper()
            return (1 if 'LATERAL' in v else 0 if 'BENIGN' in v else None), float(j.get('confidence', 0.5)), str(j.get('reason', ''))[:300], True
        except Exception:
            pass
    t = text.upper()
    if 'LATERAL' in t and 'BENIGN' not in t: return 1, 0.5, text[:300], False
    if 'BENIGN' in t: return 0, 0.5, text[:300], False
    return None, 0.5, text[:300], False


class OllamaClient:
    def __init__(self, url, model, options):
        import requests; self.requests = requests; self.url = url.rstrip('/'); self.model = model; self.options = dict(options)
    def chat(self, system, user):
        r = self.requests.post(f"{self.url}/api/chat", json={'model': self.model, 'stream': False, 'options': self.options, 'format': 'json',
                                                              'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}, timeout=600)
        r.raise_for_status(); j = r.json(); return j['message']['content'], j
    def info(self):
        try:
            r = self.requests.post(f"{self.url}/api/show", json={'model': self.model}, timeout=60); j = r.json()
            return {'model': self.model, 'details': j.get('details', {}), 'parameters': j.get('parameters', '')[:800]}
        except Exception as e:
            return {'model': self.model, 'error': str(e)}


class MockClient:
    """Deterministic stand-in (pipeline tests only): follows the model whose probability is further from 0.5."""
    def chat(self, system, user):
        p = [float(x) for x in re.findall(r'lateral-movement score ([0-9.]+)', user)]
        if len(p) < 2: p = [0.5, 0.5]
        v = 'LATERAL_MOVEMENT' if (p[0] - 0.5 + p[1] - 0.5) > 0 else 'BENIGN'
        return json.dumps({'verdict': v, 'confidence': 0.6, 'reason': 'mock arbiter'}), {'eval_count': 0, 'total_duration': 0}
    def info(self): return {'model': 'mock'}


class Arbiter:
    def __init__(self, client, cache_path, model_name):
        self.client, self.cache_path, self.model_name = client, cache_path, model_name; self.cache = {}
        if cache_path.exists():
            for line in open(cache_path):
                try: r = json.loads(line); self.cache[r['key']] = r
                except Exception: pass

    def ask(self, dataset, row_id, ctx, ablation='full'):
        user = render(ctx, ablation); key = hashlib.sha1(f"{self.model_name}|{dataset}|{row_id}|{ablation}|{user}".encode()).hexdigest()
        if key in self.cache: return self.cache[key]
        t0 = time.time(); text, raw = self.client.chat(SYSTEM_PROMPT, user); v, conf, reason, parsed = parse_verdict(text)
        rec = {'key': key, 'dataset': dataset, 'row_id': int(row_id), 'ablation': ablation, 'verdict': v, 'confidence': conf, 'reason': reason,
               'parsed_json': parsed, 'seconds': round(time.time() - t0, 2), 'output_tokens': raw.get('eval_count'), 'prompt_tokens': raw.get('prompt_eval_count'),
               'prompt_chars': len(user)}
        self.cache[key] = rec
        with open(self.cache_path, 'a') as f: f.write(json.dumps(rec) + '\n')
        return rec
