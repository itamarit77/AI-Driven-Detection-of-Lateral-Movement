"""Consistency checks + compact summary of the LLM-stage outputs (results/llm_<ds>.json, results/llm_cache.jsonl),
used to write the Part-4 interpretation. Usage: python llm_summary.py [results_dir]"""
import json, sys, os, collections
R = sys.argv[1] if len(sys.argv) > 1 else 'results'


def f3(x): return '—' if x is None else f"{x:.3f}"


for ds in ('lmd', 'mordor'):
    fp = os.path.join(R, f'llm_{ds}.json')
    if not os.path.exists(fp): print(f'[{ds}] missing'); continue
    j = json.load(open(fp)); ca = json.load(open(os.path.join(R, f'cascade_{ds}.json')))
    print(f'\n===== {ds}  backend={j.get("backend")} model={j.get("llm_info", {}).get("model")} details={j.get("llm_info", {}).get("details")}')
    patched = 'n_llm_in_cascade' in j and all('topup' in r for r in j['records'])
    print(f'top-up flags present: {patched} | n_llm {j.get("n_llm")} in_cascade {j.get("n_llm_in_cascade")} topup {j.get("n_llm_topup")} | unparsed {j.get("n_unparsed")} | '
          f'{j.get("mean_seconds_per_call")} s/call, {j.get("mean_output_tokens")} out tokens')
    print(f'cascade: disputed {ca["n_disputed"]}, escalated {len(ca["escalated_positions"])}, topups {len(ca.get("near_threshold_topup_positions", []))}, sens subset {len(ca["sensitivity_positions"])}')
    s2, s3, orc = j['stage2'], j.get('stage3'), j['stage3_oracle_upper_bound']
    if s3:
        print(f'S2 F1 {f3(s2["f1"])} FP {s2["fp"]} FN {s2["fn"]} -> S3 F1 {f3(s3["f1"])} FP {s3["fp"]} FN {s3["fn"]} | oracle F1 {f3(orc["f1"])} | S3<=oracle: {s3["f1"] <= orc["f1"] + 1e-12}')
        if ca['n_disputed'] == 0: print(f'   LMD-type dataset: S3 == S2: {s3 == s2}')
    for k in ('stage1_on_disputed', 'rf_on_disputed', 'llm_on_disputed'):
        m = j.get(k)
        if m: print(f'{k:20s} n {m["n"]} pos {m["pos"]} | TP {m["tp"]} FP {m["fp"]} FN {m["fn"]} TN {m["tn"]} | P {f3(m["precision"])} R {f3(m["recall"])} F1 {f3(m["f1"])} acc {f3(m["accuracy"])}')
    # agreement structure on the escalated rows
    rec = j['records']; c = collections.Counter()
    for r in rec:
        if r['verdict'] is None: c['unparsed'] += 1; continue
        c[('follows LGBM' if r['verdict'] == r['stage1'] else 'against LGBM') + ' / ' + ('follows RF' if r['verdict'] == r['rf'] else 'against RF') + ' / ' + ('correct' if r['verdict'] == r['label'] else 'wrong')] += 1
    print('verdict structure:', dict(c))
    conf = [r['confidence'] for r in rec if r['verdict'] is not None]
    if conf: print(f'confidence: mean {sum(conf)/len(conf):.2f} min {min(conf)} max {max(conf)}; verdict=attack share {sum(r["verdict"]==1 for r in rec if r["verdict"] is not None)/len(conf):.2f}; label=attack share {sum(r["label"] for r in rec)/len(rec):.2f}')
    if 'sensitivity' in j:
        for abl, s in j['sensitivity'].items():
            m = s.get('metrics') or {}
            print(f'  sens {abl:15s} n {s["n"]} parsed {s["n_parsed"]} acc {f3(s["accuracy"])} flip {f3(s["flip_rate_vs_full"])} F1 {f3(m.get("f1"))} | attack verdicts {m.get("tp", 0) + m.get("fp", 0) if m else "—"}')
    reasons = collections.Counter(r['reason'][:70] for r in rec if r.get('reason'))
    print('top reasons:'); [print('   ', n, '×', t) for t, n in reasons.most_common(5)]

cache = os.path.join(R, 'llm_cache.jsonl')
if os.path.exists(cache):
    rows = [json.loads(l) for l in open(cache) if l.strip()]
    by = collections.Counter((r['dataset'], r['ablation']) for r in rows)
    secs = [r['seconds'] for r in rows if r.get('seconds')]; toks = [r['output_tokens'] for r in rows if r.get('output_tokens')]; ptoks = [r['prompt_tokens'] for r in rows if r.get('prompt_tokens')]
    print(f'\ncache: {len(rows)} calls {dict(by)}; mean {sum(secs)/len(secs) if secs else 0:.1f} s/call (max {max(secs) if secs else 0}); mean output tokens {sum(toks)/len(toks) if toks else 0:.0f}; mean prompt tokens {sum(ptoks)/len(ptoks) if ptoks else 0:.0f}; parsed_json share {sum(r["parsed_json"] for r in rows)/len(rows):.3f}')
