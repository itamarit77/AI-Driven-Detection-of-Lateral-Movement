"""Label-free arbitration baselines on the escalated rows, for a controlled comparison with the LLM verdicts.
Inputs: results/escalated_rows.json (extract_escalated.py) and results/llm_<ds>.json. Usage: python baselines.py [results_dir]
Every rule below uses only what the LLM also saw (the two model scores with their thresholds and the anomaly percentile);
none uses labels. Metrics are computed on the rows the LLM answered (parsed verdicts), exactly like Table 14."""
import json, sys, os
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else 'results'


def metr(y, v):
    y = np.asarray(y); v = np.asarray(v); tp = int(((v == 1) & (y == 1)).sum()); fp = int(((v == 1) & (y == 0)).sum()); fn = int(((v == 0) & (y == 1)).sum()); tn = int(((v == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0, 'accuracy': (tp + tn) / len(y) if len(y) else 0.0}


def rules(p1, p2, an, t1, t2):
    """Returns {rule_name: verdict array}."""
    m1 = (p1 >= t1 - 1e-9).astype(int); m2 = (p2 >= t2 - 1e-9).astype(int)
    d1 = p1 - t1; d2 = p2 - t2                         # signed distance from each model's operating point
    out = {'LightGBM verdict': m1, 'Random Forest verdict': m2,
           'more confident model (larger |score − threshold|)': np.where(np.abs(d1) >= np.abs(d2), m1, m2),
           'mean signed margin ≥ 0': ((d1 + d2) >= 0).astype(int),
           'either model positive': ((m1 + m2) >= 1).astype(int), 'both models positive': ((m1 + m2) == 2).astype(int),
           'anomaly percentile ≥ 0.99': (an >= 0.99).astype(int),
           'majority of {LightGBM, RF, anomaly ≥ 0.95}': ((m1 + m2 + (an >= 0.95).astype(int)) >= 2).astype(int),
           'always attack': np.ones_like(m1), 'always benign': np.zeros_like(m1)}
    return out


ex = json.load(open(os.path.join(R, 'escalated_rows.json')))
for ds, e in ex.items():
    fp = os.path.join(R, f'llm_{ds}.json')
    if not os.path.exists(fp): print(f'[{ds}] no llm file'); continue
    L = json.load(open(fp)); rec = {r['row_id']: r for r in L['records']}
    pos = np.array(e['positions']); rid = np.array(e['row_ids']); fold = np.array(e['fold'])
    p1 = np.array(e['p_lgbm']); p2 = np.array(e['p_rf']); an = np.array(e['anom_pct']); t1 = np.array([e['thr_lgbm'][f] for f in fold]); t2 = np.array([e['thr_rf'][f] for f in fold])
    y = np.array([e['events'][i]['label'] for i in range(len(pos))])
    # consistency with the LLM records
    assert all(int(r) in rec for r in rid), 'escalated rows and LLM records differ'
    assert all(rec[int(r)]['label'] == int(yy) for r, yy in zip(rid, y)), 'label mismatch'
    llm = np.array([rec[int(r)]['verdict'] if rec[int(r)]['verdict'] is not None else -1 for r in rid]); ok = llm >= 0
    topup = np.array([rec[int(r)].get('topup', 0) for r in rid]).astype(bool)
    print(f"\n===== {ds}: {len(pos)} escalated rows ({topup.sum()} top-ups), {ok.sum()} parsed LLM answers, label=attack share {y.mean():.2f}")
    print(f"{'rule':52s} {'n':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'acc':>6s}  FP  FN")
    d2 = lambda a: np.array([float(f"{x:.2f}") for x in a]); d0 = lambda a: np.array([int(f"{100 * x:.0f}") for x in a]) / 100.0   # llm.models_text's formatting
    for label, (q1, q2, qa, u1, u2) in (('full-precision scores', (p1, p2, an, t1, t2)), ('numbers as displayed in the prompt (2 decimals, integer percentile; a score shown at its threshold counts as positive)', (d2(p1), d2(p2), d0(an), d2(t1), d2(t2)))):
        print(f"-- rules on {label}")
        rows = [('LLM verdict', llm[ok])] + [(k, v[ok]) for k, v in rules(q1, q2, qa, u1, u2).items()]
        for name, v in rows:
            m = metr(y[ok], v); print(f"{name:52s} {m['n']:4d} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['accuracy']:6.3f} {m['fp']:3d} {m['fn']:3d}")
    print(f"displayed ties: gradient-boosting score == threshold on {int((np.isclose(d2(p1), d2(t1)) & ok).sum())} rows, random-forest on {int((np.isclose(d2(p2), d2(t2)) & ok).sum())}")
    # where did the LLM side with?
    m1 = (p1 >= t1 - 1e-7).astype(int); m2 = (p2 >= t2 - 1e-7).astype(int)
    agree = {'with LightGBM only': int(((llm == m1) & (llm != m2) & ok).sum()), 'with RF only': int(((llm == m2) & (llm != m1) & ok).sum()), 'with both': int(((llm == m1) & (llm == m2) & ok).sum()), 'against both': int(((llm != m1) & (llm != m2) & ok).sum())}
    print('LLM sided:', agree)
    if (~topup).sum() and topup.sum():
        for name, mask in (('disputed rows', ~topup & ok), ('top-ups', topup & ok)):
            m = metr(y[mask], llm[mask]); print(f"  LLM on {name}: n {m['n']} F1 {m['f1']:.3f} acc {m['accuracy']:.3f}")
