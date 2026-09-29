"""Part 3 — multi-stage cascade built from the out-of-fold scores of the row-level models.

  Stage 1  LightGBM (fast; thresholds from the inner hold-out): the verdict is LightGBM's at its F1 threshold theta
           (attack iff p1 >= theta). The band (tau_lo, tau_hi) — precision >= CASCADE_PRECISION_TARGET above tau_hi,
           NPV >= CASCADE_NPV_TARGET below tau_lo on the same inner hold-out, never the test fold — decides only whether
           a second opinion is bought: rows with tau_lo < p1 < tau_hi are AMBIGUOUS, all others keep the Stage-1 verdict
           (theta may lie outside the band).
  Stage 2  Random Forest verdict (its own inner-validation threshold) on the ambiguous rows.  If RF agrees with the
           Stage-1 lean the row is RESOLVED; if they disagree it is DISPUTED and provisionally takes the RF verdict.
           The Isolation-Forest anomaly percentile is carried along as evidence for Stage 3 only (it changes no verdict).
  Stage 3  LLM arbitration on (at most LLM_MAX_ESCALATIONS) disputed rows, most ambiguous first; every other disputed
           row keeps the Stage-2 (RF) verdict.  The LLM sees the telemetry context and BOTH model opinions.
Metrics are reported after each stage so the gain of every stage is visible."""
import numpy as np
from .evaluate import metrics


def stage_decisions(y, p1, p2, anom, bands, thr1, thr2, fold_id):
    """Vectorised stage-1/2 logic. bands: list of (tau_lo, tau_hi) per fold; thr1/thr2: per-fold F1 thresholds."""
    p1 = np.asarray(p1, dtype=np.float64); p2 = np.asarray(p2, dtype=np.float64); fold_id = np.asarray(fold_id); EPS = 1e-7
    valid = (fold_id >= 0) & np.isfinite(p1) & np.isfinite(p2)     # rows without an out-of-fold score (a dropped fold) take no part
    fidx = np.where(valid, fold_id, 0)
    lo = np.array([bands[f][0] for f in fidx]); hi = np.array([bands[f][1] for f in fidx])
    t1 = np.array([thr1[f] for f in fidx]); t2 = np.array([thr2[f] for f in fidx])
    conf_attack = p1 >= hi - EPS; conf_benign = p1 <= lo + EPS; ambiguous = ~(conf_attack | conf_benign) & valid
    lean1 = (p1 >= t1 - EPS).astype(int); rf = (p2 >= t2 - EPS).astype(int)
    stage1 = lean1.copy()                                   # Stage-1-only system = LightGBM at its F1 threshold
    stage2 = lean1.copy(); stage2[ambiguous] = rf[ambiguous]  # ambiguous rows take the RF verdict
    disputed = ambiguous & (rf != lean1)
    return {'ambiguous': ambiguous, 'disputed': disputed, 'stage1': stage1, 'stage2': stage2, 'rf': rf, 'lean1': lean1,
            'anomaly_flag': anom >= 0.99, 'band_lo': lo, 'band_hi': hi, 'thr1': t1, 'thr2': t2, 'valid': valid}


def cascade_report(y, dec, llm_verdicts=None, topup=()):
    """Metrics after each stage. llm_verdicts: dict row_position -> 0/1 for every row the LLM arbitrated (parsed answers).
    topup: positions that were escalated only to exercise the arbiter (near-threshold top-ups, see pick_escalations):
    they appear in the *_on_disputed comparison (LLM vs. the two model verdicts on the escalated rows) but NEVER enter
    the Stage-3 cascade metrics, which only ever replace the Stage-2 verdict of DISPUTED rows."""
    y = np.asarray(y); v = dec.get('valid', np.ones(len(y), dtype=bool))
    out = {'n': int(v.sum()), 'pos': int(y[v].sum()), 'ambiguous_rate': float(dec['ambiguous'][v].mean()), 'disputed_rate': float(dec['disputed'][v].mean()),
           'n_ambiguous': int(dec['ambiguous'].sum()), 'n_disputed': int(dec['disputed'].sum()), 'n_without_oof_score': int((~v).sum())}
    out['stage1'] = metrics(y[v], dec['stage1'][v].astype(float), 0.5)
    out['stage2'] = metrics(y[v], dec['stage2'][v].astype(float), 0.5)
    if llm_verdicts:
        top = set(int(t) for t in topup); s3 = dec['stage2'].copy(); n_in = 0
        for pos, vv in llm_verdicts.items():
            if int(pos) not in top: s3[int(pos)] = vv; n_in += 1
        out['stage3'] = metrics(y[v], s3[v].astype(float), 0.5); out['n_llm'] = len(llm_verdicts)
        out['n_llm_in_cascade'] = n_in; out['n_llm_topup'] = len(llm_verdicts) - n_in
        d = np.array(list(llm_verdicts.keys()), dtype=int); vv = np.array(list(llm_verdicts.values()), dtype=int)
        out['llm_on_disputed'] = metrics(y[d], vv.astype(float), 0.5); out['rf_on_disputed'] = metrics(y[d], dec['rf'][d].astype(float), 0.5)
        out['stage1_on_disputed'] = metrics(y[d], dec['lean1'][d].astype(float), 0.5)
    # what a perfect arbiter would give on the disputed rows (upper bound for Stage 3)
    s_or = dec['stage2'].copy(); s_or[dec['disputed']] = y[dec['disputed']]; out['stage3_oracle_upper_bound'] = metrics(y[v], s_or[v].astype(float), 0.5)
    return out


def pick_escalations(dec, p1, k, k_min=50):
    """Up to k DISPUTED rows, most ambiguous first (Stage-1 score closest to its threshold). If fewer than k_min rows
    are disputed (e.g. a dataset the models separate almost perfectly), the escalation set is topped up with the
    most ambiguous non-disputed rows of the band so that the arbiter is still exercised; these are flagged."""
    dist = np.abs(np.asarray(p1, dtype=np.float64) - dec['thr1']); valid = dec.get('valid', np.ones(len(dist), dtype=bool))
    disputed = np.flatnonzero(dec['disputed']); disputed = disputed[np.argsort(dist[disputed])][:k]
    if len(disputed) >= k_min: return disputed, np.array([], dtype=int)
    pool = np.flatnonzero(dec['ambiguous'] & ~dec['disputed'])
    if len(pool) == 0: pool = np.flatnonzero(valid & ~dec['disputed'])
    extra = pool[np.argsort(dist[pool])][:max(k_min - len(disputed), 0)]
    return np.concatenate([disputed, extra]).astype(int), extra.astype(int)
