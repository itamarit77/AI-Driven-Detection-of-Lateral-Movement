#!/usr/bin/env python3
"""Milestone 3 runner.  Stages (each is skipped when its outputs exist; use --force to redo):

  ingest      raw sources -> work/<ds>_events.parquet               (Stage 1, dataset adapters)
  features    events -> work/<ds>_features.parquet                    (Stage 2, shared feature code)
  train       5-fold grouped CV for rf / lgbm / if / lstm on both datasets, OOF scores, error analysis (Steps 7-8)
              (afterwards: python extract_escalated.py — escalated rows + labels for the arbitration baselines, run once before collect)
  sensitivity hyper-parameter curves on fold 0
  cross       train on A -> test on B and the reverse (raw / quantile-aligned; unified / reduced features)
  cascade     multi-stage cascade from the OOF scores, escalation list for the LLM (Part 3)
  llm         LLM arbitration of the disputed events + context-ablation sensitivity (Part 4)
  refine      diagnostic-driven refinement (§2.2): base / +A / +A+B re-fits of rf and lgbm, per-fold inner-hold-out selection
  collect     results/m3_results.json + figures (incl. the two architecture diagrams) — everything the report reads
  all         everything above in order

Examples:   python run_m3.py --stage all --quick          # ~10-15 min smoke test on a subsample
            python run_m3.py --stage all --skip-llm       # full run without the LLM stage
            python run_m3.py --stage llm                  # then the LLM stage when Ollama is up
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from m3pkg import ingest as ING
from m3pkg.features import make_features, UNIFIED, REDUCED, FEATURE_COLS
from m3pkg import pipeline as PL
from m3pkg.cascade import stage_decisions, cascade_report, pick_escalations
from m3pkg import llm as LLM

log = logging.getLogger("m3")
MODELS = ('rf', 'lgbm', 'if', 'lstm')


def setup_logging(results):
    results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(results / 'run.log')])
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=1, default=conv)


def strip_arrays(res):
    return {k: v for k, v in res.items() if k not in ('oof_scores', 'fold_id', 'scores')}


# --------------------------------------------------------------------------------------------- ingest / features
def quick_subset(events, n, minutes, seed):
    """Smoke-test subsample built from WHOLE host x <minutes> groups (so grouped CV still has many groups):
    every group that contains attack rows is kept first, then random benign groups until n rows."""
    g = events['host'].astype(str) + '|' + events['ts'].dt.floor(f'{minutes}min').astype(str)
    rng = np.random.RandomState(seed); sizes = g.value_counts()
    pos_groups = np.array(sorted(g[events['label'] == 1].unique())); rng.shuffle(pos_groups)
    others = np.array(sorted(set(g.unique()) - set(pos_groups))); rng.shuffle(others)
    keep, size = [], 0
    for grp in pos_groups:                     # groups with attack rows, up to half the budget (at least 3 groups)
        if size >= n // 2 and len(keep) >= 3: break
        keep.append(grp); size += int(sizes[grp])
    for grp in others:                         # then benign groups
        if size >= n: break
        keep.append(grp); size += int(sizes[grp])
    return events[g.isin(keep)].sort_values(['host', 'ts']).reset_index(drop=True)


def stage_ingest(args):
    facts = {}
    for ds in cfg.DATASETS:
        out = cfg.WORK / f"{ds}_events.parquet"
        if out.exists() and not args.force:
            log.info("skip ingest %s (exists)", ds); continue
        if ds == 'lmd':
            ev = ING.ingest_lmd(cfg.LMD_CSV); meta = {}
        else:
            ev, meta = ING.ingest_mordor(cfg.MORDOR_JSON)
        if args.quick: ev = quick_subset(ev, cfg.QUICK_ROWS[ds], cfg.GROUP_MINUTES[ds], cfg.SEED)
        cfg.WORK.mkdir(parents=True, exist_ok=True); ev.to_parquet(out, index=False)
        facts[ds] = {**ING.dataset_facts(ev), **meta, 'quick': bool(args.quick)}
        jdump(facts[ds], cfg.RESULTS / f"facts_{ds}.json")


def stage_features(args):
    for ds, name in cfg.DATASETS.items():
        out = cfg.WORK / f"{ds}_features.parquet"
        if out.exists() and not args.force:
            log.info("skip features %s (exists)", ds); continue
        ev = pd.read_parquet(cfg.WORK / f"{ds}_events.parquet"); t0 = time.time()
        feats = make_features(ev, name); feats.to_parquet(out, index=False)
        log.info("features %s: %s rows x %d features (%.0fs)", ds, len(feats), len(FEATURE_COLS), time.time() - t0)


def load_feats(ds):
    f = pd.read_parquet(cfg.WORK / f"{ds}_features.parquet")
    return f.sort_values(['host', 'ts']).reset_index(drop=True)      # build_features already sorts; keep the invariant explicit


# --------------------------------------------------------------------------------------------- train
def stage_train(args):
    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    for ds in cfg.DATASETS:
        feats = load_feats(ds)
        for name in args.models:
            out = cfg.RESULTS / f"cv_{ds}_{name}.json"
            if out.exists() and not args.force:
                log.info("skip train %s %s (exists)", ds, name); continue
            t0 = time.time()
            if name == 'lstm':
                res = PL.cv_lstm(cfg.LSTM_PARAMS, feats, UNIFIED, cfg, cfg.GROUP_MINUTES[ds], cfg.LSTM_STRIDE[ds], args.lstm_folds, cfg.MODELS, tag=ds)
            else:
                params = {'rf': cfg.RF_PARAMS, 'lgbm': cfg.LGBM_PARAMS, 'if': cfg.IF_PARAMS}[name]
                res = PL.cv_flat_model(name, params, feats, UNIFIED, cfg, cfg.GROUP_MINUTES[ds], cfg.MODELS, tag=ds)
            res['error_analysis'] = PL.run_error_analysis(feats, res, UNIFIED); res['seconds_total'] = round(time.time() - t0, 1)
            np.save(cfg.RESULTS / f"oof_{ds}_{name}.npy", res['oof_scores'])
            if name != 'lstm': np.save(cfg.RESULTS / f"fold_id_{ds}.npy", res['fold_id'])     # same outer folds for every flat model
            jdump(strip_arrays(res), out)
            # Mordor-honest evaluation: the reduced feature set (no seed-field features) for the supervised models
            if ds == 'mordor' and name in ('rf', 'lgbm'):
                r2 = PL.cv_flat_model(name, params, feats, REDUCED, cfg, cfg.GROUP_MINUTES[ds], None, tag=ds + '-reduced'); jdump(strip_arrays(r2), cfg.RESULTS / f"cv_{ds}_{name}_reduced.json")


def stage_refine(args):
    """§2.2 diagnostic-driven refinement: base -> +A (window composition) -> +A+B (burst ratios) for the two supervised
    per-event models, same folds / inner hold-outs / hyper-parameters; per-fold adoption chosen on the inner hold-out."""
    from m3pkg.refine import add_refinement_features, VARIANTS, error_by_category, select_on_inner
    for ds in cfg.DATASETS:
        out = cfg.RESULTS / f"refine_{ds}.json"
        if out.exists() and not args.force:
            log.info("skip refine %s (exists)", ds); continue
        t0 = time.time(); feats = add_refinement_features(load_feats(ds)); y = feats['label'].values; gm = cfg.GROUP_MINUTES[ds]
        res = {'dataset': ds, 'variants': list(VARIANTS), 'features_added': {k: v for k, v in VARIANTS.items() if v}, 'models': {}, 'cascade': {}}
        runs = {}
        for name in ('rf', 'lgbm'):
            params = {'rf': cfg.RF_PARAMS, 'lgbm': cfg.LGBM_PARAMS}[name]; runs[name] = {}; res['models'][name] = {}
            for vname, extra in VARIANTS.items():
                cols = UNIFIED + extra
                r = PL.cv_flat_model(name, params, feats, cols, cfg, gm, None, tag=f"{ds}-refine-{vname}"); runs[name][vname] = r
                res['models'][name][vname] = {**strip_arrays(r), 'error_by_category': error_by_category(feats, r['oof_scores'], r['fold_id'], r['fold_thresholds'])}
            res['models'][name]['selected'] = select_on_inner(runs[name], y, runs[name]['base']['fold_id'], feats)   # incl. the composite policy's per-category errors
            stored = cfg.RESULTS / f"cv_{ds}_{name}.json"                      # the main run's numbers (Table 6): the refit must reproduce them
            if stored.exists():
                st = json.load(open(stored)); fb = runs[name]['base']
                res['models'][name]['baseline_check'] = {'stored_oof_f1': st['oof_metrics']['f1'], 'refit_oof_f1': fb['oof_metrics']['f1'],
                                                         'max_abs_diff_fold_f1': float(max(abs(a['f1'] - b['f1']) for a, b in zip(st['fold_metrics'], fb['fold_metrics']))),
                                                         'identical_thresholds': bool(np.allclose(st['fold_thresholds'], fb['fold_thresholds']))}
                log.info("refine %s %s baseline refit: OOF F1 %.4f vs stored %.4f", ds, name, fb['oof_metrics']['f1'], st['oof_metrics']['f1'])
        # the two-stage cascade (Part 3 logic, no LLM) with the models of each variant: what the refinement does to the pipeline
        for vname in VARIANTS:
            r1, r2 = runs['lgbm'][vname], runs['rf'][vname]
            dec = stage_decisions(y, r1['oof_scores'], r2['oof_scores'], np.zeros(len(y)), r1['fold_bands'], r1['fold_thresholds'], r2['fold_thresholds'], r1['fold_id'])
            rep = cascade_report(y, dec); res['cascade'][vname] = {k: rep[k] for k in ('n', 'n_ambiguous', 'n_disputed', 'ambiguous_rate', 'disputed_rate', 'stage1', 'stage2')}
        res['seconds_total'] = round(time.time() - t0, 1); jdump(res, out)
        for name in ('rf', 'lgbm'):
            log.info("refine %s %s: OOF F1 %s | selected %.4f (%s)", ds, name, {v: round(runs[name][v]['oof_metrics']['f1'], 4) for v in VARIANTS},
                     res['models'][name]['selected']['oof_metrics']['f1'], res['models'][name]['selected']['chosen_per_fold'])


def stage_sensitivity(args):
    for ds in cfg.DATASETS:
        feats = load_feats(ds)
        for name in args.models:
            out = cfg.RESULTS / f"sens_{ds}_{name}.json"
            if out.exists() and not args.force:
                log.info("skip sensitivity %s %s", ds, name); continue
            grid = cfg.SENSITIVITY[name]
            if name == 'lstm': rows = PL.sensitivity_lstm(cfg.LSTM_PARAMS, grid, feats, UNIFIED, cfg, cfg.GROUP_MINUTES[ds], cfg.LSTM_STRIDE[ds])
            else: rows = PL.sensitivity_flat(name, {'rf': cfg.RF_PARAMS, 'lgbm': cfg.LGBM_PARAMS, 'if': cfg.IF_PARAMS}[name], grid, feats, UNIFIED, cfg, cfg.GROUP_MINUTES[ds])
            jdump({'dataset': ds, 'model': name, 'base': {k: str(v) for k, v in ({'rf': cfg.RF_PARAMS, 'lgbm': cfg.LGBM_PARAMS, 'if': cfg.IF_PARAMS, 'lstm': cfg.LSTM_PARAMS}[name]).items()}, 'rows': rows}, out)


def stage_cross(args):
    fa, fb = load_feats(cfg.PRIMARY), load_feats(cfg.GENERALIZATION)
    for src, tgt, fs, ft in [(cfg.PRIMARY, cfg.GENERALIZATION, fa, fb), (cfg.GENERALIZATION, cfg.PRIMARY, fb, fa)]:
        for name in args.models:
            for fset_name, cols in (('unified', UNIFIED), ('reduced', REDUCED)):
                for variant in ('raw', 'quantile'):
                    out = cfg.RESULTS / f"cross_{src}_to_{tgt}_{name}_{fset_name}_{variant}.json"
                    if out.exists() and not args.force: continue
                    t0 = time.time()
                    if name == 'lstm':
                        if fset_name == 'reduced': continue           # one feature set for the sequence model (cost)
                        res = PL.cross_dataset_lstm(cfg.LSTM_PARAMS, fs, ft, cols, cfg, cfg.GROUP_MINUTES[src], cfg.LSTM_STRIDE[src], cfg.LSTM_STRIDE[tgt], variant)
                    else:
                        params = {'rf': cfg.RF_PARAMS, 'lgbm': cfg.LGBM_PARAMS, 'if': cfg.IF_PARAMS}[name]
                        res = PL.cross_dataset(name, params, fs, ft, cols, cfg, cfg.GROUP_MINUTES[src], variant)
                    res['source'], res['target'], res['feature_set'], res['seconds'] = src, tgt, fset_name, round(time.time() - t0, 1)
                    np.save(cfg.RESULTS / f"cross_scores_{src}_to_{tgt}_{name}_{fset_name}_{variant}.npy", res['scores']); jdump(strip_arrays(res), out)
                    log.info("cross %s->%s %s %s %s: F1 %.4f AUC %s", src, tgt, name, fset_name, variant, res['target_metrics']['f1'], res['target_metrics']['roc_auc'])


# --------------------------------------------------------------------------------------------- cascade / llm
def load_cascade_inputs(ds):
    feats = load_feats(ds); y = feats['label'].values
    p1 = np.load(cfg.RESULTS / f"oof_{ds}_lgbm.npy"); p2 = np.load(cfg.RESULTS / f"oof_{ds}_rf.npy"); an = np.load(cfg.RESULTS / f"oof_{ds}_if.npy")
    fold_id = np.load(cfg.RESULTS / f"fold_id_{ds}.npy")
    r1 = json.load(open(cfg.RESULTS / f"cv_{ds}_lgbm.json")); r2 = json.load(open(cfg.RESULTS / f"cv_{ds}_rf.json"))
    dec = stage_decisions(y, p1, p2, an, r1['fold_bands'], r1['fold_thresholds'], r2['fold_thresholds'], fold_id)
    return feats, y, p1, p2, an, dec


def stage_cascade(args):
    for ds in cfg.DATASETS:
        out = cfg.RESULTS / f"cascade_{ds}.json"
        if out.exists() and not args.force: continue
        feats, y, p1, p2, an, dec = load_cascade_inputs(ds)
        rep = cascade_report(y, dec); esc, extra = pick_escalations(dec, p1, cfg.LLM_MAX_ESCALATIONS, cfg.LLM_MIN_ESCALATIONS)
        rep['escalated_positions'] = esc.tolist(); rep['escalated_row_ids'] = feats['row_id'].values[esc].tolist(); rep['near_threshold_topup_positions'] = extra.tolist()
        rep['band_stats'] = {'tau_lo_median': float(np.median(dec['band_lo'])), 'tau_hi_median': float(np.median(dec['band_hi']))}
        # a random sample of the LLM-sensitivity subset is drawn from the escalations (deterministic)
        rng = np.random.RandomState(cfg.SEED); sub = rng.choice(esc, size=min(cfg.LLM_SENSITIVITY_SUBSET, len(esc)), replace=False) if len(esc) else np.array([], dtype=int)
        rep['sensitivity_positions'] = sub.tolist(); jdump(rep, out)
        log.info("cascade %s: ambiguous %.3f disputed %.3f | S1 F1 %.4f -> S2 F1 %.4f | escalated %d", ds, rep['ambiguous_rate'], rep['disputed_rate'], rep['stage1']['f1'], rep['stage2']['f1'], len(esc))


def stage_llm(args):
    if cfg.LLM_BACKEND != 'mock':
        import requests
        try:
            tags = requests.get(cfg.OLLAMA_URL.rstrip('/') + '/api/tags', timeout=10).json()
        except Exception as e:
            log.error("Ollama is not reachable at %s (%s). Start it with `ollama serve` and pull the model: `ollama pull %s`", cfg.OLLAMA_URL, e, cfg.LLM_MODEL); sys.exit(2)
        names = [m.get('name') for m in tags.get('models', [])]
        if cfg.LLM_MODEL not in names:
            log.error("model %s not found in Ollama (available: %s). Run: ollama pull %s", cfg.LLM_MODEL, names, cfg.LLM_MODEL); sys.exit(2)
    client = LLM.MockClient() if cfg.LLM_BACKEND == 'mock' else LLM.OllamaClient(cfg.OLLAMA_URL, cfg.LLM_MODEL, cfg.LLM_OPTIONS)
    arb = LLM.Arbiter(client, cfg.RESULTS / 'llm_cache.jsonl', cfg.LLM_MODEL if cfg.LLM_BACKEND != 'mock' else 'mock')
    info = client.info(); ABL = ['full', 'no_models', 'no_window', 'swapped_models', 'shuffled']
    for ds in cfg.DATASETS:
        out = cfg.RESULTS / f"llm_{ds}.json"
        if out.exists() and not args.force: continue
        feats, y, p1, p2, an, dec = load_cascade_inputs(ds); casc = json.load(open(cfg.RESULTS / f"cascade_{ds}.json"))
        events = pd.read_parquet(cfg.WORK / f"{ds}_events.parquet").set_index('row_id')
        esc = np.array(casc['escalated_positions'], dtype=int); sub = np.array(casc['sensitivity_positions'], dtype=int)
        topup = set(int(x) for x in casc.get('near_threshold_topup_positions', []))   # flagged: never enter the cascade metrics
        verdicts, records = {}, []
        for i, pos in enumerate(esc):
            rid = int(feats['row_id'].values[pos]); ctx = LLM.build_context(events.loc[rid], feats.iloc[pos], p1[pos], p2[pos], an[pos], dec['thr1'][pos], dec['thr2'][pos])
            r = arb.ask(ds, rid, ctx, 'full'); records.append({**r, 'label': int(y[pos]), 'stage1': int(dec['lean1'][pos]), 'rf': int(dec['rf'][pos]), 'position': int(pos), 'topup': int(int(pos) in topup)})
            if r['verdict'] is not None: verdicts[int(pos)] = int(r['verdict'])
            if (i + 1) % 10 == 0: log.info("llm %s %d/%d", ds, i + 1, len(esc))
        rep = cascade_report(y, dec, verdicts, topup=topup); rep['n_unparsed'] = int(sum(1 for r in records if r['verdict'] is None))
        rep['mean_seconds_per_call'] = float(np.mean([r['seconds'] for r in records])) if records else None
        rep['mean_output_tokens'] = float(np.mean([r['output_tokens'] for r in records if r.get('output_tokens')])) if any(r.get('output_tokens') for r in records) else None
        # ---- context sensitivity on the subset ----
        sens = {}
        for abl in ABL:
            vs, flips, correct, paired = [], 0, 0, 0
            for j, pos in enumerate(sub):
                rid = int(feats['row_id'].values[pos]); ctx = LLM.build_context(events.loc[rid], feats.iloc[pos], p1[pos], p2[pos], an[pos], dec['thr1'][pos], dec['thr2'][pos])
                r = arb.ask(ds, rid, ctx, abl); base = verdicts.get(int(pos))
                if (j + 1) % 10 == 0: log.info("llm %s ablation %s %d/%d", ds, abl, j + 1, len(sub))
                if r['verdict'] is not None:
                    vs.append((int(pos), int(r['verdict']))); correct += int(r['verdict'] == y[pos])
                    if base is not None: paired += 1; flips += int(r['verdict'] != base)      # flip rate over PAIRED usable answers only
            sens[abl] = {'n': len(sub), 'n_parsed': len(vs), 'n_paired': paired, 'accuracy': correct / len(vs) if vs else None, 'flip_rate_vs_full': flips / paired if paired else None,
                         'metrics': cascade_report(y[[p for p, _ in vs]], {'ambiguous': np.ones(len(vs), bool), 'disputed': np.ones(len(vs), bool), 'stage1': np.array([v for _, v in vs]), 'stage2': np.array([v for _, v in vs]), 'rf': np.array([v for _, v in vs]), 'lean1': np.array([v for _, v in vs])})['stage2'] if vs else None}
        rep['sensitivity'] = sens; rep['llm_info'] = info; rep['options'] = cfg.LLM_OPTIONS; rep['backend'] = cfg.LLM_BACKEND
        rep['system_prompt'] = LLM.SYSTEM_PROMPT; rep['user_template'] = LLM.USER_TEMPLATE
        if records:
            ex = records[0]; rid = ex['row_id']; pos = ex['position']
            rep['example_prompt'] = LLM.render(LLM.build_context(events.loc[rid], feats.iloc[pos], p1[pos], p2[pos], an[pos], dec['thr1'][pos], dec['thr2'][pos]))
            rep['example_answer'] = {k: ex[k] for k in ('verdict', 'confidence', 'reason', 'label')}
        rep['records'] = records; jdump(rep, out)
        log.info("llm %s: S2 F1 %.4f -> S3 F1 %.4f (LLM on disputed: acc-F1 %s)", ds, rep['stage2']['f1'], rep['stage3']['f1'] if 'stage3' in rep else float('nan'), rep.get('llm_on_disputed', {}).get('f1'))


def stage_collect(args):
    from m3pkg.collect import collect
    collect(cfg, MODELS)


STAGES = {'ingest': stage_ingest, 'features': stage_features, 'train': stage_train, 'refine': stage_refine, 'sensitivity': stage_sensitivity,
          'cross': stage_cross, 'cascade': stage_cascade, 'llm': stage_llm, 'collect': stage_collect}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all', choices=list(STAGES) + ['all'])
    ap.add_argument('--quick', action='store_true', help='smoke test on a subsample (results_quick/, not used by the report)')
    ap.add_argument('--force', action='store_true', help='recompute even if outputs exist')
    ap.add_argument('--skip-llm', action='store_true'); ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--lstm-folds', type=int, default=cfg.LSTM_FOLDS)
    ap.add_argument('--llm-backend', default=None, choices=[None, 'ollama', 'mock'])
    args = ap.parse_args(); args.models = tuple(m for m in args.models.split(',') if m)
    if args.llm_backend: cfg.LLM_BACKEND = args.llm_backend
    if args.quick:
        cfg.RESULTS = Path(str(cfg.RESULTS) + '_quick'); cfg.WORK = Path(str(cfg.WORK) + '_quick'); cfg.MODELS = Path(str(cfg.MODELS) + '_quick')
    setup_logging(cfg.RESULTS); log.info("args %s", vars(args))
    order = list(STAGES) if args.stage == 'all' else [args.stage]
    for st in order:
        if st == 'llm' and args.skip_llm: log.info("skipping llm stage"); continue
        t0 = time.time(); log.info("==== stage %s ====", st); STAGES[st](args); log.info("==== stage %s done in %.0fs ====", st, time.time() - t0)
