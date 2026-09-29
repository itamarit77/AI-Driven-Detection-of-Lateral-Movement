"""Consistency checks on a results/ folder before the report is built. Usage: python check_results.py results"""
import json, sys, os, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
def J(name):
    p = os.path.join(R, name); return json.load(open(p)) if os.path.exists(p) else None
ok = True
def chk(cond, msg):
    global ok
    print(('PASS ' if cond else 'FAIL ') + msg); ok = ok and bool(cond)

M = J('m3_results.json'); chk(M is not None, 'm3_results.json present')
for ds in ('lmd', 'mordor'):
    lg = J(f'cv_{ds}_lgbm.json'); chk(lg is not None, f'{ds}: cv_lgbm present')
    if lg:
        it = lg.get('lgbm_iterations', []); chk(it and all('best_inner_auc' in x for x in it), f'{ds}: LightGBM run stores best_inner_auc; best_iterations {[x["best_iteration"] for x in it]}, inner AUC {[round(x["best_inner_auc"], 4) if x.get("best_inner_auc") else None for x in it]}, rounds run {[x.get("rounds_run") for x in it]}')
        chk('fold_bands_raw' in lg, f'{ds}: raw bands stored: {[tuple(round(v, 3) for v in b) for b in lg.get("fold_bands_raw", [])]}')
        print(f'      thresholds {[round(t, 3) for t in lg["fold_thresholds"]]}, bands {[tuple(round(v, 3) for v in b) for b in lg["fold_bands"]]}')
        print(f'      OOF F1 {lg["oof_metrics"]["f1"]:.4f} (0.5: {lg["oof_metrics_thr05"]["f1"]:.4f}) AUC {lg["oof_metrics"]["roc_auc"]:.4f} | per-fold F1 {[round(f["f1"], 3) for f in lg["fold_metrics"]]}')
    ca = J(f'cascade_{ds}.json'); ll = J(f'llm_{ds}.json')
    if ca:
        print(f'      cascade: ambiguous {ca["n_ambiguous"]} disputed {ca["n_disputed"]} escalated {len(ca["escalated_positions"])} topups {len(ca.get("near_threshold_topup_positions", []))} | S1 {ca["stage1"]["f1"]:.4f} S2 {ca["stage2"]["f1"]:.4f} oracle {ca["stage3_oracle_upper_bound"]["f1"]:.4f}')
    if ll:
        chk('n_llm_in_cascade' in ll and all('topup' in r for r in ll['records']), f'{ds}: LLM stage stores top-up flags (n_llm {ll.get("n_llm")}, in cascade {ll.get("n_llm_in_cascade")}, topups {ll.get("n_llm_topup")}, unparsed {ll.get("n_unparsed")})')
        chk(ll['stage3']['f1'] <= ll['stage3_oracle_upper_bound']['f1'] + 1e-12, f'{ds}: stage3 F1 {ll["stage3"]["f1"]:.4f} <= oracle {ll["stage3_oracle_upper_bound"]["f1"]:.4f}')
        if ca and ca['n_disputed'] == 0: chk(ll['stage3'] == ll['stage2'], f'{ds}: no disputed rows -> stage3 == stage2')
        chk(ll.get('backend') == 'ollama', f'{ds}: backend {ll.get("backend")} model {ll.get("llm_info", {}).get("model")}')
        p = ll.get('example_prompt', ''); chk('lateral-movement score' in p and 'not calibrated' in p, f'{ds}: new prompt wording in example')
    if M and M.get('baselines', {}).get(ds):
        b = M['baselines'][ds]; chk('rules_displayed' in b and 'budget_matched_oracle' in b, f'{ds}: baselines at displayed precision + budget-matched oracle present (ties GB {b.get("displayed_ties", {}).get("gb_score_equals_threshold")}, RF {b.get("displayed_ties", {}).get("rf_score_equals_threshold")}; budget-bound F1 {round(b["budget_matched_oracle"]["f1"], 4) if b.get("budget_matched_oracle") else None})')
    if M and 'diagnostics' in M and ds in M['diagnostics']:
        d = M['diagnostics'][ds]; bd = d.get('band_vs_threshold', [])
        if ca and bd: chk(sum(b['n_ambiguous'] for b in bd) == ca['n_ambiguous'], f'{ds}: diagnostics ambiguous count {sum(b["n_ambiguous"] for b in bd)} == cascade {ca["n_ambiguous"]}')
        ov = d.get('error_overlap', {}); print(f'      overlap rf/lgbm: {ov.get("rf_lgbm")}')
    else:
        chk(False, f'{ds}: diagnostics missing in m3_results.json (run --stage collect)')
    ex = J('escalated_rows.json')
    if ex and ca: chk(ex[ds]['positions'] == ca['escalated_positions'], f'{ds}: escalated_rows.json matches cascade escalations')
    lab = os.path.join(R, f'labels_{ds}.npy')
    if os.path.exists(lab):
        y = np.load(lab); chk(M and len(y) == M['facts'][ds]['rows'], f'{ds}: labels length {len(y)}, positives {int(y.sum())}')
    rf_ = J(f'refine_{ds}.json')
    if rf_:
        for m in ('rf', 'lgbm'):
            M_ = rf_['models'][m]; bc = M_.get('baseline_check', {}); sel = M_['selected']
            chk(all(v in M_ for v in ('base', 'A', 'AB')) and len(sel['chosen_per_fold']) == len(M_['base']['fold_metrics']), f'{ds}: refine {m}: F1 base {M_["base"]["oof_metrics"]["f1"]:.4f} +A {M_["A"]["oof_metrics"]["f1"]:.4f} +A+B {M_["AB"]["oof_metrics"]["f1"]:.4f} selected {sel["oof_metrics"]["f1"]:.4f} {sel["chosen_per_fold"]}; baseline refit vs stored: {bc.get("refit_oof_f1", float("nan")):.4f} vs {bc.get("stored_oof_f1", float("nan")):.4f}')
    else:
        print(f'INFO {ds}: refine_{ds}.json absent (run_m3.py --stage refine not run yet; the report prints a pending marker in §2.2)')
    fid = os.path.join(R, f'fold_id_{ds}.npy'); oof = os.path.join(R, f'oof_{ds}_lgbm.npy')
    chk(os.path.exists(fid) and os.path.exists(oof), f'{ds}: fold_id and oof_lgbm arrays present')
for f in ('arch_pipeline.png', 'arch_cascade.png'):
    chk(os.path.exists(os.path.join(R, 'figures', f)), f'figures/{f} present (architecture diagrams, drawn by --stage collect)')
print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
