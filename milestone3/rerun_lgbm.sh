#!/usr/bin/env bash
# Re-runs every stage that depends on LightGBM after the early-stopping fix in m3pkg/models.py
# (early stopping now watches the inner-validation AUC only). RF / Isolation-Forest / LSTM results are untouched.
# Superseded LightGBM-dependent files are kept in results/before_lgbm_fix/ for the record.
#   cd <milestone3 folder>; chmod +x rerun_lgbm.sh; ./rerun_lgbm.sh 2>&1 | tee rerun_lgbm.log
set -u
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PY=${PYTHON:-python}
mkdir -p results/before_lgbm_fix
cp results/cv_*lgbm*.json results/sens_*lgbm*.json results/cross_*lgbm*.json results/cascade_*.json results/llm_*.json results/m3_results.json results/before_lgbm_fix/ 2>/dev/null
$PY run_m3.py --stage train       --models lgbm --force || { echo "train failed"; exit 1; }
$PY run_m3.py --stage sensitivity --models lgbm --force || { echo "sensitivity failed"; exit 1; }
$PY run_m3.py --stage cross       --models lgbm --force || { echo "cross failed"; exit 1; }
$PY run_m3.py --stage cascade     --force               || { echo "cascade failed"; exit 1; }
$PY run_m3.py --stage llm         --force               || { echo "llm stage failed; retrying once in 30 s"; sleep 30; $PY run_m3.py --stage llm --force; }
$PY run_m3.py --stage collect
$PY extract_escalated.py
ls -la models/ > results/models_listing.txt 2>&1
ollama ps > results/ollama_ps.txt 2>&1 || true
STAMP=$(date '+%Y%m%d_%H%M')
zip -qr "m3_results_after_fix_${STAMP}.zip" results -x 'results/cross_scores_*' 'results/oof_*_rf.npy' 'results/oof_*_if.npy' 'results/oof_*_lstm.npy' 'results/before_lgbm_fix/*'
echo "==== done: send m3_results_after_fix_${STAMP}.zip ($(du -h "m3_results_after_fix_${STAMP}.zip" | cut -f1)) ===="
