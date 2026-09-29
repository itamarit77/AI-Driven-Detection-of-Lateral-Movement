#!/usr/bin/env bash
# Milestone 3 — unattended finish. Run from a SECOND terminal while the current `--stage all --skip-llm` run
# keeps going in the first one (do not Ctrl+C it, do not close that terminal):
#
#     cd <the milestone3 folder>
#     chmod +x overnight.sh
#     nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="Milestone 3 overnight run" ./overnight.sh > overnight.log 2>&1 &
#
# What it does, in order (every step logged to overnight.log and results/OVERNIGHT_SUMMARY.txt):
#   1. waits until no run_m3.py process is running any more (the current run has finished or died)
#   2. re-runs `run_m3.py --stage all --skip-llm` — a no-op if everything finished, otherwise it resumes the missing stages
#   3. makes sure an Ollama server is reachable (starts one if needed) and that the model is pulled
#   4. runs the LLM stage (retries once), then the collect stage
#   5. writes results/models_listing.txt and zips results/ into m3_results_<timestamp>.zip
# If Ollama is missing or the LLM stage fails, the zip is still produced (without Part-4 results) and the summary says why.
set -u
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PY=${PYTHON:-python}
MODEL=$($PY - <<'EOF'
import config; print(config.LLM_MODEL)
EOF
)
URL=$($PY - <<'EOF'
import config; print(config.OLLAMA_URL.rstrip('/'))
EOF
)
SUMMARY=results/OVERNIGHT_SUMMARY.txt
mkdir -p results
log(){ echo "$(date '+%F %T') $*" | tee -a "$SUMMARY"; }
log "==== overnight.sh started (model $MODEL, ollama $URL) ===="

# 1. wait for the running pipeline to finish (poll every 60 s, up to 8 h)
waited=0
while pgrep -f "run_m3.py" > /dev/null; do
  sleep 60; waited=$((waited+60))
  if [ $((waited % 900)) -eq 0 ]; then log "still waiting for the running pipeline ($((waited/60)) min); last log line: $(tail -n 1 results/run.log 2>/dev/null)"; fi
  if [ $waited -ge 28800 ]; then log "gave up waiting after 8 h"; break; fi
done
log "no run_m3.py process running; last log line: $(tail -n 1 results/run.log 2>/dev/null)"

# 2. resume/complete the non-LLM stages (skips everything whose outputs exist)
$PY run_m3.py --stage all --skip-llm; rc=$?
log "run_m3.py --stage all --skip-llm exit code $rc"

# 3. Ollama server + model
llm_ok=0
if command -v ollama > /dev/null 2>&1; then
  if ! curl -s --max-time 5 "$URL/api/tags" > /dev/null; then
    log "no Ollama server reachable; starting one"
    nohup ollama serve > ollama_server.log 2>&1 &
    for i in $(seq 1 30); do sleep 2; curl -s --max-time 5 "$URL/api/tags" > /dev/null && break; done
  fi
  if curl -s --max-time 5 "$URL/api/tags" > /dev/null; then
    if ollama list 2>/dev/null | grep -q "^${MODEL}"; then log "model $MODEL present"; else log "pulling $MODEL"; ollama pull "$MODEL"; log "pull exit code $?"; fi
    ollama list 2>/dev/null | grep -q "^${MODEL}" && llm_ok=1
  else
    log "Ollama server did not come up (see ollama_server.log)"
  fi
else
  log "ollama is not installed: skipping the LLM stage (install it, then run: python run_m3.py --stage llm && python run_m3.py --stage collect)"
fi

# 4. LLM stage (retry once) + collect
if [ $llm_ok -eq 1 ]; then
  $PY run_m3.py --stage llm; rc=$?
  if [ $rc -ne 0 ]; then log "llm stage exit code $rc; retrying once in 60 s"; sleep 60; $PY run_m3.py --stage llm; rc=$?; fi
  log "llm stage exit code $rc"
  ollama ps > results/ollama_ps.txt 2>&1 || true
fi
$PY run_m3.py --stage collect; log "collect stage exit code $?"

# 5. results summary and listing
ls -la models/ > results/models_listing.txt 2>&1 || true
[ -f report_meta_template.json ] && cp report_meta_template.json results/ 2>/dev/null
$PY - <<'EOF' >> results/OVERNIGHT_SUMMARY.txt 2>&1
import json, os
R='results'
print('--- files present ---')
for f in sorted(os.listdir(R)):
    if f.endswith('.json') or f.endswith('.jsonl') or f.endswith('.log') or f.endswith('.txt'): print(f, os.path.getsize(os.path.join(R,f)))
try:
    j=json.load(open(os.path.join(R,'m3_results.json')))
    print('m3_results.json ok; llm results present for:', [d for d,v in j.get('llm',{}).items() if v])
except Exception as e: print('m3_results.json problem:', e)
EOF
STAMP=$(date '+%Y%m%d_%H%M')
rm -f "m3_results_${STAMP}.zip"; zip -qr "m3_results_${STAMP}.zip" results
log "==== done: send m3_results_${STAMP}.zip ($(du -h "m3_results_${STAMP}.zip" | cut -f1)) ===="
