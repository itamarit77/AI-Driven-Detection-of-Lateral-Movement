# Milestone 3 — pipeline, evaluation, cascade and LLM arbitration

This folder is the complete, self-contained Milestone-3 code. It reuses the Milestone-2 feature definitions
(`m3pkg/common_features.py`, identical apart from the passthrough columns) and adds the five-stage pipeline,
the four models, the evaluation, the cascade and the local-LLM arbitration.

## 1. What you need

* Python 3.10 – 3.12, ~16 GB RAM (LMD-2023 has 1.75 M rows), any OS. A GPU is optional (speeds up the LSTM).
* The raw data:
  * `LMD-2023 [1.75M Elements][Labelled]checked.csv` (the labelled 1.75 M-row variant).
  * Mordor APT29 day-1 and day-2 host JSON — `python get_mordor.py data` downloads and unzips them (≈2.1 GB).
* For Part 4: [Ollama](https://ollama.com) with the model pulled:
  `ollama pull llama3.1:8b-instruct-q4_K_M` (≈4.9 GB; needs ≈6 GB free RAM or VRAM).
  Fallback if that tag is unavailable: `ollama pull mistral:7b-instruct-v0.3-q4_K_M` and set `LLM_MODEL` in `config.py`.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # PyTorch: pick the CUDA build from pytorch.org if you have a GPU
mkdir -p data && cp "/path/to/LMD-2023 [1.75M Elements][Labelled]checked.csv" data/
python get_mordor.py data
```
Edit the `PATHS` block of `config.py` if your files live elsewhere.

## 2. Run order

```bash
python run_m3.py --stage all --quick --llm-backend mock    # 1) smoke test, ~5-15 min, writes results_quick/ (not used by the report)
python run_m3.py --stage all --skip-llm                     # 2) full run without the LLM: 4.1 h + 35 min refine on the 16-core laptop of Table 2
ollama serve &                                              # 3) start Ollama (skip if it already runs as a service)
python run_m3.py --stage llm                                # 4) LLM arbitration, ~15 min on a GPU, 2-4 h on CPU
python extract_escalated.py                                 # 5) escalated rows + labels for the arbitration baselines of Table 14 (seconds)
python run_m3.py --stage refine                             # 5b) §2.2 refinement (already part of --stage all; 35 min on 16 cores when run alone)
python run_m3.py --stage collect                            # 6) rebuild results/m3_results.json + figures (incl. the two architecture diagrams)
python host_counts.py                                       # 7) results/host_class_counts.json (per-host class counts quoted in §2.1)
```
Unattended alternative for steps 2–5: `nohup systemd-inhibit --what=sleep:idle:handle-lid-switch ./overnight.sh > overnight.log 2>&1 &`
(waits for a running pipeline, completes missing stages, starts Ollama, runs the LLM stage, zips `results/`).
Every stage skips work whose outputs already exist, so the run can be interrupted and resumed
(`--force` recomputes). LLM answers are cached in `results/llm_cache.jsonl`, so the `llm` stage also resumes.

Useful knobs: `--lstm-folds 2` (slow CPU), `--models rf,lgbm,if` (subset), `config.py: LLM_MAX_ESCALATIONS`
(number of disputed events sent to the LLM per dataset; 150 by default), `LLM_SENSITIVITY_SUBSET` (60).

### Re-running the LightGBM-dependent stages after the early-stopping fix
`./rerun_lgbm.sh` (≈25 min of model training plus the LLM stage): LightGBM training with cross-validation, its sensitivity sweep,
its transfer experiments, the cascade and the LLM stage are recomputed; RF, Isolation-Forest and LSTM results are reused.
The defect: `lgb.early_stopping` also watched the validation log-loss (LightGBM's default metric next to the requested AUC),
and with `scale_pos_weight` the log-loss can stagnate after one or two rounds. Measured effect on the full data (best iteration
before → after): LMD 426→596, 236→248, 79, 181→231, 224; Mordor 2, 47→48, 236→600, 146, 53→59; reduced-set LightGBM 3, 6→97,
12→16, 1, 1→2. Out-of-fold F1: LMD 0.9830→0.9831, Mordor 0.8369→0.8394, reduced 0.1475→0.1516. The superseded LightGBM cross-validation files are kept in
`results/before_lgbm_fix/`. Mordor fold 0's two-round stop is the inner AUC's own optimum (its curve peaks at round 2).

### Re-running the LLM stage after a code update
```bash
python run_m3.py --stage llm --force && python run_m3.py --stage collect
```
Every answer is cached in `results/llm_cache.jsonl` under a key that includes the rendered user prompt (not the system prompt or the
sampling options — after changing those, delete or rename the cache), so only prompts whose text
changed are sent to the model again; the rest is re-read from the cache in seconds. The `--force` flag is needed because the
stage otherwise skips datasets whose `llm_<ds>.json` already exists.

### Note on `run_m3.py` vs the run that produced `results/`
The LLM results in `results/` were produced before two bookkeeping additions to `stage_llm`: the flip rates are now computed
over event–variant pairs with a usable answer in both conditions (`n_paired` is stored), and the ablation loop logs progress.
Every answer in that run parsed (`n_unparsed = 0` in both `llm_*.json`), so paired = parsed and the reported flip
rates are unchanged; `python run_m3.py --stage llm --force` re-derives the files from `results/llm_cache.jsonl` in seconds
without new model calls. Prompts identical to those of an interrupted first LLM run (41 no-opinion prompts) were served
from that cache; the interrupted run's own results were never reported.

### What the submission must contain
The Milestone-3 brief asks for the report plus a **private GitHub/GitLab repository** (link + branch in Table 1) and a
**link to the saved models**. Push the tree described in §6 below; the 14 MB `models/` folder of the run (fold-0
artifacts per dataset, listed in `results/models_listing.txt`) goes into the same repository (or a shared drive) and its
link into Table 1's "Model artifact archive" cell. The two Table-1 cells are `repo` and `artifacts` in
`results/report_meta.json` (rebuild the report afterwards) — or edit the two cells in the .docx directly.

### The refine stage (report §2.2, v2b)
`python run_m3.py --stage refine` adds two feature families derived from the §2.1 forensics (`m3pkg/refine.py`:
A = trailing-60-s window-composition shares, B = burst ratios against the host's trailing-hour rate) and re-fits Random
Forest and LightGBM on the same folds / inner hold-outs / hyper-parameters as the main run, as increments base → +A → +A+B,
plus a per-fold *selected* variant chosen on the inner hold-out (never on the test fold). It also re-fits the baseline and
checks it against `cv_<ds>_<model>.json`, recomputes the two-stage cascade with each variant, and counts FP/FN per event
category (the categories §2.1 named). Output: `results/refine_<ds>.json`; the report reads it through `m3_results.json`
(`--stage collect`). Nothing downstream (cascade, LLM) is changed by it — Parts 3–4 keep the pre-registered models.

### Label-free arbitration rules at two precisions
`collect.py` evaluates the rules of Table 14 twice: on the full-precision scores (`baselines.rules`) and on the numbers exactly as
the prompt displayed them (`baselines.rules_displayed`: two decimals, integer percentile — `llm.models_text`'s formatting; a score
shown at its threshold counts as positive). The report's Table 14 uses the displayed numbers, i.e. what the LLM saw, and quotes the
full-precision values in the caption. `baselines.displayed_ties` counts the rows on which a score was shown equal to its threshold,
and `baselines.budget_matched_oracle` is Stage 2 with a perfect verdict on the escalated disputed rows only (the every-disputed-row
bound is `cascade_<ds>.json: stage3_oracle_upper_bound`). `python baselines.py results` prints the same two passes.

## 3. Building the report

```bash
cp report_meta_template.json results/report_meta.json   # fill in the FILL fields (IDs, links, hardware, ollama ps memory)
NODE_PATH=$(npm root -g) node build_report_m3.js results/m3_results.json Milestone3_Report.docx   # needs Node.js + `npm i -g docx`
```
Every number in the report is read from `results/m3_results.json` (and `host_class_counts.json`, `run.log`); the generator prints `[pending: ...]` where a result is missing.

## 4. What results/ contains

The **`results/`** folder holds everything the report reads: `m3_results.json`, all per-stage JSON files incl. `refine_<ds>.json`,
`run.log`, `figures/`, `llm_cache.jsonl`, the out-of-fold score arrays (`oof_<ds>_<model>.npy`, `fold_id_<ds>.npy`), the escalated
rows (`escalated_rows.json`, `labels_<ds>.npy`, `rowinfo_<ds>.npz`) and the superseded LightGBM cross-validation files (`before_lgbm_fix/`). `models_listing.txt` is the size listing of the `models/` folder cited in Table 1.


## 5. Stages (what the code does)

| Stage | Module | Output |
|---|---|---|
| 1 ingest | `m3pkg/ingest.py` — one adapter per raw source → common intermediate event schema; labels as in Milestone 2 (LMD creators' labels; Mordor provenance-only weak labels) | `work/<ds>_events.parquet`, `results/facts_<ds>.json` |
| 2 features | `m3pkg/features.py` → `build_features` (shared, identical for both datasets) | `work/<ds>_features.parquet` |
| 3 preprocess | grouped stratified 5-fold CV (host × time block), grouped inner hold-out per training fold (thresholds, early stopping, cascade bands), robust scaling fitted on training rows, class weights, quantile alignment | inside `results/cv_*.json` |
| 4 models | `m3pkg/models.py`: Random Forest, LightGBM, Isolation Forest, stacked LSTM (PyTorch) behind one interface | `models/*.pkl`, `models/*.pt` |
| 5 evaluate | out-of-fold metrics, confusion matrices, forensic error analysis, hyper-parameter sensitivity, cross-dataset transfer | `results/cv_*.json`, `sens_*.json`, `cross_*.json` |
| refine | `m3pkg/refine.py`: diagnostic-driven feature families (window composition, burst ratios), base → +A → +A+B re-fits of RF and LightGBM, inner-hold-out selection, per-category error counts, cascade per variant | `results/refine_<ds>.json` |
| cascade | `m3pkg/cascade.py`: LightGBM bands → RF on the ambiguous band → disputed rows escalated | `results/cascade_<ds>.json` |
| llm | `m3pkg/llm.py`: Ollama arbitration with cached, deterministic calls; context ablations | `results/llm_<ds>.json`, `results/llm_cache.jsonl` |
| collect | `m3pkg/collect.py`: one JSON for the report + figures (architecture diagrams `arch_pipeline.png` / `arch_cascade.png` from `m3pkg/diagrams.py`, sensitivity curves, out-of-fold metrics, transfer ROC-AUC, cascade stages, LLM ablations) | `results/m3_results.json`, `results/figures/*.png` |
| host_counts.py | per-host attack / benign row counts from the intermediate tables | `results/host_class_counts.json` |

Seeds: `SEED = 42` for every splitter, NumPy, LightGBM, scikit-learn and torch (CPU and CUDA; cuDNN deterministic).
Fold k of the LSTM uses seed 42 + k so that folds do not share an initialisation.

## 6. Repository layout (the private GitHub/GitLab repo linked in Table 1)

```
README.md         landing page (team, what is where, how to reproduce)
milestone2/       the Milestone-2 package (scripts, artifacts, figures, its README)
milestone3/       this folder: code, README, results/ (every number of the report), figures
models/           the run's saved models (14 MB: *_fold0.pkl, *_fold0.pt); listing in milestone3/results/models_listing.txt
reports/          Milestone2_Report.pdf, Milestone3_Report.pdf (+ .docx)
.gitignore        data/, work/, __pycache__/, .venv/
```
Do not commit `data/` or `work/`; `models/` is small enough to commit (or link it from a shared drive instead).
