# AI-Driven Detection of Lateral Movement (MITRE ATT&CK TA0008)

Course project "AI-Driven Intrusion and Malware Detection" — Applied AI/ML Architectures for Cyber Intrusion Detection.
Primary dataset (A): LMD-2023 (Sysmon, 1,752,836 events). Generalization dataset (B): Mordor APT29 evaluation host telemetry (783,367 events).

## Team
- Ela Ambar Grynbaum — ela.ambar2306@gmail.com
- Daniel Shostak — danshostak0@gmail.com
- Itamar Tsafrir Ornstein — itamarit77@gmail.com
- Itamar Klartag — itamarklartag@gmail.com

## What is where
| Folder | Content |
|---|---|
| `milestone2/` | Milestone 2 — data exploration, feature selection and feature engineering: scripts, artifacts (every statistic of the 42 candidate features), figures, its own README |
| `milestone3/` | Milestone 3 — the five-stage pipeline (`run_m3.py`, `m3pkg/`), the report generator (`build_report_m3.js`), and `results/`: every number, table and figure of the Milestone-3 report (cross-validation, sensitivity, transfer, refinement, cascade, LLM arbitration incl. the cached LLM answers) |
| `models/` | saved fold-0 model artifacts of the run: Random Forest, LightGBM, Isolation Forest (`.pkl`) and LSTM (`.pt`) per dataset, 14 MB (listing: `milestone3/results/models_listing.txt`) |
| `reports/` | the Milestone-2 and Milestone-3 reports (PDF and DOCX) |

## Reproducing Milestone 3
`milestone3/README.md` gives the full run order. In short:
```bash
cd milestone3 && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# place the LMD-2023 CSV under data/ and fetch Mordor with: python get_mordor.py data
python run_m3.py --stage all --skip-llm          # ingest → features → train → refine → sensitivity → cross → cascade (≈5 h on a 16-core CPU)
python run_m3.py --stage llm                     # local Llama 3.1 8B via Ollama (≈4 h on CPU; answers are cached)
python run_m3.py --stage collect                 # results/m3_results.json + figures
NODE_PATH=$(npm root -g) node build_report_m3.js results/m3_results.json Milestone3_Report.docx   # the report, every number read from results/
```
Every stage is checkpointed and skips work whose outputs exist; `python check_results.py results` verifies a results folder.
Seeds: `SEED = 42` everywhere (`config.py`); the LLM runs at temperature 0 with seed 42.

## Data
Not committed (size and licence): LMD-2023 from https://github.com/ChristosSmiliotopoulos/Lateral-Movement-Dataset--LMD_Collections
(the labelled 1.75 M-row CSV) and Mordor APT29 from https://github.com/OTRF/Security-Datasets (`get_mordor.py` downloads it).
