"""Milestone 3 configuration. Edit the PATHS block for your machine; everything else has sane defaults.

Run modes (see README):  python run_m3.py --stage all            (full run)
                         python run_m3.py --stage all --quick    (smoke test on a subsample)
"""
import os
from pathlib import Path
_env = os.environ.get

# ----------------------------------------------------------------------------------------------
# PATHS  (edit these)
# ----------------------------------------------------------------------------------------------
LMD_CSV = Path(_env("M3_LMD_CSV", r"data/LMD-2023 [1.75M Elements][Labelled]checked.csv"))     # raw labelled LMD-2023 CSV
MORDOR_JSON = [Path(p) for p in _env("M3_MORDOR_JSON", r"data/apt29_evals_day1_manual_2020-05-01225525.json;data/apt29_evals_day2_manual_2020-05-02035409.json").split(";")]   # OTRF Mordor APT29 host JSON (day 1, day 2)
WORK = Path("work")          # intermediate parquet files (created)
RESULTS = Path("results")    # everything the report needs (created)
MODELS = Path("models")      # saved model artifacts (.pkl / .pt)

# ----------------------------------------------------------------------------------------------
# DETERMINISM
# ----------------------------------------------------------------------------------------------
SEED = 42                    # every splitter, NumPy, LightGBM, RandomForest, IsolationForest, torch (CPU + CUDA)

# ----------------------------------------------------------------------------------------------
# DATA / SPLITS
# ----------------------------------------------------------------------------------------------
DATASETS = {"lmd": "LMD-2023", "mordor": "Mordor APT29"}
PRIMARY, GENERALIZATION = "lmd", "mordor"       # Dataset A (primary) / Dataset B (generalisation)
N_FOLDS = 5                                     # StratifiedGroupKFold, groups = host x time block
GROUP_MINUTES = {"lmd": 10, "mordor": 2}          # block length per dataset (Mordor’s capture spans two ~35-min windows 5 h apart)
INNER_VAL_FRACTION = 0.2                        # grouped inner hold-out inside every training fold (threshold + early stopping)
QUICK_ROWS = {"lmd": 60_000, "mordor": 150_000}   # --quick: rows kept per dataset (whole host x block groups, attack blocks included)

# ----------------------------------------------------------------------------------------------
# MODELS (non-default hyper-parameters; the sensitivity stage varies the "primary" ones)
# ----------------------------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                 class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=SEED, n_jobs=-1, verbosity=-1, deterministic=True, force_row_wise=True)     # scale_pos_weight set per training fold
LGBM_EARLY_STOPPING = 50
IF_PARAMS = dict(n_estimators=200, max_samples=4096, contamination="auto", random_state=SEED, n_jobs=-1)
LSTM_PARAMS = dict(hidden=64, hidden2=32, dropout=0.3, lr=1e-3, batch_size=1024, epochs=8, patience=2, T=20)
LSTM_STRIDE = {"lmd": 4, "mordor": 2}           # a window ends at every stride-th event of a host
LSTM_FOLDS = 5                                  # set to 2 on a slow CPU (the report states the number used)

SENSITIVITY = {   # one-at-a-time around the base configuration, evaluated on fold 0 only
    "rf":   {"max_depth": [4, 8, 12, 20], "n_estimators": [100, 300, 600], "min_samples_leaf": [5, 20, 100]},
    "lgbm": {"num_leaves": [7, 15, 31, 63, 127], "learning_rate": [0.02, 0.05, 0.1, 0.2], "min_child_samples": [20, 100, 500, 2000]},
    "if":   {"n_estimators": [50, 100, 200, 400], "max_samples": [256, 1024, 4096, 16384]},
    "lstm": {"hidden": [16, 32, 64, 128], "T": [10, 20, 40], "dropout": [0.1, 0.3, 0.5]},
}

# ----------------------------------------------------------------------------------------------
# CASCADE
# ----------------------------------------------------------------------------------------------
CASCADE_PRECISION_TARGET = 0.95     # tau_hi: smallest Stage-1 threshold reaching this precision on inner validation
CASCADE_NPV_TARGET = 0.995          # tau_lo: largest Stage-1 threshold keeping this negative predictive value
LLM_MAX_ESCALATIONS = 150           # disputed events sent to the LLM per dataset (the rest use the Stage-2 decision)
LLM_MIN_ESCALATIONS = 50            # if fewer rows are disputed, top up with the most ambiguous non-disputed rows (flagged)
LLM_SENSITIVITY_SUBSET = 60         # events per dataset re-queried under each context ablation

# ----------------------------------------------------------------------------------------------
# LLM (Part 4)
# ----------------------------------------------------------------------------------------------
LLM_BACKEND = "ollama"              # "ollama" | "mock"  (mock = deterministic rule, for pipeline testing only)
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"      # fallback: "mistral:7b-instruct-v0.3-q4_K_M"
LLM_OPTIONS = dict(temperature=0.0, top_p=0.9, num_predict=512, seed=SEED)
