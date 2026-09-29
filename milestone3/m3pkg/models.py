"""Stage 4 — the four models behind one interface:  m = make_model(name, params); m.fit(X, y, X_val, y_val); m.score(X)
score() returns a probability-like attack score in [0, 1] (higher = more likely lateral movement), so thresholds,
cascades and error analysis treat every model identically.

  rf    Random Forest (supervised, bagging)          class_weight='balanced_subsample'
  lgbm  LightGBM (supervised, boosting)              scale_pos_weight = n_neg/n_pos of the training fold, early stopping on inner val
  if    Isolation Forest (unsupervised, anomaly)     fitted on training rows WITHOUT labels; score = anomaly percentile
  lstm  Stacked LSTM (deep, sequential, PyTorch)     class-weighted BCE (pos_weight = n_neg/n_pos), early stopping on inner val
"""
import logging, math, os, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import lightgbm as lgb

log = logging.getLogger("m3.models")

try:
    import torch, torch.nn as nn
    TORCH = True
except Exception:          # the sklearn fallback below is ONLY for pipeline smoke tests without PyTorch
    TORCH = False


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    if TORCH:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class RFModel:
    name = 'rf'
    def __init__(self, params): self.params = dict(params)
    def fit(self, X, y, X_val=None, y_val=None):
        self.clf = RandomForestClassifier(**self.params).fit(X, y); return self
    def score(self, X): return self.clf.predict_proba(X)[:, 1]
    def artifact(self): return self.clf


class LGBMModel:
    name = 'lgbm'
    def __init__(self, params, early_stopping=50): self.params = dict(params); self.es = early_stopping
    def fit(self, X, y, X_val=None, y_val=None):
        p = dict(self.params); n_pos = max(int(y.sum()), 1); p['scale_pos_weight'] = (len(y) - n_pos) / n_pos
        # Early stopping must watch the inner-validation AUC ONLY. Without `metric='auc'` LightGBM also evaluates the
        # objective's default binary_logloss and the callback stops on whichever metric stagnates first; with
        # scale_pos_weight the validation log-loss is best after 1-2 rounds, which silently truncated the first run.
        p.setdefault('metric', 'auc'); self.clf = lgb.LGBMClassifier(**p); self.best_inner_auc = None; self.rounds_run = None; self.inner_auc_curve = None
        if X_val is not None and len(np.unique(y_val)) == 2:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cb = [lgb.early_stopping(self.es, first_metric_only=True, verbose=False)]
                try:
                    self.clf.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=cb)
                except TypeError:      # LightGBM >= 5 renamed the validation arguments
                    self.clf.fit(X, y, eval_X=X_val, eval_y=y_val, eval_metric='auc', callbacks=cb)
            self.best_iteration = int(self.clf.best_iteration_ or p['n_estimators'])
            try:
                curve = list(self.clf.evals_result_.values())[0]['auc']; self.inner_auc_curve = [round(float(v), 5) for v in curve]
                self.rounds_run = len(curve); self.best_inner_auc = float(curve[self.best_iteration - 1])
            except Exception:
                pass
        else:
            self.clf.fit(X, y); self.best_iteration = p['n_estimators']
        return self
    def score(self, X): return self.clf.predict_proba(X)[:, 1]
    def artifact(self): return self.clf


class IFModel:
    """Unsupervised: labels are never seen by fit(). The raw score is -decision_function; it is mapped to the
    percentile among the TRAINING rows so that 0.99 means 'more anomalous than 99% of training traffic'."""
    name = 'if'
    def __init__(self, params): self.params = dict(params)
    def fit(self, X, y=None, X_val=None, y_val=None):
        self.clf = IsolationForest(**self.params).fit(X)
        raw = -self.clf.decision_function(X); self.ref = np.sort(raw); return self
    def score(self, X):
        raw = -self.clf.decision_function(X)
        return np.searchsorted(self.ref, raw, side='right') / len(self.ref)
    def artifact(self): return self.clf


# ----------------------------------------------------------------------------------------------- LSTM
if TORCH:
    class _Net(nn.Module):
        def __init__(self, d_in, hidden, hidden2, dropout):
            super().__init__()
            self.l1 = nn.LSTM(d_in, hidden, batch_first=True); self.l2 = nn.LSTM(hidden, hidden2, batch_first=True)
            self.bn = nn.BatchNorm1d(hidden2); self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden2, 1)
        def forward(self, x, mask):
            h, _ = self.l1(x); h = self.drop(h); h, _ = self.l2(h)
            last = h[:, -1, :]                      # windows are right-aligned (pre-padded), so the last step is the event
            return self.out(self.drop(self.bn(last))).squeeze(1)


class LSTMModel:
    """Sequence model over per-host event windows. fit()/score() take a WindowSet (see windows.py), not a flat matrix."""
    name = 'lstm'
    def __init__(self, params, seed=42):
        self.p = dict(params); self.seed = seed
        self.device = 'cuda' if (TORCH and torch.cuda.is_available()) else 'cpu'

    def _batches(self, ws, idx, shuffle):
        idx = np.array(idx)
        if shuffle:
            rng = np.random.RandomState(self.seed); rng.shuffle(idx)
        bs = self.p['batch_size']
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; x, m = ws.gather(b)
            yield torch.from_numpy(x).to(self.device), torch.from_numpy(m).to(self.device), b

    def fit(self, ws, train_idx, val_idx=None):
        if not TORCH:
            return self._fit_fallback(ws, train_idx, val_idx)
        seed_everything(self.seed)
        self.net = _Net(ws.d, self.p['hidden'], self.p['hidden2'], self.p['dropout']).to(self.device)
        y = ws.y[train_idx]; n_pos = max(int(y.sum()), 1)
        pos_w = torch.tensor([(len(y) - n_pos) / n_pos], dtype=torch.float32, device=self.device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w); opt = torch.optim.Adam(self.net.parameters(), lr=self.p['lr'])
        best, best_state, bad = -1.0, None, 0; self.history = []
        for ep in range(self.p['epochs']):
            self.net.train(); tot = 0.0
            for xb, mb, b in self._batches(ws, train_idx, True):
                if len(b) < 2: continue                       # BatchNorm needs more than one sample in training mode
                yb = torch.from_numpy(ws.y[b].astype(np.float32)).to(self.device)
                opt.zero_grad(); loss = lossf(self.net(xb, mb), yb); loss.backward(); opt.step(); tot += float(loss) * len(b)
            rec = {'epoch': ep + 1, 'train_loss': tot / len(train_idx)}
            if val_idx is not None and len(val_idx):
                from sklearn.metrics import roc_auc_score
                s = self.score(ws, val_idx); yv = ws.y[val_idx]
                rec['val_auc'] = float(roc_auc_score(yv, s)) if len(np.unique(yv)) == 2 else float('nan')
                if rec['val_auc'] > best:
                    best, bad = rec['val_auc'], 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            self.history.append(rec); log.info("lstm epoch %s", rec)
            if bad >= self.p['patience']:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def score(self, ws, idx):
        if not TORCH:
            return self._score_fallback(ws, idx)
        self.net.eval(); out = np.zeros(len(idx), dtype=np.float64); pos = 0
        with torch.no_grad():
            for xb, mb, b in self._batches(ws, idx, False):
                out[pos:pos + len(b)] = torch.sigmoid(self.net(xb, mb)).cpu().numpy(); pos += len(b)
        return out

    # --- fallback used only when PyTorch is absent (pipeline test): logistic regression on the flattened window ---
    def _fit_fallback(self, ws, train_idx, val_idx):
        from sklearn.linear_model import LogisticRegression
        x, _ = ws.gather(np.array(train_idx)); X = x.reshape(len(train_idx), -1)
        self.fb = LogisticRegression(max_iter=200, class_weight='balanced').fit(X, ws.y[train_idx]); self.history = [{'fallback': True}]
        log.warning("PyTorch not available: LSTM replaced by a logistic-regression FALLBACK (test only)"); return self
    def _score_fallback(self, ws, idx):
        x, _ = ws.gather(np.array(idx)); return self.fb.predict_proba(x.reshape(len(idx), -1))[:, 1]

    def artifact(self):
        return {'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()}, 'params': self.p} if TORCH else self.fb


def make_model(name, params, **kw):
    return {'rf': RFModel, 'lgbm': LGBMModel, 'if': IFModel, 'lstm': LSTMModel}[name](params, **kw)
