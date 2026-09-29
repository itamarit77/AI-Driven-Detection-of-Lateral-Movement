"""Window container for the LSTM: keeps ONE float32 feature matrix and gathers T-step windows by index on demand
(memory-light: no dense (N, T, D) tensor).  Windows are pre-padded with zeros and a mask marks real steps."""
import numpy as np
from .features import lstm_windows


class WindowSet:
    def __init__(self, X, y, feats, T, stride):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64); self.T = T; self.d = X.shape[1]
        self.ends, self.starts = lstm_windows(feats, T, stride)     # positions in feats/X (sorted by host, ts)
        self.n = len(self.ends); self.y = self.y[self.ends]           # window label = label of its last event

    def gather(self, widx):
        T = self.T; b = len(widx); x = np.zeros((b, T, self.d), dtype=np.float32); m = np.zeros((b, T), dtype=np.float32)
        for k, w in enumerate(widx):
            s, e = self.starts[w], self.ends[w]; L = e - s + 1
            x[k, T - L:, :] = self.X[s:e + 1]; m[k, T - L:] = 1.0
        return x, m

    def window_index_for_events(self, event_positions):
        """Map outer-fold event positions to the windows that END on those events."""
        lookup = {e: i for i, e in enumerate(self.ends)}
        return np.array([lookup[e] for e in event_positions if e in lookup], dtype=np.int64)
