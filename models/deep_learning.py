"""
models/deep_learning.py
Deep-learning model for trading signal prediction.

Implements two sequence-based architectures:
  • CNN1D  — stacked 1-D convolutional layers with residual connections
  • LSTM   — bidirectional LSTM with attention

Both operate on a rolling window of the previous ``sequence_length`` bars,
giving the model temporal context unavailable to bar-by-bar ML models.

Backend
-------
Primary  : PyTorch  (install: pip install torch --index-url https://download.pytorch.org/whl/cpu)
Fallback : scikit-learn MLPClassifier  (already installed — works without PyTorch)

Usage
-----
    model = DeepModel(settings)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)   # (n, 2): [P(DOWN), P(UP)]
    model.save(Path("models/saved/model_deep.pkl"))
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.base import BaseModel

logger = logging.getLogger(__name__)

# ── PyTorch availability ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
    logger.debug("PyTorch %s available — using native deep learning backend.", torch.__version__)
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not installed — DeepModel will use sklearn MLPClassifier fallback.")

# ── Import GPU detection (lazy — avoids import-time side effects) ─────────────
def _get_device():
    """Return the best available device via gpu_utils (CUDA → DirectML → MPS → CPU)."""
    try:
        from models.gpu_utils import get_torch_device
        dev = get_torch_device()
        return dev if dev is not None else (torch.device("cpu") if _TORCH_AVAILABLE else None)
    except Exception:
        return torch.device("cpu") if _TORCH_AVAILABLE else None


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch model architectures
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class _ResidualBlock(nn.Module):
        """1D residual block: two conv layers + skip connection."""

        def __init__(self, channels: int, kernel_size: int = 3) -> None:
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
            self.bn1   = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
            self.bn2   = nn.BatchNorm1d(channels)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + residual)

    class _CNN1DNet(nn.Module):
        """
        3-stage 1D CNN with residual blocks.

        Input:  (batch, n_features, seq_len)   ← channels-first for Conv1d
        Output: (batch, 2)                     ← [logit_DOWN, logit_UP]
        """

        def __init__(self, n_features: int, seq_len: int, dropout: float = 0.3) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.stage1 = _ResidualBlock(64)
            self.down1  = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)

            self.stage2 = _ResidualBlock(128)
            self.down2  = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)

            self.stage3  = _ResidualBlock(256)
            self.pool    = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc1     = nn.Linear(256, 128)
            self.fc2     = nn.Linear(128, 2)

        def forward(self, x):
            # x: (B, seq_len, n_feat) — transpose to (B, n_feat, seq_len)
            x = x.transpose(1, 2)
            x = self.stem(x)
            x = self.stage1(x)
            x = F.relu(self.down1(x))
            x = self.stage2(x)
            x = F.relu(self.down2(x))
            x = self.stage3(x)
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class _LSTMNet(nn.Module):
        """
        Bidirectional LSTM.

        Input:  (batch, seq_len, n_features)
        Output: (batch, 2)
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 128,
            n_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True,
            )
            self.attn  = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.fc1   = nn.Linear(hidden_size * 2, 64)
            self.fc2   = nn.Linear(64, 2)

        def forward(self, x):
            out, _ = self.lstm(x)               # (B, T, 2H)
            # Attention pooling over time
            w = torch.softmax(self.attn(out), dim=1)
            ctx = (out * w).sum(dim=1)          # (B, 2H)
            ctx = self.dropout(ctx)
            ctx = F.relu(self.fc1(ctx))
            return self.fc2(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Public DeepModel class
# ─────────────────────────────────────────────────────────────────────────────

class DeepModel(BaseModel):
    """
    Deep-learning model that wraps a PyTorch CNN1D or LSTM (or sklearn MLP fallback).

    The model converts the flat 2D feature matrix into 3D rolling-window
    sequences before training/inference. This gives the network temporal
    context: each prediction is informed by the last ``sequence_length`` bars.

    Parameters (from settings.deep_learning)
    -----------------------------------------
    architecture   : cnn_1d | lstm          (default: cnn_1d)
    sequence_length: int                    (default: 20)
    epochs         : int                    (default: 50)
    batch_size     : int                    (default: 64)
    learning_rate  : float                  (default: 0.001)
    dropout        : float                  (default: 0.3)
    patience       : int  — early stopping  (default: 10)
    """

    def __init__(self, settings) -> None:
        cfg = settings.model.get("deep_learning", {})
        self._arch            = str(cfg.get("architecture", "cnn_1d")).lower()
        self._seq_len         = int(cfg.get("sequence_length", 20))
        self._epochs          = int(cfg.get("epochs", 50))
        self._batch_size      = int(cfg.get("batch_size", 64))
        self._lr              = float(cfg.get("learning_rate", 0.001))
        self._dropout         = float(cfg.get("dropout", 0.4))
        self._patience        = int(cfg.get("patience", 10))
        self._weight_decay    = float(cfg.get("weight_decay", 1e-4))
        self._label_smoothing = float(cfg.get("label_smoothing", 0.1))

        self._net       = None   # PyTorch nn.Module
        self._mlp       = None   # sklearn MLP fallback
        self._n_features: int = 0
        self._use_torch = _TORCH_AVAILABLE
        self._fitted    = False
        self._feature_means:  np.ndarray | None  = None
        self._feature_stds:   np.ndarray | None  = None
        self._feature_names_: list[str]  | None  = None  # stored at fit() time

    # ── BaseModel interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        backend = "PyTorch" if self._use_torch else "sklearn-MLP"
        return f"DeepModel({self._arch}, seq={self._seq_len}, {backend})"

    @property
    def feature_names(self) -> list[str] | None:
        """
        Feature column names used during training.

        Exposed so ``SignalGenerator`` (and any other caller) can align live
        feature matrices to the exact column set the model was trained on.
        Returns None if the model has not been fitted yet.
        """
        return self._feature_names_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_seconds: float | None = None,
        incremental: bool = False,
        groups: "np.ndarray | None" = None,
    ) -> None:
        """
        Train the deep-learning model.

        Internally converts the flat feature matrix to rolling-window sequences
        before fitting. Normalises features to zero-mean unit-variance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target.
        max_seconds : float | None
            Wall-clock time budget in seconds. Training stops after this many
            seconds even if ``epochs`` has not been reached.
            None (default) = use ``epochs`` from settings.
        incremental : bool
            If True AND the model was previously fitted (``self._net`` is set),
            continue training from the existing weights instead of starting
            from random initialisation.  Useful for extending a previous run.
        groups : np.ndarray | None
            Integer array (same length as X) identifying which ticker/group
            each row belongs to.  When provided, sequences are built only
            within the same group, preventing cross-ticker contamination.
        """
        # ── Store feature names for inference-time alignment ─────────────────
        if hasattr(X, "columns"):
            self._feature_names_ = list(X.columns)

        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int64)

        # Z-score normalise (store stats for inference)
        self._feature_means = X_arr.mean(axis=0)
        self._feature_stds  = X_arr.std(axis=0) + 1e-8
        X_norm = (X_arr - self._feature_means) / self._feature_stds

        self._n_features = X_norm.shape[1]

        # Build sequences — per-group when groups array is provided to prevent
        # cross-ticker sequence contamination.
        X_seq, y_seq = self._build_sequences(X_norm, y_arr, groups=groups)

        budget_str = f", budget={max_seconds:.0f}s" if max_seconds is not None else ""
        logger.info(
            "DeepModel training: arch=%s, sequences=%d, features=%d, seq_len=%d%s",
            self._arch, len(X_seq), self._n_features, self._seq_len, budget_str,
        )

        if self._use_torch:
            self._fit_torch(X_seq, y_seq, max_seconds=max_seconds, incremental=incremental)
        else:
            self._fit_sklearn_mlp(X_seq, y_seq, max_seconds=max_seconds)

        self._fitted = True
        logger.info("DeepModel training complete: %s", self.name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (n, 2): [P(DOWN), P(UP)]."""
        if not self._fitted:
            raise RuntimeError("DeepModel is not fitted.")

        # ── Feature alignment ─────────────────────────────────────────────
        # The live feature pipeline may produce a different number of columns
        # for US vs. French tickers (e.g. 133 vs 136).  Reindex to the exact
        # column set seen at training time: missing → 0.0, extra → dropped.
        if self._feature_names_ is not None:
            input_cols = list(X.columns) if hasattr(X, "columns") else []
            if input_cols != self._feature_names_:
                n_missing = len(set(self._feature_names_) - set(input_cols))
                n_extra   = len(set(input_cols) - set(self._feature_names_))
                if n_missing or n_extra:
                    logger.debug(
                        "predict_proba: realigning features — "
                        "input=%d cols, model=%d cols "
                        "(%d missing→0, %d extra dropped).",
                        len(input_cols), len(self._feature_names_),
                        n_missing, n_extra,
                    )
                X = X.reindex(columns=self._feature_names_, fill_value=0.0)

        X_arr = X.values.astype(np.float32)

        # ── Size-mismatch fallback for old checkpoints (no feature_names) ──
        # When feature_names_ is None the reindex above was skipped.  If the
        # raw array width still doesn't match the scaler, pad with zeros
        # (= training mean after Z-score) or truncate.  This keeps live
        # trading running while logging a clear call to retrain.
        if X_arr.shape[1] != self._n_features:
            logger.warning(
                "predict_proba: feature count mismatch — "
                "input=%d, model=%d.  "
                "Padding/truncating as stop-gap.  "
                "Retrain with --mode train --model-type deep to fix permanently.",
                X_arr.shape[1], self._n_features,
            )
            if X_arr.shape[1] < self._n_features:
                pad = np.zeros(
                    (len(X_arr), self._n_features - X_arr.shape[1]), dtype=np.float32
                )
                X_arr = np.hstack([X_arr, pad])
            else:
                X_arr = X_arr[:, : self._n_features]

        X_norm = (X_arr - self._feature_means) / self._feature_stds

        # Replace NaN / ±Inf with 0.0 (= training mean after Z-score).
        # NaN can appear when live_period is too short for long-lookback
        # features (e.g. monthly RSI needs 14 monthly bars).
        # Passing NaN to the CNN produces NaN logits → softmax([0,0]) = 0.5.
        n_nan = int(np.isnan(X_norm).sum())
        if n_nan > 0:
            logger.debug(
                "predict_proba: %d NaN values in normalised input — "
                "replaced with 0.0 (training mean).  "
                "Consider increasing live_period in settings.yaml.",
                n_nan,
            )
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        X_seq, _ = self._build_sequences(X_norm, np.zeros(len(X_norm), dtype=np.int64))

        if len(X_seq) == 0:
            # Not enough bars for a full sequence — return neutral 50/50
            return np.full((len(X_arr), 2), 0.5, dtype=np.float32)

        if self._use_torch:
            proba_seq = self._predict_torch(X_seq)
        else:
            proba_seq = self._predict_sklearn_mlp(X_seq)

        # The model produced one prediction per sequence (one per bar from seq_len onward).
        # Pad the first (seq_len - 1) rows with neutral 50/50 so the output aligns with X.
        n_pad = len(X_arr) - len(proba_seq)
        if n_pad > 0:
            pad = np.full((n_pad, 2), 0.5, dtype=np.float32)
            proba_seq = np.vstack([pad, proba_seq])

        return proba_seq

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arch":           self._arch,
            "seq_len":        self._seq_len,
            "n_features":     self._n_features,
            "feature_means":  self._feature_means,
            "feature_stds":   self._feature_stds,
            "feature_names":  self._feature_names_,
            "use_torch":      self._use_torch,
        }
        if self._use_torch and self._net is not None:
            payload["state_dict"] = self._net.state_dict()
            payload["net_class"]  = self._arch
        else:
            payload["mlp"] = self._mlp
        joblib.dump(payload, path)
        logger.info("Saved DeepModel to %s.", path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self._arch           = payload["arch"]
        self._seq_len        = payload["seq_len"]
        self._n_features     = payload["n_features"]
        self._feature_means  = payload["feature_means"]
        self._feature_stds   = payload["feature_stds"]
        self._feature_names_ = payload.get("feature_names")   # None for old checkpoints
        self._use_torch      = payload["use_torch"]

        if self._use_torch and "state_dict" in payload:
            self._net = self._build_torch_net(self._n_features)
            self._net.load_state_dict(payload["state_dict"])
            self._net.eval()
        else:
            self._mlp = payload.get("mlp")
        self._fitted = True
        logger.info("Loaded DeepModel from %s.", path)

    # ── Sequence builder ──────────────────────────────────────────────────────

    def _build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: "np.ndarray | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert (n, n_features) array into rolling-window sequences.

        When ``groups`` is provided, sequences are only built within the same
        group (ticker), preventing cross-ticker contamination.

        Returns
        -------
        X_seq : np.ndarray (n_sequences, seq_len, n_features)
        y_seq : np.ndarray (n_sequences,)
        """
        _empty = (
            np.empty((0, self._seq_len, X.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
        if len(X) < 1:
            return _empty

        if groups is None:
            # ── Single-group (flat) path ───────────────────────────────────
            n = len(X)
            if n < self._seq_len:
                return _empty
            X_seq = np.lib.stride_tricks.sliding_window_view(X, (self._seq_len, X.shape[1]))
            X_seq = X_seq[:, 0, :, :]   # (n - seq_len + 1, seq_len, n_features)
            y_seq = y[self._seq_len - 1:]
            return X_seq.astype(np.float32), y_seq.astype(np.int64)

        # ── Per-group path (prevents cross-ticker contamination) ───────────
        all_X, all_y = [], []
        for gid in np.unique(groups):
            mask = groups == gid
            X_g, y_g = X[mask], y[mask]
            if len(X_g) < self._seq_len:
                continue
            Xs = np.lib.stride_tricks.sliding_window_view(X_g, (self._seq_len, X_g.shape[1]))
            Xs = Xs[:, 0, :, :]
            ys = y_g[self._seq_len - 1:]
            all_X.append(Xs.astype(np.float32))
            all_y.append(ys.astype(np.int64))

        if not all_X:
            return _empty
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    # ── PyTorch training / inference ──────────────────────────────────────────

    def _build_torch_net(self, n_features: int):
        if self._arch == "lstm":
            return _LSTMNet(n_features, dropout=self._dropout)
        return _CNN1DNet(n_features, self._seq_len, dropout=self._dropout)

    def _fit_torch(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        max_seconds: float | None = None,
        incremental: bool = False,
    ) -> None:
        import time as _time
        deadline = (_time.monotonic() + max_seconds) if max_seconds is not None else None

        device = _get_device()
        net    = self._build_torch_net(self._n_features)

        # ── Incremental: load previous weights ────────────────────────────
        if incremental and self._net is not None:
            try:
                net.load_state_dict(self._net.state_dict())
                logger.info("Incremental training: loaded previous model weights.")
            except Exception as exc:
                logger.warning("Incremental load failed (%s) — starting fresh.", exc)

        net = net.to(device)
        logger.info(
            "DeepModel using device: %s",
            str(device).upper() if hasattr(device, "__str__") else type(device).__name__,
        )

        X_t = torch.from_numpy(X_seq)
        y_t = torch.from_numpy(y_seq)

        # Class weights for imbalanced targets
        n_neg = int((y_seq == 0).sum())
        n_pos = int((y_seq == 1).sum())
        pos_weight = torch.tensor(n_neg / (n_pos + 1e-8), dtype=torch.float32).to(device)
        criterion  = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(pos_weight)]).to(device),
            label_smoothing=self._label_smoothing,   # reduces overconfident predictions
        )
        # DirectML (AMD on Windows) has limited kernel support — use the scalar
        # Adam path and suppress the known DML CPU-fallback warning which is
        # harmless (only the EMA lerp falls back, the heavy conv ops stay on GPU).
        _is_dml = "privateuseone" in str(device).lower()
        optimiser = torch.optim.Adam(
            net.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,          # L2 regularisation
            foreach=False if _is_dml else None,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self._epochs)
        if _is_dml:
            warnings.filterwarnings(
                "ignore",
                message=".*aten::lerp.*DML.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*not currently supported on the DML backend.*",
                category=UserWarning,
            )

        # Train / validation split (last 20% as val)
        n_val  = max(1, int(len(X_t) * 0.2))
        n_train = len(X_t) - n_val
        train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
        val_ds   = TensorDataset(X_t[n_train:], y_t[n_train:])
        train_dl = DataLoader(train_ds, batch_size=self._batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self._batch_size, shuffle=False)

        # When a time budget is given, allow up to a very large number of epochs
        # (early stopping + deadline will cut it off at the right time).
        max_epochs = self._epochs if deadline is None else max(self._epochs, 100_000)

        best_val_loss  = float("inf")
        first_val_loss = None   # track improvement from epoch 1
        patience_ctr   = 0
        best_state     = None
        _interrupted_at: int | None = None  # set if Ctrl+C is pressed

        try:
         for epoch in range(1, max_epochs + 1):
            # ── Time-budget check ──────────────────────────────────────────
            if deadline is not None and _time.monotonic() >= deadline:
                logger.info("  Time budget reached at epoch %d.", epoch - 1)
                break

            # ── Train ──────────────────────────────────────────────────────
            net.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item() * len(xb)
            train_loss /= n_train
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(net(xb), yb).item() * len(xb)
            val_loss /= n_val

            if epoch % 10 == 0:
                elapsed = f"{_time.monotonic() - (deadline - max_seconds):.0f}s" if deadline is not None else ""
                logger.info(
                    "  Epoch %d — train_loss=%.4f, val_loss=%.4f%s",
                    epoch, train_loss, val_loss,
                    f"  [{elapsed} elapsed]" if elapsed else "",
                )

            if first_val_loss is None:
                first_val_loss = val_loss

            # ── Early stopping ─────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_ctr  = 0
                best_state    = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self._patience:
                    time_left = (deadline - _time.monotonic()) if deadline is not None else 0
                    if time_left > 2.0:
                        # Time budget remains — restart from best state with halved LR
                        if best_state is not None:
                            net.load_state_dict(best_state)
                        for pg in optimiser.param_groups:
                            pg["lr"] = max(pg["lr"] * 0.5, 1e-6)
                        patience_ctr = 0
                        # Reset scheduler for next cosine cycle
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimiser, T_max=max(self._epochs, 50)
                        )
                        logger.info(
                            "  Patience exhausted — restarting (LR=%.2e, %.0f s left).",
                            optimiser.param_groups[0]["lr"], time_left,
                        )
                    else:
                        improvement = first_val_loss - best_val_loss
                        pct = (improvement / (first_val_loss + 1e-8)) * 100
                        logger.info(
                            "  Early stopping at epoch %d — best val_loss=%.4f "
                            "(improved %.4f = %.1f%% from epoch 1).",
                            epoch, best_val_loss, improvement, pct,
                        )
                        break

        except KeyboardInterrupt:
            _interrupted_at = epoch
            print(f"\n  ⚠️  Training interrupted at epoch {epoch}.")
            print("  💾  Restoring best checkpoint found so far...")

            # Restore + assign net before re-raising so the caller can save it
            if best_state is not None:
                net.load_state_dict(best_state)
                print(
                    f"  ✅  Best weights restored "
                    f"(val_loss={best_val_loss:.4f})."
                )
            self._net     = net.cpu()
            self._net.eval()
            self._fitted  = True   # mark fitted so run_train can save
            print(
                "  💡  Best checkpoint will be saved — "
                "no progress lost. Resume training anytime.\n"
            )
            raise   # propagate → stops walk-forward folds + run_train in one Ctrl+C

        # Completed normally — restore best weights
        if best_state is not None:
            net.load_state_dict(best_state)

        # ── Final learning summary ─────────────────────────────────────────
        if first_val_loss is not None and best_val_loss < float("inf"):
            improvement = first_val_loss - best_val_loss
            pct = (improvement / (first_val_loss + 1e-8)) * 100
            learned = "✓ Model learned" if improvement > 0.01 else "~ Minimal improvement"
            logger.info(
                "%s — val_loss: %.4f → %.4f (Δ=%.4f, %.1f%% improvement).",
                learned, first_val_loss, best_val_loss, improvement, pct,
            )

        self._net = net.cpu()
        self._net.eval()

    def _predict_torch(self, X_seq: np.ndarray) -> np.ndarray:
        device = _get_device()
        self._net.to(device).eval()
        X_t    = torch.from_numpy(X_seq).to(device)
        chunks = []
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                logits = self._net(X_t[i:i + 256])
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        self._net.cpu()
        return np.vstack(chunks).astype(np.float32)

    # ── sklearn MLP fallback ──────────────────────────────────────────────────

    def _fit_sklearn_mlp(self, X_seq: np.ndarray, y_seq: np.ndarray, max_seconds: float | None = None) -> None:
        from sklearn.neural_network import MLPClassifier

        # Scale epochs by time budget if given (rough 1s/10 iter estimate)
        max_iter = self._epochs if max_seconds is None else max(self._epochs, int(max_seconds * 10))

        # Flatten sequences to 2D for sklearn
        X_flat = X_seq.reshape(len(X_seq), -1)
        self._mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=self._lr,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=self._patience,
            random_state=42,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._mlp.fit(X_flat, y_seq)
        logger.info("sklearn MLP training complete (%d max_iter).", max_iter)

    def _predict_sklearn_mlp(self, X_seq: np.ndarray) -> np.ndarray:
        X_flat = X_seq.reshape(len(X_seq), -1)
        return self._mlp.predict_proba(X_flat).astype(np.float32)
