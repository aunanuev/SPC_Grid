"""Predicción de deciles de retornos semanales (PDF sección 2.3).

Implementa el ciclo completo del predictor cuantílico:
- Carga de retornos SPX/CMC200 desde `data/`.
- Ventanas deslizantes (X_t = últimos H retornos, Y_t = retorno en t+1).
- Split cronológico train/valid/test sin shuffle (para evitar fuga temporal).
- Estandarizador ajustado sólo con train.
- Arquitectura `QuantileLSTM`: LSTM + cabeza densa que emite un retorno por
  (activo, decil) — dimensión (B, n_assets, n_deciles).
- `pinball_loss` (PDF ec. 14) como función objetivo de entrenamiento.
- `train_deciles`: entrena con seed averaging (mejor semilla por pinball de
  validación con early stopping).
- `save_checkpoint` / `load_checkpoint` + `predict_deciles(_batch)` para la
  etapa de inferencia que alimenta `regimen_predicted` y `generador_escenarios`.

Uso típico:
    config = DLConfig()
    result = train_deciles(config)
    save_checkpoint(result, MODELS_DIR / "decile_predictor.pt")

    model = load_checkpoint(MODELS_DIR / "decile_predictor.pt")
    deciles = predict_deciles(model, last_window)   # {asset: {q: r_hat}}
"""

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import ASSETS, DATA_DIR, DLConfig, MODELS_DIR, RETURN_CSV, RETURN_COL


# =====================================================================
# Datos: carga, ventanas, split, estandarización
# =====================================================================

@dataclass
class ChronoSplit:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_valid: np.ndarray
    Y_valid: np.ndarray
    X_test:  np.ndarray
    Y_test:  np.ndarray
    t_train: np.ndarray
    t_valid: np.ndarray
    t_test:  np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray
    std:  np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def load_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Carga retornos semanales de `ASSETS` en un DataFrame indexado por t."""
    data_dir = Path(data_dir)
    merged: pd.DataFrame | None = None
    for asset in ASSETS:
        df = pd.read_csv(data_dir / RETURN_CSV[asset])
        df.columns = [c.strip() for c in df.columns]
        df["t"] = df["t"].astype(int)
        df = df.rename(columns={RETURN_COL[asset]: asset})[["t", asset]]
        merged = df if merged is None else pd.merge(merged, df, on="t")
    return merged.sort_values("t").set_index("t")[list(ASSETS)]


def build_windows(returns: pd.DataFrame, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas deslizantes: X[n] = retornos en [t-H+1, t], Y[n] = retorno en t+1."""
    arr = returns.to_numpy(dtype=np.float32)
    T, A = arr.shape
    N = T - H
    if N <= 0:
        raise ValueError(f"No hay suficientes periodos (T={T}) para H={H}.")

    X = np.empty((N, H, A), dtype=np.float32)
    Y = np.empty((N, A),    dtype=np.float32)
    t_idx = np.empty(N, dtype=np.int64)
    t_vals = returns.index.to_numpy()

    for n in range(N):
        X[n] = arr[n : n + H]
        Y[n] = arr[n + H]
        t_idx[n] = t_vals[n + H]
    return X, Y, t_idx


def chrono_split(
    X: np.ndarray, Y: np.ndarray, t_idx: np.ndarray,
    ratios: Tuple[float, float, float],
) -> ChronoSplit:
    """Split cronológico según `ratios` (train, valid, test); sin shuffle."""
    r_tr, r_va, _ = ratios
    N = len(X)
    n_tr = int(N * r_tr)
    n_va = int(N * r_va)
    return ChronoSplit(
        X_train=X[:n_tr],              Y_train=Y[:n_tr],
        X_valid=X[n_tr : n_tr + n_va], Y_valid=Y[n_tr : n_tr + n_va],
        X_test= X[n_tr + n_va :],      Y_test= Y[n_tr + n_va :],
        t_train=t_idx[:n_tr],
        t_valid=t_idx[n_tr : n_tr + n_va],
        t_test= t_idx[n_tr + n_va :],
    )


def fit_standardizer(X_train: np.ndarray) -> Standardizer:
    """Media y std por activo sobre el conjunto de entrenamiento."""
    mean = X_train.mean(axis=(0, 1)).astype(np.float32)
    std  = X_train.std(axis=(0, 1)).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return Standardizer(mean=mean, std=std)


# =====================================================================
# Modelo: LSTM cuantílica + pinball loss
# =====================================================================

def pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """Pinball loss (PDF ec. 14) promediada sobre (batch, activos, deciles)."""
    q = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device)
    q = q.view(1, 1, -1)
    e = y_true.unsqueeze(-1) - y_pred
    return torch.maximum(q * e, (q - 1.0) * e).mean()


class QuantileLSTM(nn.Module):
    """LSTM sobre la ventana temporal; cabeza densa produce deciles por activo."""

    def __init__(self, config: DLConfig):
        super().__init__()
        self.config = config
        A = config.n_assets
        Q = config.n_quantiles

        self.lstm = nn.LSTM(
            input_size=A,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head    = nn.Linear(config.lstm_hidden, A * Q)
        self.A = A
        self.Q = Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, A)
        out, _ = self.lstm(x)                   # (B, H, hidden)
        last   = self.dropout(out[:, -1, :])    # (B, hidden)
        head   = self.head(last)                # (B, A*Q)
        return head.view(-1, self.A, self.Q)    # (B, A, Q)


# =====================================================================
# Entrenamiento (seed averaging)
# =====================================================================

@dataclass
class TrainResult:
    state_dict: Dict
    config:     DLConfig
    mean:       np.ndarray
    std:        np.ndarray
    history:    Dict[str, List[float]] = field(default_factory=dict)
    best_seed:  int = 0
    best_valid: float = float("inf")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one(
    config: DLConfig,
    split: ChronoSplit,
    scaler: Standardizer,
    seed: int,
) -> TrainResult:
    _seed_all(seed)
    device = torch.device(config.device)

    X_tr = torch.tensor(scaler.apply(split.X_train), dtype=torch.float32, device=device)
    Y_tr = torch.tensor(split.Y_train,               dtype=torch.float32, device=device)
    X_va = torch.tensor(scaler.apply(split.X_valid), dtype=torch.float32, device=device)
    Y_va = torch.tensor(split.Y_valid,               dtype=torch.float32, device=device)

    model = QuantileLSTM(config).to(device)
    # Excluir bias del weight_decay: el offset del head linear no debe ser
    # atraido a 0; si la regularizacion lo penaliza, la mediana predicha queda
    # sesgada hacia 0 cuando la media historica del retorno es no-nula.
    decay    = [p for n, p in model.named_parameters() if "bias" not in n]
    no_decay = [p for n, p in model.named_parameters() if "bias" in n]
    optim = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr,
    )

    best_valid    = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    history       = {"train": [], "valid": []}
    patience_left = config.patience
    B             = config.batch_size or len(X_tr)

    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, len(X_tr), B):
            idx = perm[i : i + B]
            optim.zero_grad()
            loss = pinball_loss(model(X_tr[idx]), Y_tr[idx], config.quantiles)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches  += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            valid_loss = pinball_loss(model(X_va), Y_va, config.quantiles).item()

        history["train"].append(train_loss)
        history["valid"].append(valid_loss)

        if valid_loss < best_valid - 1e-6:
            best_valid    = valid_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return TrainResult(
        state_dict=best_state, config=config,
        mean=scaler.mean, std=scaler.std,
        history=history, best_seed=seed, best_valid=best_valid,
    )


def train_deciles(config: DLConfig) -> TrainResult:
    """Entrena el LSTM con seed averaging; retorna el mejor modelo por pinball-valid."""
    df_ret = load_returns()
    X, Y, t_idx = build_windows(df_ret, config.H)
    split  = chrono_split(X, Y, t_idx, config.split)
    scaler = fit_standardizer(split.X_train)

    best: TrainResult | None = None
    for seed in config.seeds:
        r = _train_one(config, split, scaler, seed)
        print(f"  seed={seed}  best_valid={r.best_valid:.6f}")
        if best is None or r.best_valid < best.best_valid:
            best = r
    assert best is not None
    return best


# =====================================================================
# Persistencia e inferencia
# =====================================================================

@dataclass
class LoadedModel:
    nets:     List[QuantileLSTM]
    config:   DLConfig
    mean:     np.ndarray
    std:      np.ndarray


def save_checkpoint(result: TrainResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": result.state_dict,
        "config":     result.config,
        "mean":       result.mean,
        "std":        result.std,
        "history":    result.history,
        "best_seed":  result.best_seed,
        "best_valid": result.best_valid,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path | str) -> LoadedModel:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config: DLConfig = payload["config"]
    # Ensemble: usa state_dicts (lista) si existe; si no, fallback a state_dict singular.
    state_dicts = payload.get("state_dicts") or [payload["state_dict"]]
    nets: List[QuantileLSTM] = []
    for sd in state_dicts:
        net = QuantileLSTM(config)
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return LoadedModel(
        nets=nets, config=config,
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"],  dtype=np.float32),
    )


def predict_deciles(
    model: LoadedModel, window: np.ndarray, sort: bool = True,
) -> Dict[str, Dict[float, float]]:
    """
    window: (H, n_assets)  ->  {asset: {q: r_hat}} con un retorno predicho por decil.
    Si sort=True (default) se ordenan los deciles por activo para garantizar
    monotonicidad; sort=False devuelve la salida cruda de la red.
    """
    cfg = model.config
    if window.shape != (cfg.H, cfg.n_assets):
        raise ValueError(f"window shape {window.shape} != (H={cfg.H}, A={cfg.n_assets})")

    x = ((window.astype(np.float32) - model.mean) / model.std)
    x = torch.from_numpy(x).unsqueeze(0)                               # (1, H, A)
    with torch.no_grad():
        outs = [net(x).numpy()[0] for net in model.nets]               # K * (A, Q)
    out = np.mean(np.stack(outs, axis=0), axis=0)                      # ensemble
    if sort:
        out = np.sort(out, axis=-1)

    return {
        asset: {float(q): float(out[ai, qi]) for qi, q in enumerate(cfg.quantiles)}
        for ai, asset in enumerate(cfg.assets)
    }


def predict_deciles_batch(
    model: LoadedModel, windows: np.ndarray, sort: bool = True,
) -> np.ndarray:
    """windows: (N, H, n_assets)  ->  (N, n_assets, n_deciles).

    Si sort=True (default) se ordenan los deciles a lo largo del eje Q para que
    sean monotonicos crecientes por (escenario, activo). sort=False devuelve la
    salida cruda — util para diagnosticar cuantas veces la red viola la
    monotonicidad antes de corregir.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows debe tener shape (N, H, A); recibí {windows.shape}")
    x = ((windows.astype(np.float32) - model.mean) / model.std)
    x_tensor = torch.from_numpy(x)
    with torch.no_grad():
        outs = [net(x_tensor).numpy() for net in model.nets]           # K * (N, A, Q)
    out = np.mean(np.stack(outs, axis=0), axis=0)                      # ensemble
    if sort:
        out = np.sort(out, axis=-1)
    return out


# =====================================================================
# Visualización: fan chart de deciles vs realizado
# =====================================================================

def plot_fan_chart(
    model: LoadedModel,
    X: np.ndarray,
    Y: np.ndarray,
    t_idx: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = False,
    title_suffix: str = "test",
) -> None:
    """
    Fan chart: bandas por pares de deciles (q, 1-q), mediana y retorno realizado.

    X:     (N, H, A)   ventanas de entrada (normalmente el split de test)
    Y:     (N, A)      retornos realizados correspondientes
    t_idx: (N,)        índice temporal para el eje X
    out_path: si no es None, guarda la figura en disco
    show:  si True, abre ventana interactiva (bloquea hasta cerrarla)
    """
    preds = predict_deciles_batch(model, X)                            # (N, A, Q)
    cfg   = model.config
    Q     = cfg.n_quantiles

    fig, axes = plt.subplots(
        cfg.n_assets, 1, figsize=(10, 3 * cfg.n_assets), sharex=True,
    )
    if cfg.n_assets == 1:
        axes = [axes]

    cmap = plt.get_cmap("Blues")
    for ai, asset in enumerate(cfg.assets):
        ax = axes[ai]
        # Bandas: une cada par (q, 1-q), de los extremos hacia la mediana.
        for qi in range(Q // 2):
            lo, hi = qi, Q - 1 - qi
            alpha  = 0.25 + 0.4 * qi / max(Q // 2 - 1, 1)
            ax.fill_between(
                t_idx, preds[:, ai, lo], preds[:, ai, hi],
                color=cmap(0.3 + 0.5 * qi / max(Q // 2, 1)),
                alpha=alpha, linewidth=0,
                label=f"q{int(cfg.quantiles[lo]*100):02d}–q{int(cfg.quantiles[hi]*100):02d}",
            )
        ax.plot(t_idx, preds[:, ai, Q // 2], color="#1f3b73",
                linewidth=1.2, label="q50 (mediana)")
        ax.plot(t_idx, Y[:, ai], color="#E63946",
                linewidth=1.0, alpha=0.9, label="realizado")
        ax.set_title(f"Fan chart ({title_suffix}) — {asset}")
        ax.set_ylabel("retorno semanal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("t")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[viz] fan chart guardado en: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
# Punto de entrada: `python -m dl.prediccion_deciles`
# =====================================================================

if __name__ == "__main__":
    config = DLConfig()
    print(f"[train] H={config.H}  seeds={config.seeds}  deciles={config.n_quantiles}")
    print(f"[train] split cronologico train/valid/test = {config.split}")

    result = train_deciles(config)
    ckpt = MODELS_DIR / "decile_predictor.pt"
    save_checkpoint(result, ckpt)
    print(f"\n[train] best_valid={result.best_valid:.6f}  "
          f"seed={result.best_seed}  epochs={len(result.history['train'])}")
    print(f"[train] guardado en: {ckpt}")

    # Fan chart sobre el split de test (out-of-sample cronologico).
    df_ret = load_returns()
    X, Y, t_idx = build_windows(df_ret, config.H)
    split = chrono_split(X, Y, t_idx, config.split)
    loaded = load_checkpoint(ckpt)
    plot_fan_chart(
        loaded, split.X_test, split.Y_test, split.t_test,
        out_path=DATA_DIR / "fan_chart_test.png",
        show=True,
        title_suffix="test",
    )
