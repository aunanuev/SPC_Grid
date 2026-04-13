"""
Regret-Grid Calibration con Predicción Deep Learning (LSTM)
para el Modelo de Portafolio Dinámico.

Pipeline completo:
  1. Cargar datos y construir dataset supervisado
  2. Entrenar LSTM para predicción de quantiles (pinball loss)
  3. Generar N escenarios candidatos con el LSTM congelado
  4. Reducir a 5 escenarios representativos (quintiles)
  5. Ejecutar regret-grid: 15 combinaciones (λ, m) × 5 escenarios
  6. Seleccionar g* por average regret y worst-case regret
  7. Re-ejecutar el optimizador con g* y comparar vs benchmarks

Referencia: "Calibración tipo Regret-Grid con Predicción Deep Learning
             para un Modelo de Portafolio en GAMS" (Juan Pérez)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from basemodelGAMS import (
    load_market_data,
    solve_portfolio,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
)

# ================================================================
# Configuración global
# ================================================================

QUANTILE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
H = 20           # ventana lookback (semanas)
HIDDEN_SIZE = 32  # tamaño estado oculto LSTM
N_LAYERS = 1
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 500
PATIENCE = 30     # early stopping
N_SCENARIOS = 1000
N_QUINTILES = 5
SEED = 42


# ================================================================
# 1) Dataset supervisado
# ================================================================

class QuantileDataset(Dataset):
    """
    Dataset supervisado para predicción de quantiles.
    Input:  últimos H retornos de cada activo  → (H, n_assets)
    Target: retorno siguiente de cada activo   → (n_assets,)
    """

    def __init__(self, returns_array: np.ndarray, lookback: int = H):
        """
        returns_array: shape (T, n_assets), retornos ordenados temporalmente.
        """
        self.X = []
        self.Y = []
        for t in range(lookback, len(returns_array)):
            self.X.append(returns_array[t - lookback : t])
            self.Y.append(returns_array[t])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_datasets(context, lookback=H):
    """
    Construye train/valid/test a partir de los retornos del contexto.
    Split cronológico: 70% train, 15% valid, 15% test.
    """
    T_vals = context["T_vals"]
    assets = context["assets"]
    r = context["r"]

    returns = np.array([[r[a].loc[t] for a in assets] for t in T_vals])

    n_samples = len(returns) - lookback
    n_train = int(n_samples * 0.70)
    n_valid = int(n_samples * 0.15)

    # Offset: empezamos desde lookback para que haya ventana completa
    cut_train = lookback + n_train
    cut_valid = cut_train + n_valid

    train_ds = QuantileDataset(returns[:cut_train], lookback)
    valid_ds = QuantileDataset(returns[:cut_valid], lookback)
    # valid solo usa las muestras nuevas (después de train)
    # pero el Dataset incluye las de train porque necesita lookback;
    # lo manejamos correctamente: valid_ds incluye más samples,
    # simplemente restamos las de train para tener solo las nuevas.
    # Mejor: crear slices exactos.

    # Reconstruir de forma limpia:
    train_returns = returns[:cut_train]
    valid_returns = returns[:cut_valid]
    test_returns  = returns  # todas

    train_ds = QuantileDataset(train_returns, lookback)
    valid_ds = QuantileDataset(valid_returns, lookback)
    test_ds  = QuantileDataset(test_returns,  lookback)

    # Ahora valid_ds tiene len = cut_valid - lookback, train_ds tiene len = cut_train - lookback
    # Las muestras de validación puras son las últimas (len(valid_ds) - len(train_ds))
    # Para simplificar, usamos SubsetDataLoader en el training loop.

    return train_ds, valid_ds, test_ds, returns, n_train


# ================================================================
# 2) Modelo LSTM para quantiles
# ================================================================

class QuantileLSTM(nn.Module):
    """
    LSTM que predice |Q| quantiles para cada activo.
    Input:  (batch, H, n_assets)
    Output: (batch, n_assets, n_quantiles)
    """

    def __init__(self, n_assets=2, n_quantiles=5,
                 hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_assets * n_quantiles)
        self.n_assets = n_assets
        self.n_quantiles = n_quantiles

    def forward(self, x):
        # x: (batch, H, n_assets)
        out, _ = self.lstm(x)
        last = out[:, -1, :]              # (batch, hidden)
        last = self.dropout(last)
        pred = self.head(last)            # (batch, n_assets * n_quantiles)
        pred = pred.view(-1, self.n_assets, self.n_quantiles)
        # Ordenar quantiles para garantizar monotonía
        pred, _ = torch.sort(pred, dim=-1)
        return pred


# ================================================================
# 3) Pinball loss y entrenamiento
# ================================================================

def pinball_loss(predictions, targets, quantile_levels):
    """
    Pinball (quantile regression) loss.
    predictions: (batch, n_assets, n_quantiles)
    targets:     (batch, n_assets)
    quantile_levels: lista de floats [0.1, 0.3, 0.5, 0.7, 0.9]
    """
    q = torch.tensor(quantile_levels, dtype=torch.float32,
                     device=predictions.device)
    # targets → (batch, n_assets, 1) para broadcast
    y = targets.unsqueeze(-1)
    errors = y - predictions                       # (batch, n_assets, n_quantiles)
    loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
    return loss.mean()


def train_quantile_model(train_ds, valid_ds, n_train,
                         n_assets=2, lr=LR, epochs=EPOCHS, patience=PATIENCE,
                         verbose=True):
    """
    Entrena el LSTM con pinball loss y early stopping en validación.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = QuantileLSTM(n_assets=n_assets)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Para validación, usamos solo las muestras posteriores a train
    n_valid_total = len(valid_ds)
    valid_indices = list(range(n_train, n_valid_total))
    if len(valid_indices) == 0:
        valid_indices = list(range(n_train - 5, n_valid_total))

    valid_subset = torch.utils.data.Subset(valid_ds, valid_indices)
    valid_loader = DataLoader(valid_subset, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = pinball_loss(pred, yb, QUANTILE_LEVELS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                pred = model(xb)
                loss = pinball_loss(pred, yb, QUANTILE_LEVELS)
                val_loss_sum += loss.item()
                n_val += 1

        avg_train = train_loss_sum / max(n_batches, 1)
        avg_val = val_loss_sum / max(n_val, 1)

        if verbose and (epoch <= 5 or epoch % 50 == 0 or epoch == epochs):
            print(f"  Epoch {epoch:>3d}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping en epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  Mejor val_loss = {best_val_loss:.6f}")
    return model


# ================================================================
# 4) De quantiles a probabilidades de régimen
# ================================================================

def quantiles_to_regime_probs(quantile_preds):
    """
    Convierte predicciones de quantiles a probabilidades bear/bull.

    quantile_preds: (T, n_assets, n_quantiles)
    Retorna: dict con p_bull[asset][t] y p_bear[asset][t]

    p_bull = (1/|Q|) * sum_q 1{r_hat^(q) >= 0}
    p_bear = 1 - p_bull
    """
    n_quantiles = quantile_preds.shape[2]
    # Fracción de quantiles >= 0
    p_bull = (quantile_preds >= 0).sum(axis=2).astype(float) / n_quantiles
    # Clip para evitar 0.0 o 1.0 exactos
    p_bull = np.clip(p_bull, 0.05, 0.95)
    p_bear = 1.0 - p_bull
    return p_bull, p_bear


# ================================================================
# 5) Generación de escenarios con LSTM
# ================================================================

@torch.no_grad()
def generate_scenarios(model, initial_window, T, N=N_SCENARIOS, seed=SEED):
    """
    Genera N escenarios candidatos de longitud T.

    model: LSTM congelado
    initial_window: (H, n_assets) — últimos H retornos observados
    T: horizonte de simulación
    N: número de escenarios a generar

    Para cada escenario n:
      - Partir de la ventana observada
      - En cada t: predecir 5 quantiles, samplear uno uniformemente,
        usarlo como retorno, actualizar ventana.

    Retorna: (N, T, n_assets)
    """
    rng = np.random.RandomState(seed)
    model.eval()
    n_assets = initial_window.shape[1]
    n_q = len(QUANTILE_LEVELS)

    scenarios = np.zeros((N, T, n_assets))

    for n in range(N):
        window = initial_window.copy()  # (H, n_assets)
        for t in range(T):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, H, n_assets)
            pred = model(x)  # (1, n_assets, n_quantiles)
            pred_np = pred.squeeze(0).numpy()  # (n_assets, n_quantiles)

            # Samplear un quantile uniformemente para cada activo
            q_idx = rng.randint(0, n_q, size=n_assets)
            sampled = np.array([pred_np[a, q_idx[a]] for a in range(n_assets)])

            scenarios[n, t, :] = sampled
            # Actualizar ventana: shift left + append
            window = np.vstack([window[1:], sampled.reshape(1, -1)])

    return scenarios


# ================================================================
# 6) Reducción a quintiles
# ================================================================

def reduce_to_quintiles(scenarios, n_quintiles=N_QUINTILES):
    """
    Reduce N escenarios a n_quintiles representativos.

    1. Calcular retorno acumulado SPX por escenario
    2. Ordenar por este resumen
    3. Dividir en quintiles
    4. Tomar el escenario mediano de cada quintil

    scenarios: (N, T, n_assets) — SPX es el asset index 0
    Retorna: (n_quintiles, T, n_assets)
    """
    N = scenarios.shape[0]
    # Retorno acumulado del SPX (asset 0)
    cum_ret = np.prod(1 + scenarios[:, :, 0], axis=1) - 1  # (N,)

    order = np.argsort(cum_ret)
    quintile_size = N // n_quintiles

    selected = []
    for q in range(n_quintiles):
        start = q * quintile_size
        end = start + quintile_size if q < n_quintiles - 1 else N
        group = order[start:end]
        # Mediana: tomar el escenario central del grupo
        median_idx = group[len(group) // 2]
        selected.append(scenarios[median_idx])

    result = np.array(selected)  # (n_quintiles, T, n_assets)

    # Imprimir resumen
    for q in range(n_quintiles):
        cr = np.prod(1 + result[q, :, 0]) - 1
        print(f"    Quintil {q+1}: ret_acum_SPX = {cr:+.4f}")

    return result


# ================================================================
# 7) Simulación ex-post sobre escenarios
# ================================================================

def simulate_capital_scenario(w_sol, scenario_returns, context):
    """
    Simulación ex-post de capital usando retornos del escenario s
    y pesos w^g del optimizador.

    Fórmula (eq. 19 del PDF):
      x_{t+1} = x_t (1 + Σ_i w^g_{i,t} r^s_{i,t})
              - x_t Σ_i c_i |w^g_{i,t+1} - w^g_{i,t}|

    Nota: usa |Δw| (turnover absoluto) para costos, NO u+v del optimizador.

    w_sol: dict {(asset, t): weight}
    scenario_returns: (T, n_assets)
    context: dict con assets, T_vals, c_base, Capital_inicial
    """
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    c_base  = context["c_base"]
    Capital = context["Capital_inicial"]
    nT = len(T_vals)

    cap = Capital
    for idx in range(nT - 1):
        t      = T_vals[idx]
        t_next = T_vals[idx + 1]

        # Retorno del portafolio en este periodo (usando retornos del escenario)
        r_port = sum(
            w_sol[assets[a], t] * scenario_returns[idx, a]
            for a in range(len(assets))
        )

        # Turnover absoluto (|w_{t+1} - w_t|)
        turnover_cost = sum(
            c_base[assets[a]] * abs(w_sol[assets[a], t_next] - w_sol[assets[a], t])
            for a in range(len(assets))
        )

        cap = cap * (1 + r_port) - cap * turnover_cost

    return cap


# ================================================================
# 8) Regret-Grid
# ================================================================

def run_regret_grid(scenarios, context):
    """
    Ejecuta el regret-grid: para cada punto g = (λ, m) del grid,
    optimiza con el modelo GAMS y simula sobre cada escenario.

    scenarios: (n_quintiles, T, n_assets)
    context: dict de load_market_data

    Retorna: DataFrame con resultados y g* seleccionado.
    """
    assets = context["assets"]
    theta_neutral = {a: 1.0 for a in assets}

    lambda_grid = [0.05, 0.10, 0.20, 0.50, 1.00]
    c_mult_grid = [0.5, 1.0, 2.0]
    labels_L = ["L1", "L2", "L3", "L4", "L5"]
    labels_C = ["C1", "C2", "C3"]

    n_g = len(lambda_grid) * len(c_mult_grid)
    n_s = scenarios.shape[0]

    # Matriz V[g, s] = capital terminal
    V = np.zeros((n_g, n_s))
    grid_points = []

    g_idx = 0
    for li, lam in enumerate(lambda_grid):
        for ci, cm in enumerate(c_mult_grid):
            label = f"{labels_L[li]}/{labels_C[ci]}"
            print(f"  [{g_idx+1:>2}/{n_g}] {label}  lambda={lam:.2f}  m={cm:.1f} ...",
                  end=" ", flush=True)

            # Optimizar (una vez por g)
            z, w_sol, u_sol, v_sol, status = solve_portfolio(
                theta_neutral, context,
                lambda_riesgo=lam, costo_mult=cm,
            )

            # Simular sobre cada escenario
            for s in range(n_s):
                V[g_idx, s] = simulate_capital_scenario(
                    w_sol, scenarios[s], context
                )

            avg_v = V[g_idx, :].mean()
            print(f"z={z:.6f}  avg_cap={avg_v:,.2f}")

            grid_points.append({
                "g_idx": g_idx, "L": labels_L[li], "C": labels_C[ci],
                "lambda": lam, "c_mult": cm, "z": z,
            })
            g_idx += 1

    # --- Regret ---
    V_best = V.max(axis=0)       # (n_s,) — mejor capital por escenario
    R = V_best - V               # (n_g, n_s) — regret matrix

    avg_regret = R.mean(axis=1)  # (n_g,)
    max_regret = R.max(axis=1)   # (n_g,)

    g_star_avg = int(np.argmin(avg_regret))
    g_star_wc  = int(np.argmin(max_regret))

    # Construir DataFrame de resultados
    rows = []
    for g in range(n_g):
        row = grid_points[g].copy()
        for s in range(n_s):
            row[f"V_s{s+1}"] = round(V[g, s], 2)
            row[f"R_s{s+1}"] = round(R[g, s], 2)
        row["avg_regret"] = round(avg_regret[g], 2)
        row["max_regret"] = round(max_regret[g], 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, g_star_avg, g_star_wc, grid_points, V, R


# ================================================================
# 9) Bloque principal
# ================================================================

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  REGRET-GRID CON PREDICCIÓN DEEP LEARNING (LSTM)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Paso 1: Cargar datos
    # ------------------------------------------------------------------
    print("\n[1/7] Cargando datos...")
    base_path = str(Path(__file__).parent)
    context = load_market_data(base_path)
    assets = context["assets"]
    T_vals = context["T_vals"]
    nT = context["nT"]
    print(f"  {nT} periodos, activos: {assets}")

    # Construir array de retornos (T, 2)
    r = context["r"]
    returns_array = np.array([[r[a].loc[t] for a in assets] for t in T_vals])

    # ------------------------------------------------------------------
    # Paso 2: Dataset supervisado
    # ------------------------------------------------------------------
    print(f"\n[2/7] Construyendo dataset (H={H})...")
    train_ds, valid_ds, test_ds, _, n_train = build_datasets(context, H)
    print(f"  Train: {len(train_ds)} muestras")
    print(f"  Valid: {len(valid_ds) - n_train} muestras (nuevas)")
    print(f"  Test:  {len(test_ds) - len(valid_ds)} muestras (nuevas)")

    # ------------------------------------------------------------------
    # Paso 3: Entrenar LSTM
    # ------------------------------------------------------------------
    print(f"\n[3/7] Entrenando LSTM (quantiles {QUANTILE_LEVELS})...")
    model = train_quantile_model(
        train_ds, valid_ds, n_train,
        n_assets=len(assets),
    )

    # ------------------------------------------------------------------
    # Paso 4: Generar escenarios
    # ------------------------------------------------------------------
    print(f"\n[4/7] Generando {N_SCENARIOS} escenarios candidatos...")
    initial_window = returns_array[-H:]  # últimos H retornos observados
    scenarios = generate_scenarios(model, initial_window, T=nT, N=N_SCENARIOS)
    print(f"  Shape: {scenarios.shape}")

    # Estadísticas
    cum_spx = np.prod(1 + scenarios[:, :, 0], axis=1) - 1
    print(f"  Ret acum SPX — min={cum_spx.min():+.4f}  "
          f"median={np.median(cum_spx):+.4f}  max={cum_spx.max():+.4f}")

    # ------------------------------------------------------------------
    # Paso 5: Reducir a quintiles
    # ------------------------------------------------------------------
    print(f"\n[5/7] Reduciendo a {N_QUINTILES} escenarios representativos...")
    quintile_scenarios = reduce_to_quintiles(scenarios, N_QUINTILES)

    # ------------------------------------------------------------------
    # Paso 6: Regret-Grid
    # ------------------------------------------------------------------
    print(f"\n[6/7] Ejecutando regret-grid (15 × {N_QUINTILES} escenarios)...")
    df_regret, g_avg, g_wc, grid_pts, V, R = run_regret_grid(
        quintile_scenarios, context
    )

    # ------------------------------------------------------------------
    # Paso 7: Resultados
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTADOS DEL REGRET-GRID")
    print("=" * 70)

    # Tabla V_{g,s}
    v_cols = [c for c in df_regret.columns if c.startswith("V_")]
    print("\n--- Capital terminal V(g,s) ---")
    print(df_regret[["L", "C", "lambda", "c_mult"] + v_cols].to_string(index=False))

    # Tabla R_{g,s}
    r_cols = [c for c in df_regret.columns if c.startswith("R_")]
    print("\n--- Regret R(g,s) ---")
    print(df_regret[["L", "C"] + r_cols + ["avg_regret", "max_regret"]].to_string(index=False))

    # g* por average regret
    g_info_avg = grid_pts[g_avg]
    print(f"\n  g* (average regret): {g_info_avg['L']}/{g_info_avg['C']}  "
          f"lambda={g_info_avg['lambda']:.2f}  m={g_info_avg['c_mult']:.1f}  "
          f"avg_regret={df_regret.loc[g_avg, 'avg_regret']:.2f}")

    # g* por worst-case regret
    g_info_wc = grid_pts[g_wc]
    print(f"  g* (worst-case):     {g_info_wc['L']}/{g_info_wc['C']}  "
          f"lambda={g_info_wc['lambda']:.2f}  m={g_info_wc['c_mult']:.1f}  "
          f"max_regret={df_regret.loc[g_wc, 'max_regret']:.2f}")

    # ------------------------------------------------------------------
    # Re-ejecutar con g* (average regret) y comparar vs benchmarks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  POLÍTICA FINAL — g* = ({g_info_avg['L']}/{g_info_avg['C']})")
    print("=" * 70)

    theta_neu = {a: 1.0 for a in assets}
    z_star, w_star, u_star, v_star, status = solve_portfolio(
        theta_neu, context,
        lambda_riesgo=g_info_avg["lambda"],
        costo_mult=g_info_avg["c_mult"],
    )

    cap_star = simulate_capital_opt(w_star, u_star, v_star, context)
    cap_bh   = simulate_naive_bh(context)
    cap_rb   = simulate_naive_rb(context)

    C0 = context["Capital_inicial"]

    def summary(label, cap_dict):
        cf = cap_dict[T_vals[-1]]
        return (f"  {label:<35}  ${cf:>12,.2f}  "
                f"{cf/C0-1:>+8.2%}  {cf-C0:>+12,.2f}")

    print(f"\n  z* = {z_star:.6f}  (status: {status})")
    print(f"\n  {'Estrategia':<35}  {'cap_final':>12}  {'ret_acum':>8}  {'inc_cap':>12}")
    print(f"  {'-'*75}")
    print(summary(f"g* Optimo (lam={g_info_avg['lambda']}, m={g_info_avg['c_mult']})", cap_star))
    print(summary("Naive 50/50 Rebalanceo", cap_rb))
    print(summary("Naive Buy & Hold", cap_bh))

    # Guardar resultados
    out_path = Path(base_path) / "regret_grid_results.csv"
    df_regret.to_csv(out_path, index=False)
    print(f"\n  Resultados guardados en: {out_path}")
