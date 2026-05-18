"""
Regret-grid: seleccion de parametros (lambda, m) del portafolio usando
escenarios generados por el pipeline DL (PDF seccion 3).

Pipeline:
  1. Construye contexto DL: mu_hat/sigma_hat historicos + p(t) DL + 5 escenarios.
  2. Para cada g = (lambda, m) en la grilla: resuelve solve_portfolio una vez -> w^g.
  3. Para cada escenario s: simula capital -> V[g, s] (ec. 19).
  4. Calcula regret R[g, s] = V_best_s - V[g, s] (ec. 22).
  5. Selecciona g* por regret promedio (ec. 23) y peor caso (ec. 24).
"""
from pathlib import Path

import gamspy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    ASSETS,
    CAPITAL_INICIAL,
    C_BASE,
    CHECKPOINT_PATH,
    DATA_DIR,
    LAMBDA_GRID,
    LAMBDA_RIESGO_DEFAULT,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    PROB_CSV,
    REGIMES,
    RESULTS_DIR,
    RETURN_COL,
    RETURN_CSV,
    SCENARIO_SEED,
    SOLVER,
    SUMMARY_ASSET,
    T_HORIZON,
    V_MAX_BUFFER,
    V_MAX_REF_ASSET,
    W0,
)
from dl.generador_escenarios import (
    generate_candidate_scenarios,
    generate_representative_scenarios,
    reduce_to_representatives,
)
from dl.prediccion_deciles import load_checkpoint
from dl.regimen_predicted import regimen_from_deciles


# ================================================================
# 0) Carga de datos historicos + momentos por regimen (Opcion B del PDF)
# ================================================================

def load_market_data(base_dir_str: str | Path | None = None):
    """
    Carga los CSVs, procesa probabilidades y retornos,
    y calcula momentos mezclados (mu_mix, sigma_mix).
    Replica exactamente la Seccion 2 del GAMS (Opcion B del PDF).

    Nombres de archivos, activos, regimenes, w0, c_base y Capital_inicial
    se leen de `config.py`.
    """
    BASE_DIR = Path(base_dir_str) if base_dir_str is not None else DATA_DIR

    assets  = list(ASSETS)
    regimes = list(REGIMES)

    prob = {}
    ret  = {}
    for a in assets:
        df_p = pd.read_csv(BASE_DIR / PROB_CSV[a], sep=",")
        df_r = pd.read_csv(BASE_DIR / RETURN_CSV[a], sep=",")
        df_p.columns = [c.strip() for c in df_p.columns]
        df_r.columns = [c.strip() for c in df_r.columns]
        df_p["t"] = df_p["t"].astype(int)
        df_r["t"] = df_r["t"].astype(int)
        prob[a] = df_p
        ret[a]  = df_r

    T_vals = sorted(prob[assets[0]]["t"].unique())

    r = {a: ret[a].set_index("t")[RETURN_COL[a]] for a in assets}
    p = {a: prob[a].set_index("t")[regimes]      for a in assets}

    den_mu    = {}
    mu_hat    = {}
    den_sig   = {}
    sigma_hat = {}

    for i in assets:
        for k in regimes:
            den_mu[(i, k)] = p[i][k].sum()
            if den_mu[(i, k)] > 0:
                mu_hat[(i, k)] = (p[i][k] * r[i]).sum() / den_mu[(i, k)]
            else:
                mu_hat[(i, k)] = 0.0

    for i in assets:
        for j in assets:
            for k in regimes:
                pi_k = p[i][k]
                pj_k = p[j][k]
                den_sig[(i, j, k)] = (pi_k * pj_k).sum()
                if den_sig[(i, j, k)] > 0:
                    term = pi_k * pj_k * (r[i] - mu_hat[(i, k)]) * (r[j] - mu_hat[(j, k)])
                    sigma_hat[(i, j, k)] = term.sum() / den_sig[(i, j, k)]
                else:
                    sigma_hat[(i, j, k)] = 0.0

    mu_mix    = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}

    for i in assets:
        for k in regimes:
            mu_mix[i] += p[i][k] * mu_hat[(i, k)]

    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] += p[i][k] * p[j][k] * sigma_hat[(i, j, k)]

    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    # Presupuesto de riesgo V_max: varianza muestral del activo de referencia
    # (el "estable") escalada por V_MAX_BUFFER. Entra en la FO como penalizacion
    # lambda*(Riesgo - V_max). Es una constante => no altera el optimo w*, solo
    # desplaza el valor de z. (ddof=1, sample variance, igual que pandas default).
    V_max = float(r[V_MAX_REF_ASSET].var() * V_MAX_BUFFER)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              len(T_vals),
        "assets":          assets,
        "c_base":          dict(C_BASE),
        "w0":              dict(W0),
        "r":               r,
        "Capital_inicial": CAPITAL_INICIAL,
        "V_max":           V_max,
    }


# ================================================================
# 0.b) Solucionador GAMSPy + IPOPT
# ================================================================

def solve_portfolio(context: dict,
                    lambda_riesgo: float = LAMBDA_RIESGO_DEFAULT,
                    costo_mult:    float = 1.0,
                    verbose:       bool  = False):
    """
    Resuelve el modelo media-varianza con presupuesto de riesgo V_max y costos,
    usando GAMSPy + IPOPT.

    max z = sum_t [ sum_i w(i,t)*mu_mix(i,t)
                  - lambda * ( sum_(i,j) w(i,t)*w(j,t)*sigma_mix(i,j,t) - V_max )
                  - sum_i c_base(i)*costo_mult*(u(i,t)+v(i,t)) ]

    s.t.  sum_i w(i,t) = 1                           para todo t
          w(i,t) - w(i,t-1) = u(i,t) - v(i,t)       para t > t1
          w(i,t1) - w0(i)   = u(i,t1) - v(i,t1)     anclaje inicial
          0 <= w(i,t) <= 1;  u(i,t), v(i,t) >= 0

    V_max es una constante (no depende de las decisiones) => no altera el optimo
    w*, solo desplaza el valor de z en -lambda*T*V_max respecto a la version
    sin presupuesto. Se lee de context['V_max'].
    """
    mu_base   = context["mu_mix"]
    sigma_mix = context["sigma_mix"]
    T_vals    = context["T_vals"]
    assets    = context["assets"]
    c_base    = context["c_base"]
    w0_dict   = context["w0"]
    V_max     = float(context["V_max"])

    T_labels = [f"t{n}" for n in T_vals]   # "t1" .. "t163"

    m = gp.Container()

    i_set = gp.Set(m, "i", records=assets,           description="activos")
    j_set = gp.Alias(m, "j", i_set)
    t_set = gp.Set(m, "t", records=T_labels,          description="periodos")

    mu_records = [
        [i, f"t{t}", mu_base[i].loc[t]]
        for i in assets for t in T_vals
    ]
    mu_p = gp.Parameter(
        m, "mu_mix", domain=[i_set, t_set],
        records=pd.DataFrame(mu_records, columns=["i", "t", "value"]),
        description="media mixta por periodo",
    )

    sig_records = [
        [i, j, f"t{t}", sigma_mix[i][j].loc[t]]
        for i in assets for j in assets for t in T_vals
    ]
    sig_p = gp.Parameter(
        m, "sigma_mix", domain=[i_set, j_set, t_set],
        records=pd.DataFrame(sig_records, columns=["i", "j", "t", "value"]),
        description="covarianza mixta por periodo",
    )

    c_base_p = gp.Parameter(
        m, "c_base", domain=[i_set],
        records=pd.DataFrame(
            [[i, c_base[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="costo base de transaccion",
    )

    c_mult_p = gp.Parameter(m, "c_mult", records=costo_mult,
                            description="multiplicador de costo en FO")

    w0_p = gp.Parameter(
        m, "w0", domain=[i_set],
        records=pd.DataFrame(
            [[i, w0_dict[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="portafolio inicial 50/50",
    )

    lam_p = gp.Parameter(m, "lambda_riesgo", records=lambda_riesgo,
                          description="aversion al riesgo")
    v_max_p = gp.Parameter(m, "V_max", records=V_max,
                           description="presupuesto de riesgo")

    z_var = gp.Variable(m, "z",                              description="valor objetivo")
    w_var = gp.Variable(m, "w", domain=[i_set, t_set], type="positive", description="peso")
    u_var = gp.Variable(m, "u", domain=[i_set, t_set], type="positive", description="compras")
    v_var = gp.Variable(m, "v", domain=[i_set, t_set], type="positive", description="ventas")

    w_var.up[i_set, t_set] = 1.0
    # Acotar compras/ventas: en cualquier optimo no-degenerado |w(t)-w(t-1)| <= 1
    # implica u, v <= 1. Sin esta cota, si costo_mult=0 el termino c*(u+v) se anula
    # y IPOPT puede devolver u, v arbitrariamente grandes (pares u_i = v_i = 1e9
    # que satisfacen u-v=Δw). El simulador con c_base real explota a V absurdos.
    u_var.up[i_set, t_set] = 1.0
    v_var.up[i_set, t_set] = 1.0

    fo = gp.Equation(m, "FO_media_var_costo",
                     description="FO: retorno - lambda*(var - V_max) - costos")
    fo[...] = z_var == gp.Sum(
        t_set,
        gp.Sum(i_set, w_var[i_set, t_set] * mu_p[i_set, t_set])
        - lam_p * (gp.Sum((i_set, j_set),
                          w_var[i_set, t_set] * w_var[j_set, t_set] * sig_p[i_set, j_set, t_set])
                   - v_max_p)
        - gp.Sum(i_set, c_base_p[i_set] * c_mult_p * (u_var[i_set, t_set] + v_var[i_set, t_set]))
    )

    norm = gp.Equation(m, "normalizacion_pesos", domain=[t_set],
                       description="suma de pesos = 1")
    norm[t_set] = gp.Sum(i_set, w_var[i_set, t_set]) == 1

    rebal = gp.Equation(m, "rebalanceo_lineal", domain=[i_set, t_set],
                        description="identidad de rebalanceo")
    rebal[i_set, t_set].where[gp.Ord(t_set) > 1] = (
        w_var[i_set, t_set] - w_var[i_set, t_set.lag(1)]
        == u_var[i_set, t_set] - v_var[i_set, t_set]
    )

    anclaje = gp.Equation(m, "anclaje_inicial", domain=[i_set],
                          description="anclaje al portafolio inicial")
    anclaje[i_set] = (
        w_var[i_set, "t1"] - w0_p[i_set]
        == u_var[i_set, "t1"] - v_var[i_set, "t1"]
    )

    portfolio = gp.Model(
        m,
        name="PortafolioEstadosCostos",
        equations=m.getEquations(),
        problem="QCP",
        sense=gp.Sense.MAX,
        objective=z_var,
    )

    output = None if not verbose else __import__("sys").stdout
    portfolio.solve(solver=SOLVER, output=output)

    if portfolio.status not in (
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.OptimalGlobal,
    ):
        raise RuntimeError(
            f"GAMSPy/IPOPT no encontro solucion optima. Status: {portfolio.status}"
        )

    z_val = float(z_var.toValue())

    def _records_to_dict(var):
        sol = {}
        for _, row in var.records.iterrows():
            i_key = row["i"]
            t_key = int(row["t"][1:])
            sol[i_key, t_key] = float(row["level"])
        return sol

    w_sol = _records_to_dict(w_var)
    u_sol = _records_to_dict(u_var)
    v_sol = _records_to_dict(v_var)

    status = ("optimal" if portfolio.status == gp.ModelStatus.OptimalGlobal
              else "optimal_local")
    return z_val, w_sol, u_sol, v_sol, status


# ================================================================
# 1) DL: walking-window sobre el historico para p_bull(t)
# ================================================================

def predict_pbull_walking(model, returns_history, T):
    """p_bull(t) por ventana real del historico, sin autoalimentar.

    Para cada t en H+1..T se aplica el LSTM a la ventana real
    [r_{t-H}..r_{t-1}] y se deriva p_bull(t) via la ec. 15 del PDF.
    Para t=1..H no hay H retornos previos en el dataset; esas posiciones
    se rellenan por padding con p_bull(H+1) (decision documentada para
    no romper la alineacion temporal con el GAMS base, T_vals=1..T).

    returns_history: (T, A) retornos reales por activo, indexados 0..T-1
                     (returns_history[k] corresponde a la semana t=k+1).
    return:          (T, A) p_bull alineado a t=1..T.
    """
    cfg = model.config
    H   = cfg.H
    A   = cfg.n_assets
    if returns_history.shape != (T, A):
        raise ValueError(
            f"returns_history shape {returns_history.shape} != (T={T}, A={A})"
        )
    if T <= H:
        raise ValueError(f"T={T} debe ser > H={H} para tener al menos una ventana real.")

    p_bull = np.empty((T, A), dtype=np.float32)

    for idx in range(H, T):                          # idx 0-based; semana t1 = idx+1
        window = returns_history[idx - H : idx, :]   # (H, A) ventana real previa
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_tensor).numpy()[0] for net in model.nets]  # K * (A, Q)
        preds = np.mean(np.stack(outs, axis=0), axis=0)              # (A, Q)
        p_bull_step, _ = regimen_from_deciles(preds)                 # (A,)
        p_bull[idx] = p_bull_step

    # Padding para t=1..H (no hay ventana real previa de tamaño H).
    p_bull[:H] = p_bull[H]
    return p_bull


# ================================================================
# 2) Contexto DL -> dict compatible con solve_portfolio
# ================================================================

def _compute_hist_moments(r_hist, p_hist, assets, regimes):
    """Replica el calculo de mu_hat/sigma_hat de load_market_data (Opcion B)."""
    mu_hat = {}
    for i in assets:
        for k in regimes:
            den = p_hist[i][k].sum()
            mu_hat[(i, k)] = ((p_hist[i][k] * r_hist[i]).sum() / den
                              if den > 0 else 0.0)

    sigma_hat = {}
    for i in assets:
        for j in assets:
            for k in regimes:
                pi_k = p_hist[i][k]
                pj_k = p_hist[j][k]
                den  = (pi_k * pj_k).sum()
                if den > 0:
                    term = (pi_k * pj_k
                            * (r_hist[i] - mu_hat[(i, k)])
                            * (r_hist[j] - mu_hat[(j, k)]))
                    sigma_hat[(i, j, k)] = term.sum() / den
                else:
                    sigma_hat[(i, j, k)] = 0.0
    return mu_hat, sigma_hat


def build_dl_context(data_dir, checkpoint_path, T=T_HORIZON,
                     N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
                     seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
                     position=None):
    """
    Construye un contexto compatible con solve_portfolio siguiendo la
    descomposicion por regimen del PDF (ec. 2-5), con F2 (LSTM walking)
    como fuente unica de p_{i,k,t}.

    Diseno:
      - p_{i,k,t} = predict_pbull_walking (LSTM aplicado a la ventana real
        previa, ec. 15 del PDF). Misma fuente para los momentos por regimen
        (ec. 2-3) y para la mezcla por periodo (ec. 4-5).
      - mu_hat[(i,k)], sigma_hat[(i,j,k)] = ec. (2)-(3) con esa p.
      - mu_mix(i,t), sigma_mix(i,j,t)     = ec. (4)-(5) con esa p.
      - 5 escenarios = quintiles del rollout autoregresivo del LSTM.

    Nota sobre coherencia ex-ante / ex-post:
      La FO ve mu_mix derivado del walking; los escenarios viven en el
      rollout autoregresivo. Son dos procesos distintos => existe una
      grieta entre lo que ve la FO y lo que viven los escenarios.
      Cuantificarla via `inspeccion/inputs_out/6_coherencia_*.csv`.
    """
    data_dir = Path(data_dir)
    base_ctx = load_market_data(str(data_dir))
    assets   = list(base_ctx["assets"])
    regimes  = list(REGIMES)
    r_hist   = base_ctx["r"]

    # --- modelo DL + historico ---
    model = load_checkpoint(checkpoint_path)
    H     = model.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T] for i in assets], axis=1,
    ).astype(np.float32)                                 # (T, A)
    initial_window = returns_history[-H:, :]             # (H, A)
    T_vals = list(range(1, T + 1))

    # --- p_{i,k,t} = F2: LSTM walking sobre ventana real previa (ec. 15) ---
    p_bull_walking = predict_pbull_walking(model, returns_history, T)  # (T, A)
    p_bear_walking = 1.0 - p_bull_walking
    p_dl = {
        asset: pd.DataFrame(
            {"bear": p_bear_walking[:, ai], "bull": p_bull_walking[:, ai]},
            index=T_vals,
        )
        for ai, asset in enumerate(assets)
    }

    # r_hist truncado a t=1..T y reindexado a T_vals para alinear con p_dl.
    r_hist_T = {
        i: pd.Series(r_hist[i].sort_index().values[:T], index=T_vals)
        for i in assets
    }

    # --- Momentos por regimen (ec. 2-3) con p del walking ---
    mu_hat, sigma_hat = _compute_hist_moments(r_hist_T, p_dl, assets, regimes)

    # --- Mezcla por periodo (ec. 4-5) ---
    mu_mix    = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}

    for i in assets:
        for k in regimes:
            mu_mix[i] += p_dl[i][k] * mu_hat[(i, k)]

    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] += p_dl[i][k] * p_dl[j][k] * sigma_hat[(i, j, k)]

    # Simetrizacion numerica (la formula ya es simetrica analiticamente).
    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    # --- N candidatos del LSTM (rollout, solo para generar los 5 escenarios) ---
    candidates = generate_candidate_scenarios(
        model, initial_window, N=N_candidates, T=T, seed=seed,
    )                                                    # (N, T, A)

    from config import SCENARIO_POSITION
    pos = position if position is not None else SCENARIO_POSITION
    summary_idx = assets.index(summary_asset)
    scenarios = reduce_to_representatives(
        candidates, summary_asset_idx=summary_idx,
        n_quintiles=n_scenarios, position=pos,
    )                                                    # (n_scenarios, T, A)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              T,
        "assets":          assets,
        "c_base":          base_ctx["c_base"],
        "w0":              base_ctx["w0"],
        "Capital_inicial": base_ctx["Capital_inicial"],
        "V_max":           base_ctx["V_max"],
        "r":               r_hist,
        "scenarios":       scenarios,
        "p_dl":            p_dl,
        # Diagnostico adicional especifico de esta variante:
        "mu_hat":          mu_hat,
        "sigma_hat":       sigma_hat,
    }


# ================================================================
# 3) Simulacion de capital en un escenario
# ================================================================

def simulate_capital_on_scenario(w_sol, u_sol, v_sol, scenario,
                                 assets, c_base, C0, T_vals):
    """
    Ec. (19): x_{t+1} = x_t (1 + sum_i w(i,t)*r^s_{i,t})
                      - x_t sum_i c_i * |w(i,t) - w(i,t-1)|
    El costo cobrado en el paso t->t+1 es el del rebalanceo HACIA w(t),
    es decir u(i,t)+v(i,t). En el primer paso (t=t1) eso captura el
    rebalanceo inicial w0 -> w(t1).
    scenario: (T, A) — scenario[k, ai] es el retorno en el periodo T_vals[k].
    """
    cap = {T_vals[0]: C0}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_sol[i, t_prev] * scenario[idx - 1, ai]
                     for ai, i in enumerate(assets))
        turn   = sum(c_base[i] * (u_sol[i, t_prev] + v_sol[i, t_prev]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


# ================================================================
# 4) Regret-grid
# ================================================================

def run_regret_grid(context, lambda_grid, m_grid):
    """Alg. 1 del PDF 3.5: un solve por g, simulacion por (g, s)."""
    assets    = context["assets"]
    T_vals    = context["T_vals"]
    c_base    = context["c_base"]
    C0        = context["Capital_inicial"]
    scenarios = context["scenarios"]
    n_S       = scenarios.shape[0]

    rows     = []
    policies = {}
    total    = len(lambda_grid) * len(m_grid)
    run      = 0

    for lam in lambda_grid:
        for cm in m_grid:
            run += 1
            print(f"  [{run:>2}/{total}] lambda={lam:.2f}  m={cm:.1f} ...",
                  end=" ", flush=True)
            z, w_sol, u_sol, v_sol, _ = solve_portfolio(
                context, lambda_riesgo=lam, costo_mult=cm,
            )
            policies[(lam, cm)] = (w_sol, u_sol, v_sol, z)

            Vs = []
            for s in range(n_S):
                cap = simulate_capital_on_scenario(
                    w_sol, u_sol, v_sol, scenarios[s],
                    assets, c_base, C0, T_vals,
                )
                Vs.append(cap[T_vals[-1]])
                rows.append({"lambda": lam, "m": cm, "s": s,
                             "V": Vs[-1], "z": z})
            print(f"z={z:.4f}  V=[${min(Vs):,.0f} .. ${max(Vs):,.0f}]")

    return pd.DataFrame(rows), policies


# ================================================================
# 5) Regret y seleccion
# ================================================================

def compute_regret_and_select(V_df):
    """V_best_s (ec. 21) y R[g, s] (ec. 22); elige g* por promedio y peor caso.

    Valida que V sea finito y razonable: ratio max/median > 1000 indica una
    degeneracion (tipicamente IPOPT con costo_mult=0 o restriccion mal
    configurada). Si g* cae en la frontera del grid, emite un warning para
    sugerir extender LAMBDA_GRID o M_GRID.
    """
    V_table = V_df.pivot_table(
        index=["lambda", "m"], columns="s", values="V", aggfunc="first",
    )
    v_arr = V_table.values
    if not np.isfinite(v_arr).all():
        raise ValueError(
            f"V_table contiene valores no finitos. Probable patologia en "
            f"solve_portfolio (revisar costo_mult > 0).\n{V_table}"
        )
    median = np.median(v_arr)
    if median > 0 and v_arr.max() / median > 1000:
        raise ValueError(
            f"V_table contiene valores absurdos: max={v_arr.max():.2e}, "
            f"median={median:.2e}, ratio={v_arr.max()/median:.0f}x. "
            "Probable degeneracion de IPOPT (revisar que M_GRID > 0)."
        )

    V_best_s     = V_table.max(axis=0)
    R_table      = V_best_s - V_table
    mean_regret  = R_table.mean(axis=1)
    worst_regret = R_table.max(axis=1)
    summary = pd.DataFrame({
        "mean_regret":  mean_regret,
        "worst_regret": worst_regret,
    })
    g_mean  = mean_regret.idxmin()
    g_worst = worst_regret.idxmin()

    lambda_values = sorted(V_table.index.get_level_values("lambda").unique())
    m_values      = sorted(V_table.index.get_level_values("m").unique())
    boundary_lam  = {lambda_values[0], lambda_values[-1]}
    boundary_m    = {m_values[0],      m_values[-1]}
    import warnings
    for label, (lam, m_) in [("g*_mean", g_mean), ("g*_worst", g_worst)]:
        on_boundary = []
        if lam in boundary_lam: on_boundary.append(f"lambda={lam:.2f}")
        if m_  in boundary_m:   on_boundary.append(f"m={m_:.2f}")
        if on_boundary:
            warnings.warn(
                f"{label} cae en frontera del grid ({', '.join(on_boundary)}). "
                "Considera extender LAMBDA_GRID/M_GRID en config.py.",
                stacklevel=2,
            )

    return {
        "V_table":        V_table,
        "R_table":        R_table,
        "V_best_s":       V_best_s,
        "regret_summary": summary,
        "g_mean":         g_mean,
        "g_worst":        g_worst,
        "g_mean_metric":  mean_regret.min(),
        "g_worst_metric": worst_regret.min(),
    }


# ================================================================
# 6) Plot: evolucion del capital por escenario bajo una politica
# ================================================================

def simulate_capital_opt(w_sol, u_sol, v_sol, context):
    """Capital ex-post bajo retornos historicos (ec. 19, version historica).

    cap[t] = cap[t-1] * (1 + sum_i w(i,t-1)*r(i,t-1))
           - cap[t-1] * sum_i c_base(i)*(u(i,t-1)+v(i,t-1)).
    El costo del paso t-1 -> t es el del rebalanceo HACIA w(t-1) (incluye
    el rebalanceo inicial w0 -> w(t1) en el primer paso).
    Siempre usa c_base (sin multiplicador): el costo_mult solo penaliza ex-ante en la FO.
    """
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    c_base  = context["c_base"]
    Capital = context["Capital_inicial"]

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_sol[i, t_prev] * r[i].loc[t_prev] for i in assets)
        turn   = sum(c_base[i] * (u_sol[i, t_prev] + v_sol[i, t_prev]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def simulate_naive_bh(context):
    """Naive 50/50 buy & hold sobre retornos historicos (sin rebalanceo ni costos)."""
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    Capital = context["Capital_inicial"]
    w_naive = {i: 0.5 for i in assets}

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_naive[i] * r[i].loc[t_prev] for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port)
    return cap


def simulate_naive_rb(context):
    """Naive 50/50 con rebalanceo semanal y costos sobre retornos historicos."""
    T_vals   = context["T_vals"]
    assets   = context["assets"]
    r        = context["r"]
    c_base   = context["c_base"]
    Capital  = context["Capital_inicial"]
    w_target = 0.5

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_target * r[i].loc[t_prev] for i in assets)
        w_bh   = {i: w_target * (1.0 + r[i].loc[t_prev]) / (1.0 + r_port) for i in assets}
        turn   = sum(c_base[i] * abs(w_target - w_bh[i]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def plot_capital_evolution_historical(cap_opt, cap_rb, cap_bh, cap_regret,
                                      T_vals, lam_star, m_star, out_path):
    """OPT vs Naive 50/50 (rebal) vs Naive 50/50 (B&H) vs Regret-Grid g*_mean.

    Las cuatro politicas se simulan sobre la misma serie de retornos historicos;
    la diferencia es la (lambda, m) que produjo cada w(i,t). El "OPT" es el
    optimo media-varianza con (lambda=1.00, m=1.0); el "Regret-Grid" usa los
    parametros seleccionados ex-ante por el pipeline DL + regret.
    """
    x = list(T_vals)
    y_opt    = [cap_opt[t]    for t in x]
    y_rb     = [cap_rb[t]     for t in x]
    y_bh     = [cap_bh[t]     for t in x]
    y_regret = [cap_regret[t] for t in x]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_opt,    label="OPT",                       color="#F2B705", linewidth=1.8)
    ax.plot(x, y_rb,     label="NAIVE 50/50 (rebal)",       color="#8B3A1F", linewidth=1.2)
    ax.plot(x, y_bh,     label="NAIVE 50/50 (buy&hold)",    color="#E63946", linewidth=1.2)
    ax.plot(x, y_regret,
            label=f"Regret-Grid g*_mean (lambda={lam_star:.2f}, m={m_star:.1f})",
            color="#1f77b4", linewidth=1.8)

    ax.set_title("Evolucion de capital")
    ax.set_xlabel("t")
    ax.set_ylabel("Capital")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado en: {out_path}")


def plot_capital_curves(w_sol, u_sol, v_sol, context, title, out_path):
    assets    = context["assets"]
    T_vals    = context["T_vals"]
    c_base    = context["c_base"]
    C0        = context["Capital_inicial"]
    scenarios = context["scenarios"]
    n_S       = scenarios.shape[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for s in range(n_S):
        cap = simulate_capital_on_scenario(
            w_sol, u_sol, v_sol, scenarios[s], assets, c_base, C0, T_vals,
        )
        ax.plot(T_vals, [cap[t] for t in T_vals],
                color=cmap(s / max(n_S - 1, 1)),
                label=f"Escenario s{s + 1}", linewidth=1.3)
    ax.axhline(C0, color="#666", linestyle="--", linewidth=0.8,
               label=f"Capital inicial (${C0:,.0f})")
    ax.set_title(title)
    ax.set_xlabel("t (forward)")
    ax.set_ylabel("Capital")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado en: {out_path}")


# ================================================================
# 6.b) Backtest historico (politica DL aplicada a r_hist)
# ================================================================

def run_historical_backtest(w_star, u_star, v_star, lam_m, m_m,
                            V_mean_row, n_scenarios,
                            data_dir, out_path, hist_ctx=None):
    """Compara OPT, Naive (rebal/B&H) y Regret-Grid g*_mean sobre retornos historicos.

    El "OPT" es el optimo media-varianza con (lambda=1.00, m=1.0) sobre el contexto
    historico (replica el caso base del GAMS). La linea del Regret-Grid usa los
    pesos w_star ya calculados por el grid DL (no se re-optimiza), simulados sobre
    r_hist — es la decision real del modelo enfrentada a la trayectoria observada.
    """
    if hist_ctx is None:
        hist_ctx = load_market_data(data_dir)
    C0 = hist_ctx["Capital_inicial"]

    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        hist_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, hist_ctx)
    cap_rg  = simulate_capital_opt(w_star, u_star, v_star, hist_ctx)
    cap_rb  = simulate_naive_rb(hist_ctx)
    cap_bh  = simulate_naive_bh(hist_ctx)

    T_h     = hist_ctx["T_vals"]
    t_final = T_h[-1]
    cf_opt  = cap_opt[t_final]
    cf_rg   = cap_rg [t_final]
    cf_rb   = cap_rb [t_final]
    cf_bh   = cap_bh [t_final]

    print("\n--- Backtest historico (politica aplicada a r_hist) ---")
    print(f"  Capital inicial = ${C0:,.2f}   horizonte = t1..t{t_final} ({len(T_h)} periodos)")
    print(f"  {'politica':<42}  {'cap_final':>12}  {'ret_acum':>9}  {'inc_cap':>12}")
    print(f"  {'-' * 80}")
    fmt = "  {:<42}  ${:>11,.2f}  {:>+8.2%}  ${:>+11,.2f}"
    print(fmt.format("OPT (lambda=1.00, m=1.0)",
                     cf_opt, cf_opt/C0 - 1, cf_opt - C0))
    print(fmt.format(f"Regret-Grid g*_mean (lambda={lam_m:.2f}, m={m_m:.1f})",
                     cf_rg,  cf_rg /C0 - 1, cf_rg  - C0))
    print(fmt.format("Naive 50/50 rebalanceo",
                     cf_rb,  cf_rb /C0 - 1, cf_rb  - C0))
    print(fmt.format("Naive 50/50 buy & hold",
                     cf_bh,  cf_bh /C0 - 1, cf_bh  - C0))
    print(f"\n  Nota: el ret_acum de Regret-Grid sobre r_hist NO es comparable")
    print(f"        con el +{V_mean_row.mean()/C0 - 1:.2%} promedio sobre los "
          f"{n_scenarios} escenarios DL del bloque anterior — uno es backtest")
    print(f"        sobre la trayectoria observada, el otro es promedio sobre las")
    print(f"        trayectorias forecast del LSTM.")

    plot_capital_evolution_historical(
        cap_opt, cap_rb, cap_bh, cap_rg,
        T_h, lam_m, m_m, out_path=out_path,
    )

    return {
        "cap_opt": cap_opt, "cap_rg": cap_rg,
        "cap_rb":  cap_rb,  "cap_bh": cap_bh,
        "hist_ctx": hist_ctx,
    }


# ================================================================
# 7) Bloque principal
# ================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGRET-GRID — DL -> optimizador -> seleccion (lambda, m)")
    print("=" * 70)
    print("Cargando datos y construyendo contexto DL ...")
    ctx = build_dl_context(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON,
        N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    print(f"  Assets     : {ctx['assets']}")
    print(f"  T          : {ctx['nT']} periodos forward (t1..t{ctx['nT']})")
    print(f"  Scenarios  : {ctx['scenarios'].shape} (S, T, A)")
    for i in ctx["assets"]:
        col = ctx["p_dl"][i]["bull"]
        print(f"  p_bull {i:<7}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}")

    lambda_grid = list(LAMBDA_GRID)
    m_grid      = list(M_GRID)

    print("\n" + "-" * 70)
    print(f"Corriendo {len(lambda_grid)}x{len(m_grid)}="
          f"{len(lambda_grid) * len(m_grid)} puntos x "
          f"{ctx['scenarios'].shape[0]} escenarios")
    print("-" * 70)
    V_df, policies = run_regret_grid(ctx, lambda_grid, m_grid)

    res = compute_regret_and_select(V_df)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print("\n--- V[g, s] — capital terminal por (lambda, m) y escenario ---")
    print(res["V_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- R[g, s] = V_best_s - V[g, s] ---")
    print(res["R_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- Resumen de regret por g ---")
    print(res["regret_summary"].to_string(float_format="${:,.2f}".format))

    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    C0 = ctx["Capital_inicial"]
    V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
    V_worst_row = res["V_table"].loc[(lam_w, m_w)]
    print("\n--- Seleccion de g* ---")
    print(f"  g*_mean  (ec. 23): lambda={lam_m:.2f}  m={m_m:.1f}  "
          f"mean_regret=${res['g_mean_metric']:,.2f}")
    print(f"      V: mean=${V_mean_row.mean():>12,.2f}  "
          f"worst=${V_mean_row.min():>12,.2f}  "
          f"best=${V_mean_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios = {V_mean_row.mean()/C0 - 1:+.2%}")
    print(f"  g*_worst (ec. 24): lambda={lam_w:.2f}  m={m_w:.1f}  "
          f"worst_regret=${res['g_worst_metric']:,.2f}")
    print(f"      V: mean=${V_worst_row.mean():>12,.2f}  "
          f"worst=${V_worst_row.min():>12,.2f}  "
          f"best=${V_worst_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario     = {V_worst_row.min()/C0 - 1:+.2%}")

    # --- Persistencia ---
    out_V = RESULTS_DIR / "regret_grid_results.csv"
    V_df.to_csv(out_V, index=False)
    print(f"\n  V_df (long)           : {out_V}")

    out_R = RESULTS_DIR / "regret_table.csv"
    res["R_table"].to_csv(out_R)
    print(f"  Tabla de regret       : {out_R}")

    out_summary = RESULTS_DIR / "regret_summary.csv"
    res["regret_summary"].to_csv(out_summary)
    print(f"  Resumen por g         : {out_summary}")

    # --- Plot capital bajo g*_mean ---
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    plot_capital_curves(
        w_star, u_star, v_star, ctx,
        title=f"Capital por escenario con g*_mean (lambda={lam_m:.2f}, m={m_m:.1f})",
        out_path=RESULTS_DIR / "regret_capital_curves.png",
    )

    # --- Backtest historico: OPT vs Naive vs Regret-Grid g*_mean ---
    run_historical_backtest(
        w_star, u_star, v_star, lam_m, m_m,
        V_mean_row=V_mean_row,
        n_scenarios=ctx["scenarios"].shape[0],
        data_dir=DATA_DIR,
        out_path=RESULTS_DIR / "evolucion_capital.png",
    )

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
