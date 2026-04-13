from pathlib import Path
import pandas as pd
import gamspy as gp

# ================================================================
# 0) Carga de Datos
# ================================================================

def load_market_data(base_dir_str: str):
    """
    Carga los CSVs, procesa probabilidades y retornos,
    y calcula momentos mezclados (mu_mix, sigma_mix).
    Replica exactamente la Sección 2 del GAMS (Opción B del PDF).
    """
    BASE_DIR = Path(base_dir_str)

    prob_spx = pd.read_csv(BASE_DIR / "prob_spx.csv",  sep=",")
    prob_cmc = pd.read_csv(BASE_DIR / "prob_cmc200.csv", sep=",")
    ret_spx  = pd.read_csv(BASE_DIR / "ret_semanal_spx.csv",   sep=",")
    ret_cmc  = pd.read_csv(BASE_DIR / "ret_semanal_cmc200.csv", sep=",")

    for df in [prob_spx, prob_cmc, ret_spx, ret_cmc]:
        df.columns = [c.strip() for c in df.columns]

    prob_spx["t"] = prob_spx["t"].astype(int)
    prob_cmc["t"] = prob_cmc["t"].astype(int)
    ret_spx["t"]  = ret_spx["t"].astype(int)
    ret_cmc["t"]  = ret_cmc["t"].astype(int)

    T_vals  = sorted(prob_spx["t"].unique())
    assets  = ["SPX", "CMC200"]
    regimes = ["bear", "bull"]

    r = {
        "SPX":    ret_spx.set_index("t")["ret_semanal_spx"],
        "CMC200": ret_cmc.set_index("t")["ret_semanal_cmc200"],
    }
    p = {
        "SPX":    prob_spx.set_index("t")[regimes],
        "CMC200": prob_cmc.set_index("t")[regimes],
    }

    # Momentos por régimen
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

    # Mezcla por periodo
    mu_mix    = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}

    for i in assets:
        for k in regimes:
            mu_mix[i] += p[i][k] * mu_hat[(i, k)]

    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] += p[i][k] * p[j][k] * sigma_hat[(i, j, k)]

    # Simetrizar
    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    c_base = {"SPX": 0.005, "CMC200": 0.010}
    w0     = {"SPX": 0.5,   "CMC200": 0.5}

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              len(T_vals),
        "assets":          assets,
        "c_base":          c_base,
        "w0":              w0,
        "r":               r,
        "Capital_inicial": 10000.0,
    }


# ================================================================
# 1) Solucionador GAMSPy + IPOPT
# ================================================================

def solve_portfolio(theta: dict, context: dict,
                    lambda_riesgo: float = 0.10,
                    costo_mult:    float = 1.0,
                    verbose:       bool  = False):
    """
    Resuelve el modelo media-varianza con costos usando GAMSPy + IPOPT,
    replicando exactamente el modelo GAMS (ps(1).gms, Sección 3).

    max z = sum_t [ sum_i w(i,t)*mu_mix(i,t)*theta(i)
                  - lambda * sum_(i,j) w(i,t)*w(j,t)*sigma_mix(i,j,t)
                  - sum_i c_eff(i)*(u(i,t)+v(i,t)) ]

    s.t.  sum_i w(i,t) = 1                           para todo t
          w(i,t) - w(i,t-1) = u(i,t) - v(i,t)       para t > t1
          w(i,t1) - w0(i)   = u(i,t1) - v(i,t1)     anclaje inicial
          0 <= w(i,t) <= 1;  u(i,t), v(i,t) >= 0
    """
    mu_base   = context["mu_mix"]
    sigma_mix = context["sigma_mix"]
    T_vals    = context["T_vals"]
    assets    = context["assets"]
    c_base    = context["c_base"]
    w0_dict   = context["w0"]

    T_labels = [f"t{n}" for n in T_vals]   # "t1" .. "t163"

    # ------------------------------------------------------------------
    # Container GAMSPy
    # ------------------------------------------------------------------
    m = gp.Container()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    i_set = gp.Set(m, "i", records=assets,           description="activos")
    j_set = gp.Alias(m, "j", i_set)                  # alias para covarianza
    t_set = gp.Set(m, "t", records=T_labels,          description="periodos")

    # ------------------------------------------------------------------
    # Parámetros
    # ------------------------------------------------------------------
    # mu_mix con theta aplicado
    mu_records = [
        [i, f"t{t}", mu_base[i].loc[t] * theta[i]]
        for i in assets for t in T_vals
    ]
    mu_p = gp.Parameter(
        m, "mu_mix", domain=[i_set, t_set],
        records=pd.DataFrame(mu_records, columns=["i", "t", "value"]),
        description="media mixta por periodo (con theta)",
    )

    # sigma_mix
    sig_records = [
        [i, j, f"t{t}", sigma_mix[i][j].loc[t]]
        for i in assets for j in assets for t in T_vals
    ]
    sig_p = gp.Parameter(
        m, "sigma_mix", domain=[i_set, j_set, t_set],
        records=pd.DataFrame(sig_records, columns=["i", "j", "t", "value"]),
        description="covarianza mixta por periodo",
    )

    # costos efectivos c_eff(i) = c_base(i) * costo_mult
    c_eff_p = gp.Parameter(
        m, "c_eff", domain=[i_set],
        records=pd.DataFrame(
            [[i, c_base[i] * costo_mult] for i in assets],
            columns=["i", "value"],
        ),
        description="costo efectivo en FO",
    )

    # costos base para simulación de capital (siempre sin multiplicador)
    c_base_p = gp.Parameter(
        m, "c_base", domain=[i_set],
        records=pd.DataFrame(
            [[i, c_base[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="costo base de transacción",
    )

    # portafolio inicial
    w0_p = gp.Parameter(
        m, "w0", domain=[i_set],
        records=pd.DataFrame(
            [[i, w0_dict[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="portafolio inicial 50/50",
    )

    # aversión al riesgo (escalar)
    lam_p = gp.Parameter(m, "lambda_riesgo", records=lambda_riesgo,
                          description="aversión al riesgo")

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    z_var = gp.Variable(m, "z",                              description="valor objetivo")
    w_var = gp.Variable(m, "w", domain=[i_set, t_set], type="positive", description="peso")
    u_var = gp.Variable(m, "u", domain=[i_set, t_set], type="positive", description="compras")
    v_var = gp.Variable(m, "v", domain=[i_set, t_set], type="positive", description="ventas")

    w_var.up[i_set, t_set] = 1.0   # w(i,t) <= 1

    # ------------------------------------------------------------------
    # Ecuaciones
    # ------------------------------------------------------------------
    # 1. Función objetivo
    fo = gp.Equation(m, "FO_media_var_costo",
                     description="FO: retorno - lambda*var - costos")
    fo[...] = z_var == gp.Sum(
        t_set,
        gp.Sum(i_set, w_var[i_set, t_set] * mu_p[i_set, t_set])
        - lam_p * gp.Sum((i_set, j_set),
                         w_var[i_set, t_set] * w_var[j_set, t_set] * sig_p[i_set, j_set, t_set])
        - gp.Sum(i_set, c_eff_p[i_set] * (u_var[i_set, t_set] + v_var[i_set, t_set]))
    )

    # 2. Normalización: sum_i w(i,t) = 1
    norm = gp.Equation(m, "normalizacion_pesos", domain=[t_set],
                       description="suma de pesos = 1")
    norm[t_set] = gp.Sum(i_set, w_var[i_set, t_set]) == 1

    # 3. Rebalanceo: w(i,t) - w(i,t-1) = u(i,t) - v(i,t)  para t > t1
    rebal = gp.Equation(m, "rebalanceo_lineal", domain=[i_set, t_set],
                        description="identidad de rebalanceo")
    rebal[i_set, t_set].where[gp.Ord(t_set) > 1] = (
        w_var[i_set, t_set] - w_var[i_set, t_set.lag(1)]
        == u_var[i_set, t_set] - v_var[i_set, t_set]
    )

    # 4. Anclaje inicial: w(i,t1) - w0(i) = u(i,t1) - v(i,t1)
    anclaje = gp.Equation(m, "anclaje_inicial", domain=[i_set],
                          description="anclaje al portafolio inicial")
    anclaje[i_set] = (
        w_var[i_set, "t1"] - w0_p[i_set]
        == u_var[i_set, "t1"] - v_var[i_set, "t1"]
    )

    # ------------------------------------------------------------------
    # Modelo y resolución
    # ------------------------------------------------------------------
    portfolio = gp.Model(
        m,
        name="PortafolioEstadosCostos",
        equations=m.getEquations(),
        problem="NLP",
        sense=gp.Sense.MAX,
        objective=z_var,
    )

    output = None if not verbose else __import__("sys").stdout
    portfolio.solve(solver="IPOPT", output=output)

    if portfolio.status not in (
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.OptimalGlobal,
    ):
        raise RuntimeError(
            f"GAMSPy/IPOPT no encontró solución óptima. Status: {portfolio.status}"
        )

    z_val = float(z_var.toValue())

    # Extraer resultados — convertir etiqueta "t1".."t163" a entero
    def _records_to_dict(var):
        sol = {}
        for _, row in var.records.iterrows():
            i_key = row["i"]
            t_key = int(row["t"][1:])   # "t3" -> 3
            sol[i_key, t_key] = float(row["level"])
        return sol

    w_sol = _records_to_dict(w_var)
    u_sol = _records_to_dict(u_var)
    v_sol = _records_to_dict(v_var)

    status = ("optimal" if portfolio.status == gp.ModelStatus.OptimalGlobal
              else "optimal_local")
    return z_val, w_sol, u_sol, v_sol, status


# ================================================================
# 2) Simulación Ex-Post de Capital (Sección 4 del GAMS)
# ================================================================

def simulate_capital_opt(w_sol, u_sol, v_sol, context):
    """
    cap(t) = cap(t-1)*(1 + r_port(t-1)) - cap(t-1)*sum_i c_base(i)*(u(i,t)+v(i,t))
    Replica exactamente el LOOP de la sección 4 del GAMS.
    Siempre usa c_base (sin multiplicador) en la simulación de capital.
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
        turn   = sum(c_base[i] * (u_sol[i, t] + v_sol[i, t]) for i in assets)
        cap[t] = cap[t_prev] * (1 + r_port) - cap[t_prev] * turn
    return cap


def simulate_naive_bh(context):
    """Naive Buy & Hold 50/50 sin rebalanceo ni costos."""
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
        cap[t] = cap[t_prev] * (1 + r_port)
    return cap


def simulate_naive_rb(context):
    """Naive 50/50 con rebalanceo semanal y costos de transacción."""
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
        w_bh   = {i: w_target * (1 + r[i].loc[t_prev]) / (1 + r_port) for i in assets}
        turn   = sum(c_base[i] * abs(w_target - w_bh[i]) for i in assets)
        cap[t] = cap[t_prev] * (1 + r_port) - cap[t_prev] * turn
    return cap


# ================================================================
# 3) Análisis de Sensibilidad (Sección 5 del GAMS)
# ================================================================

def run_sensitivity_grid(context):
    """
    Grid lambda (5) x c_mult (3) = 15 combinaciones.
    Replica el LOOP(L, LOOP(C, ...)) de la sección 5 del GAMS.
    """
    assets      = context["assets"]
    T_vals      = context["T_vals"]
    Capital_ini = context["Capital_inicial"]
    theta_neu   = {a: 1.0 for a in assets}

    lambda_grid = [0.05, 0.10, 0.20, 0.50, 1.00]
    c_mult_grid = [0.5,  1.0,  2.0]
    labels_L    = ["L1", "L2", "L3", "L4", "L5"]
    labels_C    = ["C1", "C2", "C3"]

    rows  = []
    total = len(lambda_grid) * len(c_mult_grid)
    run   = 0
    for li, lam in enumerate(lambda_grid):
        for ci, cm in enumerate(c_mult_grid):
            run += 1
            print(f"  [{run:>2}/{total}] {labels_L[li]}/{labels_C[ci]}  "
                  f"lambda={lam:.2f}  c_mult={cm:.1f} ...", end=" ", flush=True)
            z, w_sol, u_sol, v_sol, _ = solve_portfolio(
                theta_neu, context, lambda_riesgo=lam, costo_mult=cm
            )
            cap       = simulate_capital_opt(w_sol, u_sol, v_sol, context)
            cap_final = cap[T_vals[-1]]
            ret_acum  = cap_final / Capital_ini - 1
            print(f"z={z:.6f}  cap_final=${cap_final:,.2f}  ret={ret_acum:+.2%}")
            rows.append({
                "L": labels_L[li], "C": labels_C[ci],
                "lambda": lam,     "c_mult": cm,
                "z":         round(z, 6),
                "cap_final": round(cap_final, 2),
                "ret_acum":  round(ret_acum, 6),
            })

    return pd.DataFrame(rows)


# ================================================================
# 4) Bloque Principal
# ================================================================

if __name__ == "__main__":
    print("Cargando datos...")
    base_path_str = str(Path(__file__).parent)
    context = load_market_data(base_path_str)
    assets  = context["assets"]
    T_vals  = context["T_vals"]
    print(f"Datos cargados: {len(T_vals)} periodos, activos: {assets}\n")

    # ------------------------------------------------------------------
    # Caso Base: Neutral (lambda=0.10, c_mult=1.0, theta=1.0)
    # ------------------------------------------------------------------
    print("=" * 65)
    print("CASO BASE — Neutral (theta=1.0, lambda=0.10, c_mult=1.0)")
    print("=" * 65)
    theta_neutral = {a: 1.0 for a in assets}
    z_neu, w_neu, u_neu, v_neu, status_neu = solve_portfolio(
        theta_neutral, context, lambda_riesgo=0.10
    )
    print(f"  Status : {status_neu}")
    print(f"  z      : {z_neu:.6f}")

    cap_opt = simulate_capital_opt(w_neu, u_neu, v_neu, context)
    cap_bh  = simulate_naive_bh(context)
    cap_rb  = simulate_naive_rb(context)

    C0 = context["Capital_inicial"]
    def summary(label, cap):
        cf = cap[T_vals[-1]]
        return f"  {label:<30}  ${cf:>12,.2f}  {cf/C0-1:>+8.2%}  {cf-C0:>+12,.2f}"

    print("\n--- cap_opt / cap_naive_rb / cap_naive_bh ---")
    print(f"  {'':30}  {'cap_final':>12}  {'ret_acum':>8}  {'inc_cap':>12}")
    print(f"  {'-'*65}")
    print(summary("Óptimo (cap_opt)",       cap_opt))
    print(summary("Naive 50/50 Rebalanceo", cap_rb))
    print(summary("Naive Buy & Hold",       cap_bh))

    print(f"\n  ret_acum_opt      = {cap_opt[T_vals[-1]]/C0 - 1:+.6f}")
    print(f"  ret_acum_naive_rb = {cap_rb[T_vals[-1]]/C0 - 1:+.6f}")
    print(f"  ret_acum_naive_bh = {cap_bh[T_vals[-1]]/C0 - 1:+.6f}")
    print(f"  inc_cap_opt       = {cap_opt[T_vals[-1]] - C0:+.6f}")
    print(f"  inc_cap_naive_rb  = {cap_rb[T_vals[-1]] - C0:+.6f}")
    print(f"  inc_cap_naive_bh  = {cap_bh[T_vals[-1]] - C0:+.6f}")

    print("\n--- w.l (primeros 10 periodos) ---")
    print(f"  {'t':>5}  {'w_SPX':>8}  {'w_CMC200':>10}  {'u_SPX':>8}  {'v_SPX':>8}")
    for t in T_vals[:10]:
        print(f"  {t:>5}  {w_neu['SPX',t]:>8.6f}  {w_neu['CMC200',t]:>10.6f}"
              f"  {u_neu['SPX',t]:>8.6f}  {v_neu['SPX',t]:>8.6f}")

    # ------------------------------------------------------------------
    # Caso Bullish SPX
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("CASO BULLISH SPX — theta_SPX=1.1, lambda=0.10, c_mult=1.0")
    print("=" * 65)
    theta_bull = {a: 1.0 for a in assets}
    theta_bull["SPX"] = 1.10
    z_bull, w_bull, u_bull, v_bull, _ = solve_portfolio(
        theta_bull, context, lambda_riesgo=0.10
    )
    cap_bull = simulate_capital_opt(w_bull, u_bull, v_bull, context)
    print(f"  z      : {z_bull:.6f}")
    print(summary("Óptimo Bullish SPX", cap_bull))

    # ------------------------------------------------------------------
    # Grid de Sensibilidad
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SENSIBILIDAD — Grid lambda (L1-L5) x c_mult (C1-C3)")
    print("=" * 65)
    df_grid = run_sensitivity_grid(context)

    print("\n--- res(L,C,'z') ---")
    print(df_grid.pivot(index="L", columns="C", values="z").to_string())

    print("\n--- res(L,C,'cap_final') ---")
    pivot_cap = df_grid.pivot(index="L", columns="C", values="cap_final")
    print(pivot_cap.to_string(float_format="  ${:,.2f}".format))

    print("\n--- res(L,C,'ret_acum') ---")
    pivot_ret = df_grid.pivot(index="L", columns="C", values="ret_acum")
    print(pivot_ret.map(lambda x: f"{x:+.6f}").to_string())

    out_path = Path(base_path_str) / "sensitivity_results_gams.csv"
    df_grid.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")
