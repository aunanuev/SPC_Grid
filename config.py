"""Configuracion unica del proyecto SPC_Grid3.

Todas las perillas (hiperparametros + variables de ejecucion que cambian el
resultado final) viven aqui. Cualquier modulo del proyecto debe importar
*desde este archivo* — tanto `basemodelGAMS.py` y `Regret_Grid.py` como los
modulos de `dl/` y los scripts de `inspeccion/`.

Organizacion:
  1. Paths del proyecto
  2. Universo de activos y esquemas de CSV
  3. Horizonte y regla de regimen (bull/bear)
  4. Deep Learning        -> DLConfig         (deciles, LSTM, entrenamiento)
  5. Generador escenarios -> ScenarioConfig   (N, n_escenarios, summary, seed)
  6. Mercado / Portafolio -> capital, w0, costos base
  7. Optimizador          -> OptConfig        (lambda, costo_mult, solver)
  8. Regret-grid          -> RegretGridConfig (grilla lambda x m)

Convencion: las constantes en MAYUSCULAS son los *defaults* globales;
los dataclasses las toman como valor inicial para poder sobrescribirlas
puntualmente en tiempo de ejecucion sin tocar este archivo.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


# =====================================================================
# 1) Paths
# =====================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR:     Path = PROJECT_ROOT / "data"
MODELS_DIR:   Path = PROJECT_ROOT / "models"
RESULTS_DIR:  Path = PROJECT_ROOT / "resultados"

CHECKPOINT_NAME: str  = "decile_predictor.pt"
CHECKPOINT_PATH: Path = MODELS_DIR / CHECKPOINT_NAME


# =====================================================================
# 2) Universo de activos + esquemas de CSV
# =====================================================================
ASSETS:  Tuple[str, ...] = ("SPX", "CMC200")
REGIMES: Tuple[str, ...] = ("bear", "bull")

# CSV de retornos semanales (carpeta DATA_DIR).
RETURN_CSV: Dict[str, str] = {
    "SPX":    "ret_semanal_spx.csv",
    "CMC200": "ret_semanal_cmc200.csv",
}
# Nombre de la columna que contiene el retorno dentro de cada CSV de retornos.
RETURN_COL: Dict[str, str] = {
    "SPX":    "ret_semanal_spx",
    "CMC200": "ret_semanal_cmc200",
}
# CSV de probabilidades historicas (una columna por regimen, indexado por t).
PROB_CSV: Dict[str, str] = {
    "SPX":    "prob_spx.csv",
    "CMC200": "prob_cmc200.csv",
}


# =====================================================================
# 3) Horizonte + regla de regimen
# =====================================================================
# Largo del horizonte forward (coincide con t1..t163 del GAMS base).
T_HORIZON: int = 163

# Umbral para clasificar un retorno como "bull": r >= BULL_THRESHOLD.
BULL_THRESHOLD: float = 0.0


# =====================================================================
# 4) Deep Learning (LSTM cuantilica + entrenamiento)
# =====================================================================
# Deciles objetivo del LSTM (PDF ec. 12, adaptacion de los quintiles).
DECILES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

# Hiperparametros por defecto (reflejados en DLConfig).
H_WINDOW:     int   = 60       # largo de la ventana del LSTM (PDF sec. 2.2 ejemplo). H=4 mejora pinball test pero hace que el rollout autoregresivo explote (SPX p_bull=100%, ver inspeccion/generador_escenarios/probe_H_intermedio.py).
LSTM_HIDDEN:  int   = 16       # sweep_modelos: unico modelo con test_skill > 0 (+0.0137)
LSTM_LAYERS:  int   = 1        # sweep_modelos: 1 capa generaliza mejor que 2 (5x menos params)
DROPOUT:      float = 0.1
LR:           float = 1e-3
WEIGHT_DECAY: float = 1e-4
EPOCHS:       int   = 300
PATIENCE:     int   = 15
BATCH_SIZE:   Optional[int] = None   # None => batch completo
SPLIT:        Tuple[float, float, float] = (0.70, 0.15, 0.15)
SEEDS:        Tuple[int, ...]            = (0, 1, 2)
DEVICE:       str   = "cpu"


@dataclass
class DLConfig:
    """Parametros de la LSTM cuantilica y su entrenamiento."""
    H:            int                    = H_WINDOW
    quantiles:    Tuple[float, ...]      = DECILES
    assets:       Tuple[str, ...]        = ASSETS
    lstm_hidden:  int                    = LSTM_HIDDEN
    lstm_layers:  int                    = LSTM_LAYERS
    dropout:      float                  = DROPOUT
    lr:           float                  = LR
    weight_decay: float                  = WEIGHT_DECAY
    epochs:       int                    = EPOCHS
    patience:     int                    = PATIENCE
    batch_size:   Optional[int]          = BATCH_SIZE
    split:        Tuple[float, float, float] = SPLIT
    seeds:        Tuple[int, ...]        = SEEDS
    device:       str                    = DEVICE

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def n_assets(self) -> int:
        return len(self.assets)


# =====================================================================
# 5) Generador de escenarios (seccion 2.5)
# =====================================================================
N_CANDIDATES:   int = 1000
N_SCENARIOS:    int = 5            # quintiles representativos
SUMMARY_ASSET:  str = "SPX"        # activo usado para ordenar escenarios
SCENARIO_SEED:  int = 0
# "median" (PDF default), "min" (mas pesimista por quintil), "max"
SCENARIO_POSITION: str = "median"


@dataclass
class ScenarioConfig:
    N_candidates:  int = N_CANDIDATES
    n_scenarios:   int = N_SCENARIOS
    summary_asset: str = SUMMARY_ASSET
    T:             int = T_HORIZON
    seed:          int = SCENARIO_SEED
    position:      str = SCENARIO_POSITION


# =====================================================================
# 6) Mercado / Portafolio (compartido por base y regret-grid)
# =====================================================================
CAPITAL_INICIAL: float = 10_000.0

# Costos de transaccion por activo (fraccion).
C_BASE: Dict[str, float] = {"SPX": 0.001, "CMC200": 0.004}

# Portafolio inicial (suma 1).
W0: Dict[str, float] = {"SPX": 0.5, "CMC200": 0.5}

# Presupuesto de riesgo V_max para la FO.
# V_max = Var(returns historicos de V_MAX_REF_ASSET) * V_MAX_BUFFER.
# El activo de referencia es el "estable" (SPX) y el buffer da holgura sobre
# su varianza muestral; entra en la FO como penalizacion lambda*(Riesgo - V_max).
V_MAX_REF_ASSET: str   = "SPX"
V_MAX_BUFFER:    float = 1.2


# =====================================================================
# 7) Optimizador (defaults del solve)
# =====================================================================
LAMBDA_RIESGO_DEFAULT: float = 0.10
COSTO_MULT_DEFAULT:    float = 1.0
SOLVER:                str   = "IPOPT"


@dataclass
class OptConfig:
    lambda_riesgo: float = LAMBDA_RIESGO_DEFAULT
    costo_mult:    float = COSTO_MULT_DEFAULT
    solver:        str   = SOLVER
    verbose:       bool  = False


# =====================================================================
# 8) Regret-grid (G = Lambda x M)
# =====================================================================
# LAMBDA_GRID: rango ampliado tras inspeccion (`inspeccion/grid.py`).
#   Con [0.3, 1.5] g*_mean tocaba la esquina superior; el worst-case V se
#   satura cerca de lambda = 2.0. M_GRID refinado entre 0.01 y 0.5 para
#   capturar el "kink" de turnover: m=0.01 deja casi 6x mas rebalanceo que
#   m=0.2, mientras que m=0.2 y m=0.5 son indistinguibles.
# IMPORTANTE: M_GRID debe ser estrictamente > 0. Con m=0 el termino c*(u+v)
# del FO se anula y IPOPT puede devolver soluciones degeneradas (ver fix en
# `solve_portfolio` que ahora acota u, v <= 1).
LAMBDA_GRID: Tuple[float, ...] = (0.3, 0.9, 1.2, 1.5, 1.8)
M_GRID:      Tuple[float, ...] = (0.1,  0.5, 1)


@dataclass
class RegretGridConfig:
    lambda_grid: Tuple[float, ...] = LAMBDA_GRID
    m_grid:      Tuple[float, ...] = M_GRID
