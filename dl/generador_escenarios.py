"""Generador de escenarios de retornos futuros (PDF sección 2.5).

Pipeline en dos pasos:

1. `generate_candidate_scenarios`: desde la última ventana observada, simula
   N trayectorias (por defecto N = 1000) de largo T (por defecto `T_HORIZON`,
   que coincide con t1..t163 del GAMS). En cada paso:
     - se predicen los deciles con el LSTM congelado;
     - se muestrea uniformemente un nivel q ∈ Q **comun a todos los activos**
       (lectura literal del PDF). El q independiente por activo se probo y
       descartado: rompia la correlacion SPX-CMC a ~0 (historico +0.31), lo
       que generaba escenarios "imposibles" (SPX -40% mientras CMC +1000%)
       que secuestraban la seleccion por regret. Mismo q da corr ~0.85
       (mas cerca del historico que 0) y escenarios ordenados de peor a
       mejor para ambos activos;
     - se fija r_cand_{i,t} = r_hat^(q_i)_{i,t};
     - se rola la ventana (se descarta el retorno más viejo y se agrega el
       nuevo) para poder predecir el paso siguiente.

2. `reduce_to_representatives`: ordena los N candidatos por un resumen
   económico (retorno acumulado del activo de referencia — SPX por defecto),
   los parte en 5 quintiles del peor al mejor, y elige 1 escenario mediano
   por quintil. Resultado: S con |S| = 5 trayectorias explicables que
   alimentan el regret-grid de ps.gms."""

from typing import Optional

import numpy as np
import torch

from config import T_HORIZON
from .prediccion_deciles import LoadedModel


def generate_candidate_scenarios(
    model: LoadedModel,
    initial_window: np.ndarray,
    N: int = 1000,
    T: int = T_HORIZON,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Genera N trayectorias de largo T partiendo de `initial_window`.

    initial_window: (H, n_assets)
    return:         (N, T, n_assets)
    """
    cfg = model.config
    if initial_window.shape != (cfg.H, cfg.n_assets):
        raise ValueError(
            f"initial_window shape {initial_window.shape} != (H={cfg.H}, A={cfg.n_assets})"
        )

    rng = np.random.default_rng(seed)
    A   = cfg.n_assets
    Q   = cfg.n_quantiles

    # Una ventana independiente por escenario (todas parten iguales).
    windows   = np.tile(initial_window.astype(np.float32), (N, 1, 1))   # (N, H, A)
    scenarios = np.empty((N, T, A), dtype=np.float32)

    for t in range(T):
        # Predice deciles para los N escenarios en paralelo (ensemble de seeds).
        x = ((windows - model.mean) / model.std).astype(np.float32)
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            preds_list = [net(x_tensor).numpy() for net in model.nets]  # K * (N, A, Q)
        preds = np.mean(np.stack(preds_list, axis=0), axis=0)
        # Garantizamos monotonicidad: q_idx=0 debe ser el peor caso para cada activo.
        preds = np.sort(preds, axis=-1)

        # Mismo q para todos los activos en cada paso (lectura literal del PDF).
        # El q independiente por activo se descarto: daba corr SPX-CMC ~0 y
        # generaba el escenario artefacto (SPX cae / CMC explota) que
        # secuestraba la seleccion por regret. Mismo q => corr ~0.85.
        q_idx = np.repeat(rng.integers(low=0, high=Q, size=(N, 1)),
                          A, axis=1)                                    # (N, A) comonotonia
        r_t   = np.take_along_axis(
            preds, q_idx[:, :, None], axis=2
        ).squeeze(-1)                                                   # (N, A)

        scenarios[:, t, :] = r_t

        # Roll de la ventana: desecha el retorno mas viejo, agrega el recien muestreado.
        windows = np.concatenate([windows[:, 1:, :], r_t[:, None, :]], axis=1)

    return scenarios


def reduce_to_representatives(
    scenarios: np.ndarray,
    summary_asset_idx: int = 0,
    n_quintiles: int = 5,
    position: str = "median",
) -> np.ndarray:
    """
    Reduce los N candidatos a n_quintiles representativos.

    scenarios:         (N, T, n_assets)
    summary_asset_idx: indice del activo usado como resumen economico
                       (PDF ec. 17, default SPX = 0).
    position:          que escenario tomar dentro de cada quintil.
                       - "median" (default, ejemplo del PDF): el mediano del bucket
                       - "min":    el peor del bucket (mas pesimista en cada quintil)
                       - "max":    el mejor del bucket
    return:            (n_quintiles, T, n_assets)
    """
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios debe ser (N, T, A); recibi {scenarios.shape}")
    N = scenarios.shape[0]
    if N < n_quintiles:
        raise ValueError(f"N={N} insuficiente para {n_quintiles} quintiles")
    if position not in {"median", "min", "max"}:
        raise ValueError(f"position invalido: {position!r}")

    # Retorno acumulado del activo resumen por escenario (PDF ec. 17).
    cum   = np.prod(1.0 + scenarios[:, :, summary_asset_idx], axis=1) - 1.0  # (N,)
    order = np.argsort(cum)                                                  # peor -> mejor

    # Particion en n_quintiles (el ultimo absorbe el remanente si N no divide).
    edges = np.linspace(0, N, n_quintiles + 1, dtype=int)
    reps  = []
    for k in range(n_quintiles):
        lo, hi = edges[k], edges[k + 1]
        bucket = order[lo:hi]
        if   position == "median": idx = bucket[len(bucket) // 2]
        elif position == "min":    idx = bucket[0]
        else:                      idx = bucket[-1]   # max
        reps.append(scenarios[idx])

    return np.stack(reps, axis=0)                                            # (n_q, T, A)


def generate_representative_scenarios(
    model: LoadedModel,
    initial_window: np.ndarray,
    N: int = 1000,
    T: int = T_HORIZON,
    n_quintiles: int = 5,
    summary_asset: str = "SPX",
    seed: Optional[int] = None,
    position: str = "median",
) -> np.ndarray:
    """Pipeline completo: genera N candidatos y devuelve n_quintiles representativos."""
    assets = tuple(model.config.assets)
    if summary_asset not in assets:
        raise ValueError(f"summary_asset {summary_asset!r} no esta en {assets}")
    idx = assets.index(summary_asset)

    candidates = generate_candidate_scenarios(
        model, initial_window, N=N, T=T, seed=seed,
    )
    return reduce_to_representatives(
        candidates, summary_asset_idx=idx, n_quintiles=n_quintiles, position=position,
    )
