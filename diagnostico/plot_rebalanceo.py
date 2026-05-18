"""Genera un grafico extra para diagnostico/figuras/ que muestra como se
va rebalanceando el portafolio a lo largo del tiempo bajo cada una de las
4 politicas guardadas en `inspeccion/grid_out/4_politicas_w.csv`.

El plot es un panel 4x1: cada fila es una politica distinta, y cada subplot
muestra:
  - stacked area chart de w_SPX(t) (azul) + w_CMC200(t) (naranja),
    visualizando como se reparte el 100% del portafolio en cada semana
  - debajo, una linea negra de turnover por paso = |Delta w_SPX(t)| (el
    turnover por activo es identico al del otro porque sum_i w(i,t) = 1).

Salida: diagnostico/figuras/21_rebalanceo_portafolio.png
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = PROJECT_ROOT / "inspeccion" / "grid_out" / "4_politicas_w.csv"
OUT_PATH = PROJECT_ROOT / "diagnostico" / "figuras" / "21_rebalanceo_portafolio.png"

# Nombre humano por label. g*_mean puede coincidir con una esquina del grid
# (p.ej. si g*_mean = (lambda_min, m_min) no se escribe low_lambda_low_m
# porque la clave colisiona con g_mean en diag_politicas). Por eso el set de
# labels se deriva de lo que realmente esta en el CSV, no de una lista fija.
HUMAN = {
    "g_mean":             "g*_mean (seleccionada)",
    "low_lambda_low_m":   "low lambda, low m",
    "low_lambda_high_m":  "low lambda, high m",
    "high_lambda_low_m":  "high lambda, low m",
    "high_lambda_high_m": "high lambda, high m",
}
# Orden de presentacion (g*_mean primero); solo se grafican los presentes.
ORDER = ["g_mean", "low_lambda_low_m", "low_lambda_high_m",
         "high_lambda_low_m", "high_lambda_high_m"]


def main():
    df = pd.read_csv(CSV_PATH)
    # wide: index=t, columnas=(label, asset)
    wide = df.pivot_table(
        index="t", columns=["label", "asset"], values="w"
    ).sort_index()

    present = set(df["label"].unique())
    LABELS = [(lbl, HUMAN.get(lbl, lbl)) for lbl in ORDER if lbl in present]

    fig, axes = plt.subplots(
        len(LABELS), 1, figsize=(11, 2.75 * len(LABELS)), sharex=True,
    )
    if len(LABELS) == 1:
        axes = [axes]
    for ax, (label, human) in zip(axes, LABELS):
        w_spx = wide[(label, "SPX")].values
        w_cmc = wide[(label, "CMC200")].values
        T = wide.index.values

        # stacked area: SPX abajo, CMC encima
        ax.fill_between(T, 0.0,  w_spx, color="#1f77b4", alpha=0.7, label="w_SPX")
        ax.fill_between(T, w_spx, w_spx + w_cmc, color="#ff7f0e", alpha=0.7, label="w_CMC200")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("peso w(t)")

        # info de lambda/m + turnover por activo (cualquiera de los dos)
        lam, m = df[df["label"] == label][["lambda", "m"]].iloc[0]
        delta_w = np.abs(np.diff(w_spx, prepend=0.5))  # w0=0.5
        turnover_total_one_asset = delta_w.sum()
        ax.set_title(
            f"{human}  —  lambda={lam:.2f}, m={m:.2f}  —  "
            f"turnover SPX (sum_t |delta w|) = {turnover_total_one_asset:.1f}"
        )

        # linea de turnover por paso (eje secundario)
        ax2 = ax.twinx()
        ax2.plot(T, delta_w, color="black", lw=0.6, alpha=0.7, label="|delta w_SPX|")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("|Δw(t)|", color="black")

        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("t (semana, t1..t163)")
    fig.suptitle(
        f"Rebalanceo del portafolio bajo {len(LABELS)} politicas\n"
        "(area apilada = composicion; linea negra = magnitud del rebalanceo en cada paso)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)
    print(f"OK -> {OUT_PATH}")


if __name__ == "__main__":
    main()
