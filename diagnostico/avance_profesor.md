# Avance SPC_Grid — resumen para reunion (20 min)

Pipeline media-varianza con costos (GAMS original) + capa de Deep Learning
que predice retornos y genera escenarios + seleccion de hiperparametros por
**regret**. Configuracion evaluada: `LSTM h=16, layers=1, ventana=60`
(ganadora del sweep por pinball loss).

---

## Estado del pipeline tras la ultima iteracion

Se reescribio `build_dl_context` para que **siga la descomposicion por
regimen del PDF (ec. 2-5)**, usando como fuente de `p_{i,k,t}` la prediccion
del LSTM aplicada a la ventana real previa (LSTM walking). La version
anterior calculaba `mu_mix` y `Sigma_mix` como momentos muestrales sobre los
1000 candidatos del rollout autoregresivo — formalmente correcta pero
desalineada con la estructura del PDF.

```
LSTM cuantilico ──► p_{i,k,t} = LSTM walking (ec. 15)
                    │
                    ▼
                    ec. (2)-(3): mu_hat[i,k], Sigma_hat[i,j,k]
                    │
                    ▼
                    ec. (4)-(5): mu_mix(i,t), Sigma_mix(i,j,t)   ──►  FO
                                                                       │
LSTM cuantilico ──► rollout autoregresivo (N=1000)                     │
                    │                                                  │
                    ▼                                                  │
                    5 escenarios representativos por quintiles ──► simulacion ex-post
                                                                       │
                                                                       ▼
                                                          regret en 5 escenarios reps
                                                          => g*_mean y g*_worst
                                                                       │
                                                                       ▼
                                                          Backtest sobre realidad historica
```

Que se gano con el cambio (versus version anterior unificada):
- **Anti-timing invertido**: `corr(mu_DL, real)` paso de `-0.13` a `+0.11` para CMC200.
- **Bias eliminado** por construccion matematica.
- **Turnover -29x** en la esquina patologica del grid.
- **Backtest historico mejora +8 pp** (`-14.81%` → `-6.63%`).

Que sigue siendo problema:
- El backtest sigue perdiendo contra las naive (`+20%` a `+25%`).
- Los inputs DL resultan **practicamente identicos a OPT base**, lo que cuestiona si el LSTM aporta valor sobre la formulacion clasica del PDF.

---

## 1) LSTM cuantilico — skill marginal pero ya no anti-timing

![skill pinball por split](figuras/01_lstm_skill.png)

**Que muestra**: mejora porcentual del LSTM respecto al baseline naive
(deciles empiricos in-window), por split y activo.

**Lectura**: gana apenas en test (`+0.8%` SPX, `+2%` CMC). En validacion
pierde. La skill estadistica es marginal pero existe.

![mu_DL vs realizado](figuras/12_inputs_mu_vs_real.png)

**Que muestra**: `mu_mix(t)` que ve la FO comparada con el retorno
realizado en cada `t` historico.

**Lectura clave**: el problema central del diagnostico anterior se
revirtio.

| activo | corr(mu_DL, real) antes | ahora | hit-rate signo antes | ahora | bias antes | ahora |
|---|---:|---:|---:|---:|---:|---:|
| SPX | -0.030 | **+0.070** | 53.4% | 55.8% | -0.017% | ≈0 |
| CMC200 | **-0.133** | **+0.109** | 49.7% | 53.4% | -0.193% | ≈0 |

El bias desaparece por construccion: `Σ_k p_{i,k,t}·μ̂_{i,k}` promediado
sobre `t` reproduce la media empirica de `r`. La inversion del `corr` es el
sintoma de que la FO ya no esta apostando contra la direccion del retorno
realizado.

---

## 2) Escenarios — el sesgo de CMC sigue presente

![sesgo cumret terminal](figuras/05_escen_sesgo_terminal.png)

**Que muestra**: retorno acumulado a 163 semanas — distribucion de los
1000 candidatos del rollout, los 5 representativos, y la realidad
historica.

**Lectura**:
- **SPX**: candidatos `+38%`, reps `+34%`, real `+30%`. Coherente.
- **CMC200**: candidatos `+52%` promedio, reps `+193%`, real `-13%`. La
  cola derecha del rollout arrastra los reps hacia arriba.

El generador de escenarios **no cambio** en esta iteracion — los reps
siguen viniendo del rollout autoregresivo. El sesgo de magnitud de CMC
persiste y todavia hay que decidir si se ataca (shrinkage, reduccion
ponderada por verosimilitud, etc.) o si se acepta como caracteristica.

---

## 3) Optimizador — politica civilizada

![rebalanceo del portafolio](figuras/21_rebalanceo_portafolio.png)

**Que muestra**: composicion del portafolio (area apilada: azul=SPX,
naranja=CMC) y magnitud del rebalanceo (linea negra) para 4 politicas
representativas del grid.

**Lectura**: la patologia desaparecio. Comparativa de turnover total
(suma de `u(t) + v(t)` sobre 163 semanas):

| (lambda, m) | antes | ahora |
|---|---:|---:|
| (0.30, 0.1) | **85.80** | **2.95** |
| (0.30, 0.5) seleccionada `g*_mean` | 13.84 | 0.85 |
| (1.80, 0.1) | 23.66 | 1.18 |
| (1.80, 0.5) | 2.97 | 0.77 |

Antes la politica rotaba `~50%` semanal en el rincon sin friccion;
ahora rota `~2%`. La causa: `mu_mix(t)` paso de ser ruido muestral (cada
`t` su propia media de 1000 candidatos) a una mezcla suave
`Σ_k p_t,k · μ̂_k` que oscila mucho menos.

---

## 4) Backtest — mejora 8 pp, pero sigue perdiendo

![DL vs OPT base por escenario](figuras/20_dl_vs_optbase.png)

**Que muestra**: capital terminal por escenario para la politica DL
seleccionada vs la solucion "OPT base" (formulacion clasica del PDF con
`p` del CSV legado).

**Sobre los 5 escenarios DL**:
- Promedio: DL `+117%` vs OPT base `+91%`.
- Peor escenario: DL `-82%` vs OPT base `-59%`.

**Sobre la realidad historica**:

| Politica | V terminal | retorno |
|---|---:|---:|
| OPT base (lambda=1, m=1) | $14,522 | +45.22% |
| Naive Buy-and-Hold 50/50 | $12,258 | +22.58% |
| Naive Rebalance 50/50 | $12,092 | +20.92% |
| **Regret-Grid DL `g*_mean`** | **$9,337** | **-6.63%** |

**Lectura**: el modelo "optimo" segun el regret-grid mejora `+8 pp`
respecto a la version anterior (`-14.81%`), pero todavia pierde plata
sobre el historico real. Las dos naive lo superan por `>27 pp`, y el OPT
base (sin DL) por `>50 pp`. El LSTM **achica la perdida pero no la
elimina**.

---

## Sintesis y caminos para discutir

**Diagnostico actual**: la reescritura del pipeline siguiendo el PDF
soluciono los tres sintomas mas graves (anti-timing, bias, turnover
patologico). El backtest mejora pero sigue siendo negativo.

**Hallazgo conceptual nuevo**: los inputs DL ahora son **practicamente
identicos a OPT base** (mismas medias, sigmas muy similares, Sharpe casi
igual). Tiene sentido matematicamente: con `μ̂_{i,k}` calculado sobre
historico, la fluctuacion temporal que aporta el LSTM vive solo en
`p_{i,k,t}`. Si esa `p` no captura algo no contenido en los medias
historicas, el modelo DL "se convierte en OPT base".

**Preguntas abiertas para la reunion**:

1. **¿El LSTM aporta valor sobre OPT base?** Con 163 datos semanales y
   este resultado, la respuesta empirica es: "muy poco o nada en el
   backtest, aunque corrige direccionalmente la prediccion paso a paso".
   Caminos: cambiar a frecuencia diaria, aceptar que el LSTM es
   decorativo en este dataset, o cambiar la arquitectura para que aporte
   algo mas que `p_bull` (por ejemplo predecir conjunto activo correlado
   en regimenes alternativos).

2. **¿Se acepta la grieta ex-ante/ex-post que reaparecio?** La FO usa
   `mu_mix` derivado del LSTM walking, los escenarios viven en el
   rollout autoregresivo. Son procesos distintos => `V[g, s]` muestra
   dispersion mucho mayor (peor escenario cae a `-82%`). Alternativas:
   generar tambien `mu_mix` como promedio de candidatos, o generar
   escenarios con walking en vez de rollout.

3. **¿Como evitar la perdida del backtest historico?** Aun corrigiendo
   timing, el modelo pierde. Posibles causas: el sesgo de magnitud de los
   escenarios CMC (`+193%` reps vs `-13%` real), la decision de elegir
   `g*` por regret promedio en lugar de minimax, o que el LSTM
   verdaderamente no tiene informacion utilizable con tan pocos datos.

El diagnostico completo (con tablas por hallazgo y todas las figuras)
esta en `diagnostico/diagnostico.md`.
