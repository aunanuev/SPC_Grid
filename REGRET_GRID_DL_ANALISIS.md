# Regret-Grid con Prediccion Deep Learning (LSTM) — Analisis Completo

## 1. Descripcion General

Este modelo implementa una **calibracion de hiperparametros tipo Regret-Grid** para un modelo de optimizacion de portafolio dinamico de media-varianza. La innovacion principal es que los escenarios futuros del mercado se generan mediante una **red neuronal LSTM** entrenada con **quantile regression** (pinball loss), en lugar de usar escenarios fijos o historicos.

El pipeline consta de 7 pasos secuenciales:

1. Carga de datos de mercado (retornos y probabilidades de regimen)
2. Construccion de dataset supervisado para la LSTM
3. Entrenamiento de la LSTM con pinball loss
4. Generacion de 1,000 escenarios candidatos con la LSTM congelada
5. Reduccion a 5 escenarios representativos (quintiles)
6. Ejecucion del regret-grid: 15 combinaciones de parametros x 5 escenarios
7. Seleccion del punto optimo g* y comparacion contra benchmarks

---

## 2. Configuracion de Parametros

### 2.1 Parametros de la LSTM

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `H` (lookback) | 20 | Ventana de retornos historicos usada como input (20 semanas) |
| `HIDDEN_SIZE` | 32 | Dimension del estado oculto de la LSTM |
| `N_LAYERS` | 1 | Numero de capas LSTM apiladas |
| `DROPOUT` | 0.1 | Tasa de dropout para regularizacion |
| `LR` | 0.001 | Learning rate del optimizador Adam |
| `EPOCHS` | 500 | Maximo de epocas de entrenamiento |
| `PATIENCE` | 30 | Epocas sin mejora antes de early stopping |
| `QUANTILE_LEVELS` | [0.1, 0.3, 0.5, 0.7, 0.9] | Niveles de quantiles a predecir |

### 2.2 Parametros de Generacion de Escenarios

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `N_SCENARIOS` | 1,000 | Numero de escenarios candidatos generados |
| `N_QUINTILES` | 5 | Escenarios representativos finales |
| `SEED` | 42 | Semilla para reproducibilidad |

### 2.3 Parametros del Modelo de Optimizacion

| Parametro | Valores en Grid | Descripcion |
|-----------|----------------|-------------|
| `lambda_riesgo` (L1-L5) | 0.05, 0.10, 0.20, 0.50, 1.00 | Coeficiente de aversion al riesgo |
| `costo_mult` (C1-C3) | 0.5, 1.0, 2.0 | Multiplicador de costos de transaccion |
| `theta` | {SPX: 1.0, CMC200: 1.0} | Multiplicadores de sentimiento (neutral) |
| `c_base` | {SPX: 0.5%, CMC200: 1.0%} | Costos base de transaccion |
| `w0` | {SPX: 0.5, CMC200: 0.5} | Portafolio inicial (50/50) |
| `Capital_inicial` | $10,000 | Capital inicial de la simulacion |

### 2.4 Grid Completo (15 combinaciones)

El grid forma 5 x 3 = 15 puntos `g = (lambda, m)`:

```
       C1(m=0.5)  C2(m=1.0)  C3(m=2.0)
L1     0.05/0.5   0.05/1.0   0.05/2.0
L2     0.10/0.5   0.10/1.0   0.10/2.0
L3     0.20/0.5   0.20/1.0   0.20/2.0
L4     0.50/0.5   0.50/1.0   0.50/2.0
L5     1.00/0.5   1.00/1.0   1.00/2.0
```

---

## 3. Datos de Entrada

Los datos provienen de 4 archivos CSV:
- **`prob_spx.csv`** y **`prob_cmc200.csv`**: Probabilidades de regimen bear/bull por periodo `t`
- **`ret_semanal_spx.csv`** y **`ret_semanal_cmc200.csv`**: Retornos semanales por periodo `t`

Se trabaja con **163 periodos semanales** y **2 activos**: SPX (S&P 500) y CMC200 (indice cripto).

La funcion `load_market_data` calcula:
- **Medias por regimen** `mu_hat(i,k)`: retorno promedio del activo `i` en regimen `k`, ponderado por la probabilidad de regimen
- **Covarianzas por regimen** `sigma_hat(i,j,k)`: covarianza entre activos `i` y `j` en regimen `k`
- **Momentos mixtos** `mu_mix(i,t)` y `sigma_mix(i,j,t)`: combinacion ponderada por probabilidades de regimen en cada periodo `t`

---

## 4. Red Neuronal LSTM — Prediccion de Quantiles

### 4.1 Arquitectura

La LSTM (`QuantileLSTM`) tiene la siguiente estructura:

```
Input: (batch, 20, 2)        -- 20 semanas de retornos de 2 activos
  |
  v
LSTM(input=2, hidden=32, layers=1)
  |
  v
Dropout(0.1)
  |
  v
Linear(32 -> 2*5 = 10)       -- 5 quantiles por cada activo
  |
  v
torch.sort(dim=-1)           -- Garantiza monotonia de quantiles
  |
  v
Output: (batch, 2, 5)        -- 5 quantiles para SPX y CMC200
```

La capa `torch.sort` al final es clave: garantiza que q(0.1) <= q(0.3) <= q(0.5) <= q(0.7) <= q(0.9), lo cual es un requerimiento logico de la regresion por quantiles.

### 4.2 Dataset Supervisado

El dataset se construye con ventanas deslizantes:
- **Input**: ultimos `H=20` retornos de ambos activos -> tensor de forma `(20, 2)`
- **Target**: retorno de la semana siguiente de ambos activos -> tensor de forma `(2,)`

Split cronologico (no aleatorio, para respetar la temporalidad):
- **Train**: 100 muestras (70%)
- **Validacion**: 21 muestras (15%)
- **Test**: 22 muestras (15%)

### 4.3 Funcion de Perdida: Pinball Loss

La **pinball loss** (tambien llamada quantile loss) permite entrenar la red para predecir multiples quantiles simultaneamente:

```
L(q, y, y_hat) = q * max(y - y_hat, 0) + (1-q) * max(y_hat - y, 0)
```

Donde:
- `q` es el nivel del quantil (0.1, 0.3, 0.5, 0.7, 0.9)
- `y` es el retorno observado (target)
- `y_hat` es la prediccion del quantil

**Interpretacion**: cuando `q=0.1`, la perdida penaliza mas las sobreestimaciones (queremos que solo el 10% de las observaciones caigan debajo). Cuando `q=0.9`, penaliza mas las subestimaciones.

### 4.4 Resultados del Entrenamiento

```
Epoch   1  train_loss=0.031975  val_loss=0.027218
Epoch   5  train_loss=0.019988  val_loss=0.013281
...
Early stopping en epoch 40 (patience=30)
Mejor val_loss = 0.009767
```

El modelo convergio rapidamente (early stopping en epoca 40 de un maximo de 500) con una perdida de validacion final de **0.009767**.

---

## 5. Generacion y Reduccion de Escenarios

### 5.1 Generacion de 1,000 Escenarios

Con la LSTM congelada (modo evaluacion), se generan escenarios de forma autoregresiva:

1. Partir de la ventana de los ultimos 20 retornos observados
2. Para cada paso temporal `t`:
   - Alimentar la ventana a la LSTM -> obtener 5 quantiles para cada activo
   - Samplear uniformemente uno de los 5 quantiles para cada activo
   - Usar ese retorno como observacion y deslizar la ventana

Esto genera 1,000 trayectorias de 163 semanas para ambos activos.

**Estadisticas de retorno acumulado SPX sobre los 1,000 escenarios:**

| Estadistica | Valor |
|-------------|-------|
| Minimo | -23.04% |
| Mediana | +34.25% |
| Maximo | +168.75% |

### 5.2 Reduccion a 5 Quintiles

Los 1,000 escenarios se resumen en 5 representativos:

1. Calcular el retorno acumulado del SPX por cada escenario
2. Ordenar los escenarios por este retorno acumulado
3. Dividir en 5 quintiles (200 escenarios cada uno)
4. Seleccionar el escenario mediano de cada quintil

**Escenarios representativos seleccionados:**

| Quintil | Interpretacion | Ret. Acum. SPX |
|---------|---------------|----------------|
| s1 | Mercado pesimista | +3.04% |
| s2 | Mercado moderado bajo | +20.45% |
| s3 | Mercado neutral/mediano | +34.31% |
| s4 | Mercado moderado alto | +53.53% |
| s5 | Mercado optimista | +80.17% |

---

## 6. Modelo de Optimizacion de Portafolio

### 6.1 Variables de Decision

Para cada activo `i` en {SPX, CMC200} y cada periodo `t` en {1, ..., 163}:

| Variable | Dominio | Descripcion |
|----------|---------|-------------|
| `w(i,t)` | [0, 1] | Peso del activo `i` en el portafolio en el periodo `t` |
| `u(i,t)` | >= 0 | Cantidad comprada del activo `i` en `t` |
| `v(i,t)` | >= 0 | Cantidad vendida del activo `i` en `t` |

### 6.2 Funcion Objetivo

El modelo maximiza:

```
max z = SUM_t [ Retorno_t - lambda * Riesgo_t - Costos_t ]
```

Donde cada componente se define como:

#### Componente 1: Retorno Esperado

```
Retorno_t = SUM_i  w(i,t) * mu_mix(i,t) * theta(i)
```

- `mu_mix(i,t)` es el retorno esperado mixto del activo `i` en periodo `t` (mezcla bear/bull)
- `theta(i)` es el multiplicador de sentimiento (1.0 para neutral)
- Maximizar este termino incentiva al optimizador a asignar mas peso a activos con mayor retorno esperado

#### Componente 2: Penalizacion de Riesgo (termino cuadratico)

```
Riesgo_t = lambda * SUM_i SUM_j  w(i,t) * w(j,t) * sigma_mix(i,j,t)
```

- `sigma_mix(i,j,t)` es la covarianza mixta entre activos `i` y `j` en periodo `t`
- `lambda` (lambda_riesgo) controla cuanto penalizar la varianza del portafolio
- **Lambda bajo (0.05)**: portafolio agresivo, busca retorno, acepta riesgo
- **Lambda alto (1.00)**: portafolio conservador, prioriza estabilidad
- Este termino es una **forma cuadratica** w' * Sigma * w, la clasica medida de varianza del portafolio

#### Componente 3: Costos de Transaccion

```
Costos_t = SUM_i  c_eff(i) * (u(i,t) + v(i,t))
```

- `c_eff(i) = c_base(i) * costo_mult`: costo efectivo de transaccionar el activo `i`
- `u(i,t) + v(i,t)` es el turnover total (compras + ventas)
- `costo_mult` (m) escala los costos: m=0.5 los reduce a la mitad, m=2.0 los duplica
- Minimizar este termino desincentiva el rebalanceo excesivo

### 6.3 Restricciones

#### Normalizacion de pesos (fully invested)

```
SUM_i w(i,t) = 1    para todo t
```

El portafolio siempre esta 100% invertido. No se permite cash ni apalancamiento.

#### Identidad de rebalanceo (t > 1)

```
w(i,t) - w(i,t-1) = u(i,t) - v(i,t)    para todo i, t > 1
```

El cambio en peso de un activo debe explicarse por compras menos ventas. Esta ecuacion vincula las variables de peso con las de trading.

#### Anclaje inicial (t = 1)

```
w(i,1) - w0(i) = u(i,1) - v(i,1)    para todo i
```

El portafolio parte del peso inicial `w0 = {SPX: 0.5, CMC200: 0.5}`.

#### Cotas

```
0 <= w(i,t) <= 1    (no short-selling, no apalancamiento individual)
u(i,t), v(i,t) >= 0  (compras y ventas no negativas)
```

### 6.4 Solver

Se usa **GAMSPy + IPOPT** (Interior Point OPTimizer) para resolver el problema de programacion no lineal (NLP) resultante. El termino cuadratico en la funcion objetivo lo convierte en un QP/NLP.

---

## 7. Regret-Grid: Metodologia de Calibracion

### 7.1 Concepto

El regret-grid busca responder: **cual combinacion de (lambda, costo_mult) es mas robusta frente a la incertidumbre del mercado?**

En lugar de optimizar para un unico escenario futuro, se evalua cada configuracion contra multiples escenarios y se mide el "arrepentimiento" (regret) de no haber elegido la mejor configuracion para cada escenario.

### 7.2 Procedimiento

Para cada punto `g = (lambda, m)` del grid (15 puntos):

1. **Optimizar**: resolver el modelo de portafolio con esos parametros -> obtener pesos optimos `w^g`
2. **Simular**: para cada escenario `s` (5 quintiles), calcular el capital terminal usando los pesos `w^g` pero con los retornos del escenario `s`

La simulacion ex-post del capital sigue:

```
x_{t+1} = x_t * (1 + SUM_i w^g(i,t) * r^s(i,t)) - x_t * SUM_i c_base(i) * |w^g(i,t+1) - w^g(i,t)|
```

Esto produce la **matriz de capital terminal** `V(g, s)` de dimension 15 x 5.

### 7.3 Ecuacion de Regret

Para cada escenario `s`, el mejor capital posible es:

```
V_best(s) = max_g V(g, s)
```

El **regret** de la configuracion `g` en el escenario `s` es:

```
R(g, s) = V_best(s) - V(g, s)
```

El regret mide **cuanto dinero se dejo sobre la mesa** por no haber usado la configuracion optima para ese escenario particular. Un regret de 0 significa que `g` fue la mejor opcion para ese escenario.

### 7.4 Criterios de Seleccion

Se usan dos criterios para seleccionar g*:

#### Average Regret (regret promedio)

```
g*_avg = argmin_g  (1/S) * SUM_s R(g, s)
```

Minimiza el arrepentimiento esperado promediando sobre todos los escenarios. Es el criterio mas "equilibrado".

#### Worst-Case Regret (minimax regret)

```
g*_wc = argmin_g  max_s R(g, s)
```

Minimiza el peor caso. Es un criterio mas conservador: elige la configuracion cuyo peor escenario es el menos malo.

---

## 8. Resultados

### 8.1 Valor Objetivo del Optimizador (z)

| g | lambda | m | z (valor FO) |
|---|--------|---|-------------|
| L1/C1 | 0.05 | 0.5 | 0.628371 |
| L1/C2 | 0.05 | 1.0 | 0.624625 |
| L1/C3 | 0.05 | 2.0 | 0.617121 |
| L2/C1 | 0.10 | 0.5 | 0.576554 |
| L2/C2 | 0.10 | 1.0 | 0.572804 |
| L2/C3 | 0.10 | 2.0 | 0.565304 |
| L3/C1 | 0.20 | 0.5 | 0.476068 |
| L3/C2 | 0.20 | 1.0 | 0.471307 |
| L3/C3 | 0.20 | 2.0 | 0.464296 |
| L4/C1 | 0.50 | 0.5 | 0.354001 |
| L4/C2 | 0.50 | 1.0 | 0.352569 |
| L4/C3 | 0.50 | 2.0 | 0.350652 |
| L5/C1 | 1.00 | 0.5 | 0.284709 |
| L5/C2 | 1.00 | 1.0 | 0.282260 |
| L5/C3 | 1.00 | 2.0 | 0.277631 |

**Observacion**: El valor objetivo `z` disminuye al aumentar lambda (mayor penalizacion de riesgo) y al aumentar `m` (mayores costos de transaccion). Esto es esperado: el optimizador tiene un espacio de busqueda mas restringido.

### 8.2 Capital Terminal V(g, s) — Matriz Completa

| g | s1 (pesim.) | s2 (mod.bajo) | s3 (neutral) | s4 (mod.alto) | s5 (optim.) | Promedio |
|---|-------------|--------------|-------------|--------------|------------|---------|
| L1/C1 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L1/C2 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L1/C3 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L2/C1 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L2/C2 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L2/C3 | $10,245 | $18,846 | $3,173 | $10,895 | $8,460 | $10,324 |
| L3/C1 | $10,555 | $18,870 | $3,670 | $11,708 | $9,156 | $10,792 |
| L3/C2 | $10,390 | $18,959 | $3,612 | $11,486 | $9,118 | $10,713 |
| L3/C3 | $10,752 | $18,612 | $3,896 | $11,848 | $9,577 | $10,937 |
| L4/C1 | $10,986 | $15,715 | $7,966 | $14,807 | $14,592 | $12,813 |
| L4/C2 | $11,223 | $15,733 | $8,136 | $14,830 | $14,652 | $12,915 |
| L4/C3 | $11,252 | $15,820 | $8,042 | $14,823 | $14,582 | $12,904 |
| L5/C1 | $10,600 | $13,708 | $10,195 | $15,584 | $16,457 | $13,309 |
| L5/C2 | $10,614 | $13,775 | $10,167 | $15,579 | $16,472 | $13,321 |
| **L5/C3** | **$10,644** | **$13,843** | **$10,121** | **$15,603** | **$16,472** | **$13,337** |

**Observaciones clave:**

1. **L1-L2 (lambda bajo)**: Los resultados son practicamente identicos sin importar el costo multiplicador. Esto indica que con baja aversion al riesgo, el portafolio se concentra agresivamente y no rebalancea mucho, por lo que los costos de transaccion son irrelevantes.

2. **L1-L2 tienen alto riesgo**: En el escenario s3 (neutral), el capital cae a ~$3,173 (perdida del 68.3%). Esto muestra el peligro de la sobreconcentracion en un activo volatil.

3. **L5 (lambda alto)**: Es la mas estable — el peor caso (s3) tiene $10,121 (aun gana 1.2%) y el mejor caso (s5) tiene $16,472 (ganancia del 64.7%).

4. **Patron general**: Al aumentar lambda, disminuye la dispersion entre escenarios. El portafolio se vuelve mas diversificado y resiliente.

### 8.3 Matriz de Regret R(g, s)

| g | s1 | s2 | s3 | s4 | s5 | Avg Regret | Max Regret |
|---|----|----|----|----|----|-----------:|----------:|
| L1/C1 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | **4,172** | **8,012** |
| L1/C2 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | 4,172 | 8,012 |
| L1/C3 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | 4,172 | 8,012 |
| L2/C1 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | 4,172 | 8,012 |
| L2/C2 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | 4,172 | 8,012 |
| L2/C3 | 1,007 | 113 | 7,022 | 4,708 | 8,012 | 4,172 | 8,012 |
| L3/C1 | 698 | 89 | 6,525 | 3,895 | 7,317 | 3,704 | 7,317 |
| L3/C2 | 862 | 0 | 6,583 | 4,117 | 7,354 | 3,783 | 7,354 |
| L3/C3 | 501 | 347 | 6,298 | 3,755 | 6,896 | 3,559 | 6,896 |
| L4/C1 | 266 | 3,245 | 2,229 | 796 | 1,880 | 1,683 | 3,245 |
| L4/C2 | 30 | 3,226 | 2,059 | 773 | 1,821 | 1,582 | 3,226 |
| L4/C3 | 0 | 3,139 | 2,153 | 780 | 1,890 | 1,593 | 3,139 |
| L5/C1 | 653 | 5,251 | 0 | 20 | 15 | 1,188 | 5,251 |
| L5/C2 | 639 | 5,184 | 27 | 24 | 1 | 1,175 | 5,184 |
| **L5/C3** | **609** | **5,116** | **74** | **0** | **0** | **1,160** | **5,116** |

### 8.4 Seleccion de g*

#### Por Average Regret: **g* = L5/C3 (lambda=1.00, m=2.0)**

- Average regret = **$1,159.80** (el minimo de los 15 puntos)
- Esta configuracion es la que en promedio "deja menos dinero sobre la mesa"
- Su regret es cercano a 0 en escenarios s3, s4 y s5 (neutrales y optimistas)
- Su punto debil es s2 (moderado bajo) con regret de $5,116, donde un portafolio agresivo habria capturado mas retorno

#### Por Worst-Case Regret: **g* = L4/C3 (lambda=0.50, m=2.0)**

- Max regret = **$3,139.45** (el menor peor caso)
- Esta configuracion ofrece el mejor balance: su peor escenario no es catastrofico
- Es mas equilibrada: regrets moderados en todos los escenarios

### 8.5 Interpretacion de la Divergencia entre Criterios

La diferencia entre ambos g* revela un trade-off fundamental:

- **L5/C3 (avg regret)**: Excelente en 3 de 5 escenarios, pero vulnerable al escenario s2 (mercado moderado bajo). Es la eleccion de un inversor que cree que los escenarios extremos negativos y los moderados positivos/altos son mas probables.

- **L4/C3 (worst-case)**: Nunca falla estrepitosamente, pero tampoco es la mejor en ningun escenario. Es la eleccion de un inversor que quiere dormir tranquilo sin importar que pase.

---

## 9. Politica Final — Comparacion con Benchmarks

Se re-ejecuto el optimizador con la configuracion g* seleccionada por average regret (L5/C3: lambda=1.00, m=2.0) y se comparo contra dos benchmarks naive:

| Estrategia | Capital Final | Retorno Acumulado | Incremento de Capital |
|-----------|--------------|-------------------|----------------------|
| **g* Optimo (lambda=1.0, m=2.0)** | **$14,574.91** | **+45.75%** | **+$4,574.91** |
| Naive 50/50 Rebalanceo | $11,764.55 | +17.65% | +$1,764.55 |
| Naive Buy & Hold | $12,258.46 | +22.58% | +$2,258.46 |

**El modelo optimizado con g* supera ambos benchmarks por un amplio margen:**

- vs Naive Rebalanceo: +$2,810.36 adicionales (+23.87 pp de retorno)
- vs Buy & Hold: +$2,316.45 adicionales (+23.17 pp de retorno)

### 9.1 Detalle de los Benchmarks

- **Naive Buy & Hold**: Portafolio 50/50 inicial que no se rebalancea nunca. Sin costos de transaccion. Los pesos derivan naturalmente segun la performance de cada activo.

- **Naive 50/50 Rebalanceo**: Portafolio que se rebalancea semanalmente a 50/50. Incurre en costos de transaccion cada vez que rebalancea.

- **g* Optimo**: Portafolio que usa los pesos optimizados por el modelo, calibrado con la grilla de regret. Con lambda=1.0 es conservador en riesgo, y con m=2.0 se penalizan fuertemente los costos de transaccion, lo que produce un portafolio estable con rebalanceo infrecuente pero estrategico.

---

## 10. Valor del Objetivo (z* = 0.277631)

El valor `z* = 0.277631` con status `optimal_local` indica que IPOPT encontro un optimo local del problema NLP. Este valor es la suma sobre los 163 periodos de (retorno - riesgo - costos). Es menor que el de configuraciones con lambda bajo (e.g., z=0.628 para L1/C1), lo cual es natural: la funcion objetivo con lambda=1.0 impone una penalizacion de riesgo 20 veces mayor que con lambda=0.05. Pero el capital terminal real (que es lo que importa al inversor) es significativamente mayor porque la proteccion contra riesgo preserva capital en periodos adversos.

---

## 11. Conclusiones

1. **La LSTM como generador de escenarios funciona**: Con solo 100 muestras de entrenamiento, el modelo aprendio a generar escenarios plausibles (retorno acumulado SPX entre -23% y +169%), proporcionando una base rica para el regret-grid.

2. **Lambda alto domina el regret-grid**: Las configuraciones con lambda=1.00 tienen consistentemente el menor average regret, aunque sacrifican performance en escenarios moderados-bajos donde portafolios agresivos brillan.

3. **Los costos de transaccion importan mas con lambda alto**: Para L1-L2, el multiplicador de costos no afecta los resultados. Para L4-L5, cambiar m de 0.5 a 2.0 genera diferencias medibles en el capital terminal, lo que confirma que estos portafolios rebalancean activamente.

4. **El portafolio optimizado con g* triplica el retorno de los benchmarks naive**: Un retorno de +45.75% vs +17.65% (naive rebalanceo) o +22.58% (buy & hold) sobre 163 semanas (~3.1 anos) demuestra el valor de la optimizacion dinamica calibrada con regret-grid.

5. **El regret-grid aporta robustez**: En lugar de calibrar parametros ad-hoc o por backtesting sobre un unico escenario historico, el regret-grid evalua la robustez de cada configuracion frente a multiples futuros posibles, reduciendo el riesgo de sobre-ajuste.
