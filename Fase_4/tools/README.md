# Fase 4 — tools

Tres cosas: post-proceso de CSV, el **gate del ancla FP64** (`gate3_ancla.py`, nuevo) y el análisis final (estadística + Pareto).

Los dos extractores son ahora **idénticos** a los de `Fase_3/tools/` (antes `extract_csv.py` divergía). Los seis binarios emiten el mismo esquema de columnas, y mantener dos variantes solo invitaba a que una se quedara atrás.

| Script | Para |
|---|---|
| `extract_csv.py` | `stencil_tensor_activation.cu` (Fase 3 y Fase 4) |
| `extract_csv_chained.py` | `gemm_chained.cu` y `conv_chained.cu` (Fase 3 y Fase 4) |
| `gate3_ancla.py` | Validación automatizada del ancla, **los tres kernels** |

```bash
python3 extract_csv.py --input run_123.log --outdir results --job-id 123 --kernel stencil
python3 extract_csv_chained.py --input run_456.log --outdir results --job-id 456 --kernel gemm
python3 extract_csv_chained.py --input run_789.log --outdir results --job-id 789 --kernel conv
```

## `gate3_ancla.py` — las dos puertas del ancla, automatizadas

Es lo que `Fase_4/Stencil/README.md` nombraba como `gate3_ancla.sbatch` y lo que `Fase_4/GEMM/README.md` y `Fase_4/Convolution/README.md` pedían con idéntica redacción ("mientras tanto, correr las dos puertas de la sección Validación arriba a mano"). Sigue el estilo de `old/Fase_4/Stencil/tools/comparar_gate1.py`: separar columnas deterministas de columnas de medición, veredicto solo sobre las primeras, código de salida `0`/`1` (más `2` = no evaluable, que **no** es un "pasa").

```bash
python3 gate3_ancla.py --kernel gemm \
    --gate0-base fase3_k0.log --gate0-nuevo fase4_k0.log --gate1 fase4_k1.log
```

En la práctica no se invoca a mano: cada `Fase_4/<kernel>/gate3_ancla.sbatch` compila los dos binarios, corre las tres pasadas y lo llama.

**Gate K=0** — `--anchor-every 0` debe reproducir Fase 3 columna por columna en lo determinista. Las columnas de tiempo/energía se reportan como desviación relativa pero **no deciden**: dos corridas del mismo binario ya difieren ahí por ruido. (Comprobado: en la primera corrida real las columnas deterministas salieron idénticas mientras `t_iter_ms` variaba un 59 % entre las dos pasadas.)

**Gate K=1 — leer antes de tocar las tolerancias.** La formulación ingenua ("con K=1, `rel_l2` debe caer a nivel de ruido de punto flotante, ~`1e-16`") **no puede pasar** en GEMM ni en Convolución, y no por un bug:

- Ahí `CSV_DRIFT` compara la referencia FP64 contra el buffer `T` **tal cual se guarda** (FP16/BF16), nunca contra `T+comp`. Con K=1 la reconstrucción interna es *exacta* — `comp64 = out64 − dequant(q)` es una resta exacta por Sterbenz, y por tanto `dequant(q) + comp64 == out64` bit a bit —, así que lo único que separa a `T` de la referencia es la **cuantización al formato de 16 bits**. De ahí sale una cota derivada, no un umbral inventado: para redondeo al más cercano con `p` bits de significando, `rel_l2 ≤ 2^-p` y `rel_linf ≤ 2^-p`, con `p=11` en FP16 (`4.883e-04`) y `p=8` en BF16 (`3.906e-03`).
- **Verificado en GPU real** (`sm_86`, `conv_chained --hw 64 --iters 12 --tc both --comp on --anchor-every 1`): FP16 llegó a `rel_linf = 3.74e-04` y BF16 a `3.00e-03` — ambos al **0.77** de su cota, el *mismo* factor en los dos formatos, que es la confirmación empírica de que el modelo es el correcto.
- **En Stencil el test no puede ser el mismo**, porque `CSV_DRIFT` mide otro objeto: compara contra `d_out_fp32`, el acumulador FP32 *sin* el redondeo de almacenamiento ("ancla de no-regresión"). El análogo real de `rel_l2` de GEMM/Conv es `rel_l2_prop`, que sí mide el estado propagado en 16 bits. Por eso el gate de Stencil tiene **dos criterios**: (a) `rel_l2`/`rel_linf` de la ruta anclada deben alcanzar el nivel de `GPU_FP64` de la misma corrida (con K=1 el paso se sustituye entero por `stencil2d_fp64_kernel`, así que la trayectoria anclada *es* la de `GPU_FP64`), y (b) `rel_l2_prop`/`rel_linf_prop` deben cumplir la misma cota `2^-p`.

Aplicar el criterio de un kernel al otro es el error fácil aquí, y es la razón por la que el script no tiene un solo umbral global.

**Un solo script para los tres kernels, no tres.** La lógica de las dos puertas es literalmente la misma; lo único que cambia es el esquema de columnas, aislado en una tabla de datos (`ESQUEMAS`). Tres copias habrían divergido a la primera corrección. Vive en `Fase_4/tools/` porque es donde ya está el resto de la herramienta transversal a kernels.

**Probado en GPU real, positiva y negativamente** (`sm_86`, 2026-09-06): pasa con logs reales de GEMM (`n=256`) y Convolución (`hw=64`); falla con código `1` si se corrompe un `rel_l2` determinista del log K=0, falla con `1` si se infla un `rel_linf` por encima de la cota, y devuelve `2` (no evaluable) si se le pasa un log K=0 como si fuera K=1 o un log con varias corridas concatenadas. Stencil no se pudo ejercitar localmente: su `.cu` tiene un `static_assert(sizeof(long) >= 8)` que rechaza Windows a propósito.

## `anchor_every`: ahora es COLUMNA REAL, ya no reconstruida

Los seis binarios la escriben como **última** columna de `CSV_DRIFT` y `CSV_SUMMARY` (y de `CSV_ENERGY` en Stencil). Al final, no junto a `kahan`, para no correr los índices posicionales que `SUMMARY_VALUE_FIELDS`/`SUMMARY_FIELD_COUNT` ya usan.

En Stencil es contexto de **invocación completa** (una constante por proceso), no de ruta: si una corrida mezcla rutas de referencia (`gpu_fp64`, `cpu_fp64`) con la ruta WMMA bajo ancla, todas comparten el mismo `anchor_every` — correcto, porque el ancla es un parámetro de la corrida. En GEMM/Convolución, en cambio, **varía por ruta** dentro de la misma corrida: `_none` siempre reporta `0`, `_comp` el valor real. La implementación codifica esa diferencia (variable de proceso en Stencil, valor en el punto de uso en GEMM/Conv) justamente para que no se pierda en una refactorización.

Los extractores conservan el **respaldo** de reconstruirla desde la línea de configuración del binario, para logs anteriores al cambio. `Fase_3/tools/extract_csv.py` dejó de estar "sin tocar": ahora es el mismo archivo que este.

## Análisis: `common_analysis.py`, `run_statistics.py`, `pareto_front.py`

Etapas 7 y 9 del plan — estadística inferencial y Frente de Pareto 3D, sobre los CSV que producen los scripts de arriba.

| Script | Qué hace | Etapa del plan |
|---|---|---|
| `common_analysis.py` | Normaliza los esquemas de Stencil y de GEMM/Convolución a una sola tabla larga común (`kernel, format, treatment, anchor_every, size, iters, rel_l2, t_iter_ms, energy_gpu_j_per_iter, ...`) — lo usan los otros dos, no se invoca directo salvo para su propio smoke test (`python3 common_analysis.py`). | (soporte de 7 y 9) |
| `run_statistics.py` | ANOVA factorial (formato × tratamiento × horizonte) + post-hoc Tukey HSD sobre Stencil (contraste `none`/`kahan_local`/`spatial`); y efecto del ancla FP64 (`K` como variable ordinal vía regresión en `log(K+1)`, **además de** un ANOVA categórico sobre los niveles de `K`), en los tres kernels con `kernel` como factor adicional. | Etapa 7 |
| `pareto_front.py` | Dominancia de Pareto en 3D (tiempo, energía, error) **un frente por kernel** (nunca uno combinado), visualización 3D, y tabla de directrices (para cada tolerancia de error, qué formato/tratamiento/`K` minimiza el EDP). | Etapa 9 |

```bash
# Autodetecta CSV por prefijo (drift_/summary_/energy_) en un directorio:
python3 run_statistics.py --results-dir results/ --outdir stats_out/
python3 pareto_front.py   --results-dir results/ --outdir pareto_out/

# O apuntando a archivos/patrones concretos (glob) por kernel:
python3 run_statistics.py \
    --stencil-summary "results/summary_stencil_*.csv" --stencil-energy "results/energy_stencil_*.csv" \
    --gemm-summary "results/summary_gemm_*.csv" --gemm-drift "results/drift_gemm_*.csv" \
    --conv-summary "results/summary_conv_*.csv" --conv-drift "results/drift_conv_*.csv" \
    --outdir stats_out/
```

**Verificado, no solo escrito**: sin datos reales de PACCA (nada corrió en GPU en este entorno de desarrollo), los tres scripts SÍ se ejecutaron de verdad contra datos sintéticos con el esquema exacto de columnas que documentan `Fase_3/tools/README.md`/este archivo — `python3 common_analysis.py`, `python3 run_statistics.py --self-test`, `python3 pareto_front.py --self-test` corren el pipeline completo (normalización → ANOVA/Tukey/regresión → Pareto → guía) sin excepciones y con salidas no vacías. Dos bugs reales aparecieron y se corrigieron durante esas pruebas (una colisión de columna al deduplicar claves en `Dataset.cells()`, y un crash de `statsmodels`/`scipy.linalg.qr` cuando el diseño queda sin grados de libertad residuales con muy pocas filas — ahora se detecta antes y se avisa por `stderr` en vez de propagar la excepción). Lo que **no** está verificado es que las conclusiones tengan sentido físico — eso exige una campaña real.

**`run_full_pipeline.sh`** (raíz del repo) encadena las cuatro fases y termina invocando estos dos scripts sobre todo lo que se haya generado — ver su propio comentario de cabecera.

## Qué falta

- Los mismos gates de validación que `Fase_3/tools/README.md` — comparar_gate1.py/validar_gate2.py siguen sin migrar (solo existen para Stencil en `old/Fase_4/Stencil/`), y no existe todavía un equivalente para el ancla de GEMM/Conv.
- `anchor_every` como columna real del CSV en los tres binarios seguiría siendo más robusto que reconstruirla por línea de configuración — pendiente si alguna vez se retoca el formato de salida de los tres `.cu`.
- **Aleatorización del orden de envío de réplicas** (exigida por la Etapa 7 si el aislamiento térmico de la Etapa 5 no se concede) — no es responsabilidad de `run_statistics.py` (que solo analiza lo que ya se corrió), sino de cómo se lanzan los jobs; no hay todavía un script que aleatorice el orden de envío de una campaña de réplicas.
- Validación de estos scripts contra una campaña real en PACCA — lo único que se ha verificado es que corren correctamente contra datos sintéticos con el esquema de columnas documentado.
