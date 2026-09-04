# Fase 4 — tools

**No son copias idénticas de `Fase_3/tools/`** — a diferencia de `extract_csv_chained.py` (idéntico en ambas fases, GEMM/Conv de Fase 3 no tiene ancla que reconstruir), `extract_csv.py` de esta carpeta SÍ tiene una extensión propia sobre la versión de `Fase_3/tools/`: reconstruye la columna `anchor_every`. Alimentan a los binarios de Fase 4:

| Script | Para |
|---|---|
| `extract_csv.py` | `Fase_4/Stencil/stencil_tensor_activation.cu` — **extendido** respecto a `Fase_3/tools/extract_csv.py` (ver abajo) |
| `extract_csv_chained.py` | `Fase_4/GEMM/gemm_chained.cu`, `Fase_4/Convolution/conv_chained.cu` — idéntico al de `Fase_3/tools/` |

```bash
python3 extract_csv.py --input run_123.log --outdir results --job-id 123 --kernel stencil
python3 extract_csv_chained.py --input run_456.log --outdir results --job-id 456 --kernel gemm
python3 extract_csv_chained.py --input run_789.log --outdir results --job-id 789 --kernel conv
```

## `anchor_every`: ahora reconstruido en LOS TRES kernels

Ninguno de los tres binarios escribe `anchor_every` como columna real de `CSV_DRIFT`/`CSV_SUMMARY` — los tres extractores lo reconstruyen leyendo, en cambio, la línea de configuración que cada binario imprime **una vez por invocación**, antes de sus filas `CSV_*`:

- GEMM/Convolución (`extract_csv_chained.py`): `N=1024 ... anchor_every=5 (activa)` / `HW=64 ... anchor_every=5 (activa)`.
- Stencil (`extract_csv.py`, extensión de esta carpeta sobre `Fase_3/tools/extract_csv.py`): `Ancla FP64 (anchor-every)  : 5 (activa)` — nuevo regex `ANCHOR_RE`, nueva columna al final de `DRIFT_HEADER`/`SUMMARY_HEADER`/`ENERGY_HEADER` (al final, no junto a `kahan`, para no correr los índices posicionales que `SUMMARY_VALUE_FIELDS` ya usa sobre la línea `CSV_SUMMARY`).

En Stencil, `anchor_every` es contexto de **invocación completa** (una constante por proceso), no de ruta: si una corrida mezcla rutas de referencia (`gpu_fp64`, `cpu_fp64`) con la ruta WMMA bajo ancla, todas las filas de esa invocación comparten el mismo `anchor_every` — correcto, porque el ancla es un parámetro de la corrida, no de la ruta individual (a diferencia de GEMM/Conv, donde `anchor_every` sí varía por ruta dentro de la misma corrida: `_none` siempre reporta `0`, `_comp` reporta el valor real — ver el comentario de `extract_csv_chained.py`).

`Fase_3/tools/extract_csv.py` (sin ancla, Fase 3 nunca imprime esa línea) se dejó **sin tocar** — no tiene sentido buscar una línea que su binario no emite.

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
