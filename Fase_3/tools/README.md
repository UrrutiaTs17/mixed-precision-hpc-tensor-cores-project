# Fase 3 — tools

Dos cosas distintas viven aquí: el **post-proceso de CSV** (dos scripts) y la **verificación del orden de operandos de `cublasDgemm`** (dos scripts más, nuevos).

## Verificación del orden de operandos — `verificar_orden_operandos_{gemm,conv}.py`

Es el chequeo que los comentarios de cabecera de `gemm_chained.cu` y `conv_chained.cu` piden con todas las letras ("el punto de mayor riesgo de error silencioso… verificar con un tamaño chico contra una referencia independiente **antes de confiar en cualquier resultado**") y que no existía en ningún lado del proyecto.

| Script | Qué verifica |
|---|---|
| `verificar_orden_operandos_gemm.py` | Que `gpu_fp64_step()` calcule `X·A` y no `A·X`, `Xᵀ·A` ni `(X·A)ᵀ`. |
| `verificar_orden_operandos_conv.py` | Lo mismo en `gpu_fp64_conv_step()`, **más** la indexación del `im2col` (transposición `r↔s`, filtro volteado) y el padding "SAME" (ceros fuera del dominio), **más** que `build_block_diagonal_filter()` produzca de verdad 64 canales independientes con el Laplaciano exacto. |

```bash
python3 verificar_orden_operandos_gemm.py --n 32
python3 verificar_orden_operandos_conv.py --hw 64
```

Códigos de salida: `0` correcto, `1` **falla**, `2` no se pudo verificar (sin `nvcc`/GPU). El `2` se distingue del `1` a propósito: "no se sabe" no es "pasa".

**No reimplementan el cálculo, lo extraen.** Cada script lee del `.cu` el texto de las funciones bajo prueba y lo pega, sin tocar un carácter, dentro de un binario mínimo que expone el resultado crudo. Reimplementar la llamada a `cublasDgemm` aquí y compararla con NumPy solo probaría que el script está bien. Además exigen que Fase 3 y Fase 4 tengan la **misma** función: los README de Fase 4 afirman que la reutilizan "tal cual", y si eso dejara de ser cierto, verificar una sola daría una falsa sensación de seguridad.

**Por qué un binario propio y no `./gemm_chained --n 16`**: (a) `gemm_chained` no publica en ningún lado el resultado crudo de `gpu_fp64_step` — solo `rel_l2` de la ruta WMMA *contra* esa referencia, lo que no distingue "la referencia está mal" de "la ruta WMMA está mal"; (b) `parse_args()` **rechaza** `--n 16` y `--n 32` (exige múltiplo de `kBlockTileM=64`, requisito de `wmma_gemm_kernel`, no de cuBLAS).

**Por qué dos scripts y no uno con `--kernel`**: lo único compartido es la plomería de detectar `nvcc`, compilar y correr (~60 líneas mecánicas). Lo demás es disjunto — allá se extrae una función, acá cinco símbolos y cuatro constantes; allá la referencia NumPy es `X @ A`, acá una correlación 2D por canal con padding explícito más una segunda referencia `im2col` independiente como control cruzado. Un script único sería un `if kernel == "gemm"` en cada paso, y cada gate tiene que poder correrse y auditarse solo dentro de una sesión de PACCA.

**Resultado de la primera corrida** (GPU Ampere real, `sm_86`, 2026-09-06): los dos pasan. GEMM da `rel_linf = 0.0` contra `X @ A`, con las hipótesis alternativas a distancia `1.28` (`A@X`), `1.12` (`Xᵀ·A`) y `1.23` (`(X·A)ᵀ`). Convolución da `2.2e-16` contra la correlación con padding de ceros, con `1.04` para el filtro volteado y `0.48` para el transpuesto. El orden de operandos de los dos kernels es correcto.

## Post-proceso de CSV

Dos scripts, dos esquemas — **no comparten código** porque los binarios que los alimentan emiten columnas distintas:

| Script | Para | Esquema de entrada |
|---|---|---|
| `extract_csv.py` | `Fase_3/Stencil/stencil_tensor_activation.cu` | `CSV_DRIFT`/`CSV_SUMMARY`/`CSV_ONSET`/`CSV_HORIZON`/`CSV_STORE`/`CSV_ENERGY`, columnas `nx`/`ny`/`kahan`/`route` |
| `extract_csv_chained.py` | `gemm_chained.cu` (`Fase_3/GEMM`) y `conv_chained.cu` (`Fase_3/Convolution`) | Solo `CSV_DRIFT`/`CSV_SUMMARY`, columnas `n`/`hw` (unificadas aquí como `size`) |

Ambos se invocan igual, con `--kernel` seleccionando el esquema y el nombre de archivo de salida:

```bash
python3 extract_csv.py --input run_123.log --outdir results --job-id 123 --kernel stencil
python3 extract_csv_chained.py --input run_456.log --outdir results --job-id 456 --kernel gemm
python3 extract_csv_chained.py --input run_789.log --outdir results --job-id 789 --kernel conv
```

Los `.sbatch` de cada carpeta (`Fase_3/Stencil/`, `Fase_3/GEMM/`, `Fase_3/Convolution/`) los invocan automáticamente al terminar la corrida, buscándolos en `../tools/` — ver el bloque final de cualquiera de esos `.sbatch`.

## `anchor_every` ya es columna real del CSV

Los seis binarios (GEMM, Convolución y Stencil, en Fase 3 y Fase 4) escriben `anchor_every` como **última** columna de `CSV_DRIFT` y `CSV_SUMMARY` (y de `CSV_ENERGY` en Stencil). Va al final para no correr ningún índice posicional que las herramientas ya usaban. Los binarios de Fase 3, que no tienen ancla, emiten `0` literal: mismo esquema de columnas entre fases, contenido correcto en ambas — lo que importa porque `run_full_pipeline.sh` concatena los `results/` de las dos fases en el mismo análisis.

**Dos semánticas distintas bajo el mismo nombre de columna** — no las confundas:

| | GEMM / Convolución | Stencil |
|---|---|---|
| Alcance | **Por fila**: `_none` reporta `0` y `_comp` reporta `K`, dentro de la *misma* invocación (ambas rutas corren en la misma pasada del binario). | **Por invocación**: todas las filas comparten el valor, incluidas las rutas de referencia `GPU_FP64`/`CPU_FP64` que nunca ejecutan el ancla. |
| Cómo está implementado | Se imprime en el punto de uso, desde `opt.anchor_every`, ruta por ruta. | Sale de `g_anchor_every_csv`, una variable de alcance de proceso que `main()` fija una sola vez. |

Esa asimetría de implementación es deliberada: codifica la diferencia semántica en la estructura del código, para que no se pueda perder en una refactorización. Los gates de `Fase_4/tools/gate3_ancla.py` comparan *valores*, no significados — no atraparían un cruce entre las dos convenciones.

Los extractores conservan el camino de **respaldo** (reconstruir `anchor_every` desde la línea de configuración que el binario imprime al arrancar) para logs generados antes de que la columna existiera. Borrarlo volvería inanalizable cualquier log anterior al cambio y no cuesta nada mantenerlo.

## Qué falta

- Migrar `validar_gate2.py`/`gate1_regresion.sbatch` (hoy solo en `old/Fase_4/Stencil/`), que validan **otras** dos cosas: regresión de la parametrización del operador y checkpoints/archivado. No son el gate del ancla — ese ya existe, ver `Fase_4/tools/gate3_ancla.py`.
- **`t_iter_ms`/`energy_gpu_j` no distinguen la ruta `_none` de la `_comp`** en GEMM y Convolución: un único cronómetro envuelve las tres trayectorias de cada iteración (referencia FP64 + WMMA sin comp + WMMA con comp) y se imprime idéntico en las dos filas, y los dos `PowerBuffer` se abren y cierran en los mismos instantes sobre un contador NVML que es **de todo el dispositivo**. Consecuencia: los ejes tiempo y energía del Frente de Pareto 3D de esos dos kernels no distinguen `comp=off` de `comp=on`, y ambos están contaminados por el costo de la referencia FP64. Stencil no tiene este problema (cronometra y mide energía por ruta). Arreglarlo exige reescribir el bucle de medición de los cuatro `.cu` encadenados y está fuera del alcance de la auditoría que agregó esta nota.
