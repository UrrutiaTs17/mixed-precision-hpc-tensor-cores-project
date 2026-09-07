# Fase 4 — Convolución: ancla FP64

`conv_chained.cu` es `Fase_3/Convolution/conv_chained.cu` + `--anchor-every K`. Todo lo demás (filtro bloque-diagonal, 64 canales, compensación por linealidad, referencia FP64, ventana de energía) es idéntico — ver `Fase_3/Convolution/README.md` primero si no lo has leído; este documento solo cubre la extensión.

Compilado y verificado en GPU Ampere+ real (`sm_89`), incluyendo el barrido K=0/K=1.

## Qué hace el ancla

Idéntico mecanismo que `Fase_4/GEMM/gemm_chained.cu` (ver ese README para el razonamiento completo) aplicado aquí: cada `K` iteraciones de la ruta **con compensación** (`--comp on`), en vez del paso WMMA normal:

1. **Reconstruye** el estado exacto en `double`: `exact = double(tc_to_float(T)) + comp64`.
2. **Avanza un paso con la referencia FP64** — reutilizando `gpu_fp64_conv_step()` tal cual, la misma función que ya calcula la trayectoria de referencia de este archivo.
3. **Re-siembra** `T` y el residuo, ahora en `double`, sin pasar por `float` en el camino.

## Un detalle propio de Convolución: el scratch de `im2col` (ya no compartido)

A diferencia de GEMM (donde el paso FP64 no necesita ningún buffer intermedio más allá de `X` y `A`), el paso FP64 de Convolución pasa primero por `im2col_double_kernel` hacia un buffer scratch (`d_col_scratch`, forma `[CRS, Ncol]`) antes de la llamada a `cublasDgemm`.

**Esto era, hasta 2026-09-06, la única diferencia estructural del mecanismo de ancla frente a GEMM**: la trayectoria de referencia y el paso del ancla usaban la misma función `gpu_fp64_conv_step()` dentro de la misma iteración del mismo bucle, y compartían el buffer `d_col64`. Era seguro (mismo *stream*, llamadas secuenciales), pero obligaba a plantearse la pregunta sobre carreras de datos cada vez que alguien tocaba el bucle.

Desde que la medición se separó por fases (ver más abajo), **el buffer ya no se comparte**: la fase de referencia libera su scratch antes de que empiecen las rutas WMMA, y el ancla reserva el suyo. La pregunta desaparece, y con ella la asimetría respecto a GEMM. El pico de memoria no sube — de hecho baja un poco, porque los dos buffers ya no coexisten.

## La siembra inicial de `comp`

Mismo criterio que `Fase_4/GEMM/README.md`: `comp`/`comp64` arrancan sembrados con el redondeo *real* de `x0→T` (`seed_comp_from_double_kernel`/`seed_comp64_from_double_kernel`), no desde cero — de lo contrario la primera iteración reconstruiría `dequantize(T(x0))` en vez de `x0`, y esa diferencia se amplificaría en cada paso sin que nada la corrija.

## Validación: dos puertas, ahora automatizadas

```bash
sbatch gate3_ancla.sbatch          # o: bash gate3_ancla.sbatch, sin SLURM
```

Compila los **dos** binarios (Fase 3 y este), corre las tres pasadas (`Fase_3` sin el flag, `Fase_4` con `--anchor-every 0` y con `--anchor-every 1`) a `--hw 64` y le pasa los logs a `../tools/gate3_ancla.py --kernel conv`.

- **`--anchor-every 0`** debe reproducir `Fase_3/Convolution/conv_chained.cu` columna por columna en lo determinista. Las columnas de tiempo/energía se reportan como desviación relativa pero **no deciden**: en la corrida real de validación variaron un 59 % entre dos pasadas del mismo código mientras lo determinista salía idéntico.
- **`--anchor-every 1`** debe cumplir la **cota de cuantización** del formato: `rel_l2 ≤ 2^-p` y `rel_linf ≤ 2^-p`, con `p=11` en FP16 (`4.883e-04`) y `p=8` en BF16 (`3.906e-03`). No es un umbral inventado: con K=1 la reconstrucción interna es exacta, así que lo único que separa a `T` de la referencia es el redondeo al formato de 16 bits. Ver `Fase_4/tools/README.md` para la derivación.

**Verificado en GPU Ampere real** (`sm_86`, `--hw 64 --iters 12 --tc both --comp on --anchor-every 1`): FP16 llegó a `rel_linf = 3.74e-04` y BF16 a `3.00e-03`, ambos al **0.77** de su cota — el mismo factor en los dos formatos. Las dos puertas pasan.

**Ojo con el criterio ingenuo**: "con K=1 el error debe caer a `1e-16`" es imposible aquí, y no por un bug — `CSV_DRIFT` compara la referencia FP64 contra `T` **tal cual se guarda**, nunca contra `T+comp`, así que el piso está acotado por la precisión de `T` sin importar qué tan exacta sea la reconstrucción interna. Es la misma limitación que ya documenta `Fase_4/GEMM/README.md`.

**Si este gate falla y el de GEMM pasa con parámetros equivalentes**, el sospechoso histórico era el buffer scratch de `im2col` compartido entre la referencia y el ancla. Desde la separación de fases ese buffer ya no se comparte (ver arriba), así que esa hipótesis está descartada por construcción: hoy los dos mecanismos de ancla son estructuralmente idénticos y una divergencia entre kernels apunta al `im2col` o al filtro, no al ancla.

## Medicion por ruta: el costo del ancla ahora SI se ve

Hasta 2026-09-06, `t_iter_ms`/`gflops`/`energy_gpu_j` de este binario no
distinguian la ruta `_none` de la `_comp` (un solo cronometro envolvia las tres
trayectorias) y ademas descontaban del tiempo medido el computo que el
`cudaMemcpy` del checkpoint absorbia al esperar la cola asincrona. Ver
`Fase_3/GEMM/README.md`, seccion "Los numeros de tiempo/energia anteriores a
2026-09-06 no son utilizables", para el detalle completo y la tabla de
antes/despues.

**En Fase 4 el dano era mayor que en Fase 3**: el costo del ancla -- que es
justo lo que el barrido de `K` existe para medir -- tampoco era visible, porque
el tiempo de la ruta compensada no era suyo. Con la medicion por fases, el
efecto aparece de inmediato (GPU Ampere real, `--hw 128 --iters 20 --tc fp16`):

| | `t_iter_ms` de `FP16_comp` |
|---|---|
| `--anchor-every 0` | 5.74 |
| `--anchor-every 1` | 19.33 (**3.37x**) |

Ese 3.37x es lo esperado en una tarjeta con FP64 a 1/64 del ritmo: anclar en
cada iteracion agrega un `cublasDgemm` completo por paso. En A100 el factor
sera muy distinto (FP64 a 1/2), y medirlo es exactamente el objetivo del
barrido de `K`.

**Los dos costos FP64 ahora estan separados**, que antes no lo estaban:
- La trayectoria de REFERENCIA (con la que se mide el error) queda fuera de las
  rutas de baja precision, publicada como ruta propia `GPU_FP64`.
- Los pasos FP64 que el ANCLA inyecta quedan DENTRO de la ruta compensada, que
  es donde corresponde.

Lo vigila `../tools/gate4_medicion.py`, que corre en
`tools/validacion_preliminar.sbatch`.

## Costo de memoria

El ancla agrega 4 buffers `double` de tamaño `kChannels·hw²` (`d_comp64_in`, `d_comp64_out`, `d_exact64`, `d_out64`) más su propio buffer de `im2col` en `double` (`kCRS·hw²`, que es 9× el campo y por tanto el término dominante). Todo eso solo se reserva cuando `--anchor-every > 0`.

Aunque el ancla ya no reutilice el scratch de la referencia, **el pico de memoria baja**: la fase de referencia libera sus buffers (`d_x64_in/out` + su `d_col64`) antes de que las rutas WMMA reserven los suyos, así que los dos scratch de `im2col` nunca coexisten. Del cálculo del `.sbatch`, 202 B por elemento de campo pasan a ~194 B; los límites de `HW_LIST` no cambian.

## Uso

```bash
./conv_chained --hw 64 --iters 40 --tc fp16 --comp on --anchor-every 5
```

| Flag | Default | Qué hace |
|---|---|---|
| `--anchor-every` | 0 | 0 = deshabilitado (idéntico a Fase 3). K>0 = ancla FP64 cada K iteraciones. **Requiere `--comp on`.** |

(El resto de flags — `--hw`, `--iters`, `--tc`, `--comp`, `--checkpoint-every`, `--seed` — son idénticos a `Fase_3/Convolution/conv_chained.cu`, ver su README.)

## Campaña por defecto

`run_conv_chained.sbatch` corre el barrido completo si no se le exporta nada:

| Variable | Default | Nota |
|---|---|---|
| `HW_LIST` | `64 128 256 512` | Techo por memoria: 202 B/elemento de campo (12.9 KB por celda espacial) → 3.39 GB a `hw=512`, margen 12×. `hw=1024` (13.6 GB) es el techo real bajo el criterio de 2×; `hw=2048` (54.2 GB) es imposible. El cálculo completo, y por qué **no** es un workspace de cuDNN, están en `Fase_3/Convolution/README.md` y en el propio `.sbatch`. |
| `ITERS_LIST` | `20 40 80` | **Nueva**: reemplaza al escalar `ITERS`, que sigue funcionando y gana si se exporta. |
| `ANCHOR_LIST` | `0 1 5 20` | Misma escala que GEMM. |
| `COMP_LIST` | `off on` | `ANCHOR_LIST` solo se recorre con `COMP=on`. |
| `SMOKE_TEST` | `0` | `1` recorta a `hw=64`, 3 iteraciones, `ANCHOR_LIST="0 1"` y `RUN_NCU=0`. |

## `anchor_every` en el CSV: columna real, **por fila**

Última columna de `CSV_DRIFT` y `CSV_SUMMARY`. Varía dentro de la misma invocación (`_none` = `0`, `_comp` = `K`), igual que en GEMM y a diferencia de Stencil. Comprobado en GPU real:

```
CSV_DRIFT,FP16_none,64,2,0.000322504,0.000595989,1,0
CSV_DRIFT,FP16_comp,64,2,0.000205673,0.000300221,1,2
```

## Qué falta

- **`run_conv_chained.sbatch`**: ✅ hecho — con `ANCHOR_LIST`, `ITERS_LIST` y `SMOKE_TEST`.
- **Post-proceso de CSV**: ✅ hecho — mismo `../tools/extract_csv_chained.py` que GEMM, que ahora lee `anchor_every` **directo de la fila**.
- **Scripts de gate** (K=0/K=1, automatizados): ✅ hechos — `gate3_ancla.sbatch` de esta carpeta y `../tools/gate3_ancla.py`. Las dos puertas **pasan** en GPU Ampere real.
- **`t_iter_ms`/`gflops`/`energy_gpu_j` no distinguen `_none` de `_comp`**: mismo hallazgo (y misma causa) que documenta `Fase_4/GEMM/README.md` en su "Qué falta". Sin corregir.
- **Campaña real en PACCA**: compilado y verificado con `--hw` chico; falta el barrido completo. Antes de lanzarlo, `tools/validacion_preliminar.sbatch`.
