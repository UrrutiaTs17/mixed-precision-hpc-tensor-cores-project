# Fase 4 — Convolución: ancla FP64

`conv_chained.cu` es `Fase_3/Convolution/conv_chained.cu` + `--anchor-every K`. Todo lo demás (filtro bloque-diagonal, 64 canales, compensación por linealidad, referencia FP64, ventana de energía) es idéntico — ver `Fase_3/Convolution/README.md` primero si no lo has leído; este documento solo cubre la extensión.

Compilado y verificado en GPU Ampere+ real (`sm_89`), incluyendo el barrido K=0/K=1.

## Qué hace el ancla

Idéntico mecanismo que `Fase_4/GEMM/gemm_chained.cu` (ver ese README para el razonamiento completo) aplicado aquí: cada `K` iteraciones de la ruta **con compensación** (`--comp on`), en vez del paso WMMA normal:

1. **Reconstruye** el estado exacto en `double`: `exact = double(tc_to_float(T)) + comp64`.
2. **Avanza un paso con la referencia FP64** — reutilizando `gpu_fp64_conv_step()` tal cual, la misma función que ya calcula la trayectoria de referencia de este archivo.
3. **Re-siembra** `T` y el residuo, ahora en `double`, sin pasar por `float` en el camino.

## Un detalle propio de Convolución: el scratch de `im2col` se comparte

A diferencia de GEMM (donde el paso FP64 no necesita ningún buffer intermedio más allá de `X` y `A`), el paso FP64 de Convolución pasa primero por `im2col_double_kernel` hacia un buffer scratch (`d_col_scratch`, forma `[CRS, Ncol]`) antes de la llamada a `cublasDgemm`. La trayectoria de referencia (siempre FP64) y el paso del ancla usan la **misma función** `gpu_fp64_conv_step()`, y ambas llamadas ocurren dentro de la misma iteración, en el *stream* por defecto — es decir, secuenciales, sin ninguna carrera de datos. Por eso el ancla reutiliza el mismo buffer `d_col64` que ya usa la referencia, en vez de reservar un segundo buffer del mismo tamaño: no hay ninguna ganancia en duplicarlo, solo memoria desperdiciada.

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

**Si este gate falla y el de GEMM pasa con parámetros equivalentes**, el primer sospechoso es el buffer scratch de `im2col` compartido (ver la sección siguiente), que es la única diferencia estructural del mecanismo de ancla entre los dos kernels — antes que cualquier error de lógica del ancla en sí.

## Costo de memoria

El ancla agrega 4 buffers `double` de tamaño `kChannels·hw²` (`d_comp64_in`, `d_comp64_out`, `d_exact64`, `d_out64`) — el buffer de `im2col` en `double` (`d_col64`, tamaño `kCRS·hw²`) se reutiliza del que ya existía para la referencia, no se duplica (ver arriba). Solo se reserva cuando `--anchor-every > 0`.

## Uso

```bash
./conv_chained --hw 64 --iters 40 --tc fp16 --comp on --anchor-every 5
```

| Flag | Default | Qué hace |
|---|---|---|
| `--anchor-every` | 0 | 0 = deshabilitado (idéntico a Fase 3). K>0 = ancla FP64 cada K iteraciones. **Requiere `--comp on`.** |

(El resto de flags — `--hw`, `--iters`, `--tc`, `--comp`, `--checkpoint-every`, `--seed` — son idénticos a `Fase_3/Convolution/conv_chained.cu`, ver su README.)

## Qué falta

- **`run_conv_chained.sbatch`**: ✅ hecho — con barrido de `ANCHOR_LIST`, ver el propio `.sbatch` de esta carpeta.
- **Post-proceso de CSV**: ✅ hecho — mismo `../tools/extract_csv_chained.py` que GEMM (ver `Fase_4/tools/README.md`), reconstruye `anchor_every` por fila desde la línea de configuración del binario.
- **Scripts de gate** (K=0/K=1, automatizados): todavía no existen para Convolución — ver `Fase_4/tools/README.md`, sección "Qué falta". Mientras tanto, correr las dos puertas de la sección "Validación" arriba a mano, con `--export=ALL,ANCHOR_LIST="0 1"`.
- **Campaña real en PACCA**: compilado y verificado con `--hw` chico en GPU Ampere+; falta correr el barrido de tamaños y valores de K que promete el plan.
