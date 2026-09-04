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

## Validación: dos puertas antes de confiar en cualquier número

Ver `Fase_4/GEMM/README.md`, sección de Validación, para el detalle completo: `CSV_DRIFT` compara la referencia FP64 contra `T` tal cual se guarda, nunca contra `T+comp`, así que el piso reportable de `rel_l2` está acotado por la precisión de `T` (~`1e-3` a `1e-4` en FP16), sin importar qué tan exacta sea la reconstrucción interna del ancla.

- **`--anchor-every 1`** (ancla en cada iteración): verificar que `rel_l2`/`rel_linf` sean medibles y estables (no crecientes) a lo largo de muchas iteraciones, y que difieran de la ruta K=0 (evidencia de que el ancla realmente se ejecuta, no un no-op silencioso).
- **`--anchor-every 0`** debe ser bit-idéntico a correr `Fase_3/Convolution/conv_chained.cu` con los mismos flags.

Verificar con `--hw 64` (el mínimo) antes de cualquier campaña con `--hw` mayor.

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
