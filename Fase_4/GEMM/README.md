# Fase 4 — GEMM: ancla FP64

`gemm_chained.cu` es `Fase_3/GEMM/gemm_chained.cu` + `--anchor-every K`. Todo lo demás (operador `A = c·H`, compensación por linealidad, referencia FP64, ventana de energía) es idéntico — ver `Fase_3/GEMM/README.md` primero si no lo has leído; este documento solo cubre la extensión.

Compilado y verificado en GPU Ampere+ real (`sm_89`), incluyendo el barrido K=0/K=1.

## Qué hace el ancla

Cada `K` iteraciones de la ruta **con compensación** (`--comp on` — el ancla no tiene sentido sin ella, ver validación abajo), en vez del paso WMMA normal:

1. **Reconstruye** el estado exacto en `double`: `exact = double(tc_to_float(T)) + comp64`.
2. **Avanza un paso con la referencia FP64** — reutilizando `gpu_fp64_step()` tal cual, la misma función que ya calcula la trayectoria de referencia de este archivo. El ancla no agrega ninguna llamada a cuBLAS nueva.
3. **Re-siembra** `T` (cuantizado desde el resultado exacto) y el residuo, ahora en `double`, sin pasar por `float` en el camino.

En las iteraciones que no son de ancla, el residuo `float` (el de siempre, calculado por `finalize_step_kernel`) se ensancha a `double` (`widen_comp_to_double_kernel`) para que el estado `double` esté listo si la *siguiente* iteración sí es de ancla. Después de una iteración de ancla, el residuo `double` recién calculado se angosta a `float` (`narrow_double_to_float_kernel`) para que la siguiente iteración normal (que usa `cast_float_to_tc_kernel`, en `float`) tenga de dónde partir.

Es exactamente el mismo mecanismo que `Fase_4/Stencil/stencil_tensor_activation.cu` (buscar `ANCLA FP64` en ambos archivos), adaptado a que aquí el paso encadenado ya estaba compuesto de kernels separados — no hay un kernel WMMA monolítico que evitar tocar, y no existe `ExecutionMode::Graph` en este archivo (nada que declarar incompatible).

## Por qué el residuo del ancla es `double`, no `float`

Si el residuo tras un paso de ancla se guardara en `float` (truncando el resultado de `gpu_fp64_step`), `--anchor-every 1` — un ancla en *cada* iteración — dejaría de reproducir exactamente la trayectoria FP64 pura: el truncamiento a `float` introduciría un error que no está en la referencia. `reseed_double_from_fp64_kernel` calcula `comp64_out = out64 - double(tc_to_float(q))` enteramente en `double` para evitarlo — mismo criterio que documenta el comentario de cabecera de `Fase_4/Stencil/stencil_tensor_activation.cu`.

## La siembra inicial de `comp`

El residuo de compensación se siembra desde el redondeo *real* de `x0→T` (`seed_comp_from_double_kernel`/`seed_comp64_from_double_kernel`, en `Fase_3/GEMM/gemm_chained.cu` y este archivo respectivamente) — **no desde cero**. Con `comp` en cero, la primera iteración reconstruiría un estado "exacto" que en realidad no es `x0` sino `dequantize(T(x0))`, y esa diferencia se amplificaría por `A` en cada paso sin que nada la corrija (ni la compensación ni el ancla evitan un error que ya se perdió antes de que `comp` empezara a rastrearlo) — mismo patrón que `Fase_4/Stencil`, ver el comentario de `seed_comp_from_double_kernel` en el `.cu`. **Esto no es lo mismo que el piso de `1e-16` que el gate K=1 nunca puede alcanzar** (ver la sección de Validación, abajo) — son dos limitaciones distintas: la siembra determina desde qué estado arranca la trayectoria; el piso de `rel_l2` es una limitación estructural de qué compara `CSV_DRIFT`.

## Validación: dos puertas, ahora automatizadas

```bash
sbatch gate3_ancla.sbatch          # o: bash gate3_ancla.sbatch, sin SLURM
```

`gate3_ancla.sbatch` compila los **dos** binarios (Fase 3 y este), corre las tres pasadas que las puertas necesitan (`Fase_3` sin el flag, `Fase_4` con `--anchor-every 0` y con `--anchor-every 1`) y le pasa los logs a `../tools/gate3_ancla.py --kernel gemm`, que decide y sale con `0`/`1`/`2`. Es deliberadamente chico y corto (`N=256`, `--time 00:30:00`): un error de lógica del ancla se delata igual a `N=256` que a `N=8192`, y este job tiene que poder colarse en la cola antes de comprometer las horas de la campaña.

**El gate K=1 no usa el criterio ingenuo, y no puede.** "Con K=1, `rel_l2` debe caer a nivel de ruido de punto flotante (~`1e-16`)" es imposible aquí por lo que `CSV_DRIFT` compara — ver la sección siguiente. El criterio real es una **cota derivada**: con K=1 la reconstrucción interna es exacta (`comp64 = out64 − dequant(q)` es una resta exacta por Sterbenz, luego `dequant(q) + comp64 == out64` bit a bit), así que lo único que separa a `T` de la referencia es la cuantización, y por tanto `rel_l2 ≤ 2^-p` y `rel_linf ≤ 2^-p` con `p=11` en FP16 (`4.883e-04`) y `p=8` en BF16 (`3.906e-03`). Medido en GPU real, ambos formatos caen al **0.77** de su cota respectiva — el mismo factor en los dos, que es la confirmación de que el modelo es correcto.

El gate está probado positiva y negativamente sobre logs reales (`sm_86`, 2026-09-06): pasa con la corrida buena, falla con `1` si se corrompe un `rel_l2` determinista o si se infla un `rel_linf` por encima de la cota, y devuelve `2` (no evaluable, que no es un "pasa") si se le pasa el log equivocado.

## Qué comparan realmente esas puertas

`CSV_DRIFT` compara la referencia FP64 contra el buffer `T` (FP16/BF16) **tal cual se guarda**, nunca contra `T + comp` — así que el piso de `rel_l2` que puede reportar está acotado por la propia precisión de `T` (~`1e-3` a `1e-4` relativo en FP16), sin importar qué tan exacto sea el mecanismo interno de reconstrucción. Verificado empíricamente en GPU real: con `--n 256 --comp on`, K=0 y K=1 dan `rel_l2` casi idénticos (~`1.8e-4`) incluso a 60 iteraciones — la compensación por linealidad, sola, ya mantiene esa cota estable para este operador bien condicionado (`λ≈1.1`). No implica que el ancla no funcione: el metro que usa `CSV_DRIFT` no puede distinguir "`T+comp` exacto a `1e-16`" de "`T` exacto a `1e-4`" porque solo mira `T`.

- **`--anchor-every 1`** (ancla en cada iteración): verifica que `rel_l2`/`rel_linf` sean **medibles y estables** (no crecientes) a lo largo de muchas iteraciones, y que difieran de la ruta K=0 (evidencia de que el camino del ancla realmente se ejecuta, no un no-op silencioso — confirmado: a `--n 256 --iters 10`, K=0 y K=1 dan `rel_linf` distinto en la 2ª cifra significativa, `0.000244264` vs `0.000244145`). Para una prueba más estricta de la reconstrucción interna, comparar `T+comp` (no solo `T`) contra la referencia — no lo hace `CSV_DRIFT` hoy, ver "Qué falta".
- **`--anchor-every 0`** (deshabilitado) debe ser bit-idéntico a correr `Fase_3/GEMM/gemm_chained.cu` con los mismos flags — el código de la ruta normal no cambió una sola línea, solo se envolvió en un `if`.

Ninguna campaña real (`K` intermedios, ej. 5/10/20) tiene sentido reportar antes de verificar estas dos puertas con un `--n` chico (64 o 128).

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
efecto aparece de inmediato (GPU Ampere real, `--n 1024 --iters 20 --tc fp16`):

| | `t_iter_ms` de `FP16_comp` |
|---|---|
| `--anchor-every 0` | 3.32 |
| `--anchor-every 1` | 24.05 (**7.25x**) |

Ese 7.25x es lo esperado en una tarjeta con FP64 a 1/64 del ritmo: anclar en
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

El ancla agrega 4 buffers `double` de tamaño `N²` (`d_comp64_in`, `d_comp64_out`, `d_exact64`, `d_out64`) — 32 bytes/celda adicionales sobre lo que ya usa la ruta con compensación, solo cuando `--anchor-every > 0`. Con `--anchor-every 0` no se reserva nada de esto (`anchor_enabled = false`).

## Uso

```bash
./gemm_chained --n 1024 --iters 20 --tc fp16 --comp on --anchor-every 5
```

| Flag | Default | Qué hace |
|---|---|---|
| `--anchor-every` | 0 | 0 = deshabilitado (idéntico a Fase 3). K>0 = ancla FP64 cada K iteraciones. **Requiere `--comp on`.** |

(El resto de flags — `--n`, `--iters`, `--tc`, `--comp`, `--checkpoint-every`, `--csv`, `--seed`, `--target-lambda` — son idénticos a `Fase_3/GEMM/gemm_chained.cu`, ver su README.)

## Campaña por defecto

`run_gemm_chained.sbatch` corre el barrido completo si no se le exporta nada:

| Variable | Default | Nota |
|---|---|---|
| `N_LIST` | `1024 2048 4096 8192` | Techo por memoria: 90 B/elemento con `--tc both`, `--comp on` y ancla → 6.04 GB a `N=8192`. `N=16384` (24.2 GB, 61 % de la tarjeta) queda **descartado** por falta de margen. El cálculo completo está en el `.sbatch`. |
| `ITERS_LIST` | `20 40 80` | **Nueva**: reemplaza al escalar `ITERS`, que sigue funcionando y gana si se exporta. |
| `ANCHOR_LIST` | `0 1 5 20` | `0` control, `1` gate de exactitud, `5` y `20` los intermedios que responden la pregunta real. |
| `COMP_LIST` | `off on` | `ANCHOR_LIST` solo se recorre con `COMP=on`. |
| `SMOKE_TEST` | `0` | `1` recorta a `N=1024`, 3 iteraciones, `ANCHOR_LIST="0 1"` y `RUN_NCU=0`. |

## `anchor_every` en el CSV: columna real, **por fila**

`CSV_DRIFT` y `CSV_SUMMARY` la traen como última columna. Varía **dentro de la misma invocación**: la ruta `_none` reporta `0` siempre y la `_comp` el valor real de `K`, porque las dos rutas corren en la misma pasada del binario y el ancla solo aplica a la compensada (`parse_args` exige `--comp on` para `K>0`).

Eso **no** es lo mismo que en Stencil, donde `anchor_every` es una constante de la invocación completa que comparten hasta las filas de `GPU_FP64`/`CPU_FP64`. Ver `Fase_4/tools/README.md`. Comprobado en GPU real:

```
CSV_SUMMARY,FP16_none,64,4,20.8294,83.3177,14.4982,NA,0,3,0
CSV_SUMMARY,FP16_comp,64,4,20.8294,83.3177,14.4982,NA,0,3,2
```

## Qué falta

- **`run_gemm_chained.sbatch`**: ✅ hecho — con `ANCHOR_LIST`, `ITERS_LIST` y `SMOKE_TEST`.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py` lee `anchor_every` **directo de la fila** (conserva la reconstrucción desde la línea de cabecera solo como respaldo para logs viejos).
- **Scripts de gate** (K=0/K=1, automatizados): ✅ hechos — `gate3_ancla.sbatch` de esta carpeta y `../tools/gate3_ancla.py`.
- **`t_iter_ms`/`gflops`/`energy_gpu_j` no distinguen `_none` de `_comp`**: un solo cronómetro envuelve las tres trayectorias de cada iteración (referencia FP64 + WMMA sin comp + WMMA con comp) y se imprime idéntico en las dos filas; los dos `PowerBuffer` se abren y cierran en los mismos instantes sobre un contador NVML **de todo el dispositivo**, así que las dos columnas de energía son el mismo número. Los ejes tiempo y energía del Frente de Pareto de este kernel no pueden separar `comp=off` de `comp=on`, y ambos incluyen el costo de la referencia FP64 — que en A100 domina. Hallazgo de la auditoría de 2026-09-06, **sin corregir**: exige reescribir el bucle de medición de los cuatro `.cu` encadenados. Se ve en el ejemplo de arriba (`20.8294` en las dos filas).
- **Campaña real en PACCA**: compilado y verificado con `--n` chico; falta el barrido completo. Antes de lanzarlo, `tools/validacion_preliminar.sbatch`.
