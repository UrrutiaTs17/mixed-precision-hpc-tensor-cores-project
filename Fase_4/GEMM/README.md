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

## Validación: dos puertas antes de confiar en cualquier número

`CSV_DRIFT` compara la referencia FP64 contra el buffer `T` (FP16/BF16) **tal cual se guarda**, nunca contra `T + comp` — así que el piso de `rel_l2` que puede reportar está acotado por la propia precisión de `T` (~`1e-3` a `1e-4` relativo en FP16), sin importar qué tan exacto sea el mecanismo interno de reconstrucción. Verificado empíricamente en GPU real: con `--n 256 --comp on`, K=0 y K=1 dan `rel_l2` casi idénticos (~`1.8e-4`) incluso a 60 iteraciones — la compensación por linealidad, sola, ya mantiene esa cota estable para este operador bien condicionado (`λ≈1.1`). No implica que el ancla no funcione: el metro que usa `CSV_DRIFT` no puede distinguir "`T+comp` exacto a `1e-16`" de "`T` exacto a `1e-4`" porque solo mira `T`.

- **`--anchor-every 1`** (ancla en cada iteración): verifica que `rel_l2`/`rel_linf` sean **medibles y estables** (no crecientes) a lo largo de muchas iteraciones, y que difieran de la ruta K=0 (evidencia de que el camino del ancla realmente se ejecuta, no un no-op silencioso — confirmado: a `--n 256 --iters 10`, K=0 y K=1 dan `rel_linf` distinto en la 2ª cifra significativa, `0.000244264` vs `0.000244145`). Para una prueba más estricta de la reconstrucción interna, comparar `T+comp` (no solo `T`) contra la referencia — no lo hace `CSV_DRIFT` hoy, ver "Qué falta".
- **`--anchor-every 0`** (deshabilitado) debe ser bit-idéntico a correr `Fase_3/GEMM/gemm_chained.cu` con los mismos flags — el código de la ruta normal no cambió una sola línea, solo se envolvió en un `if`.

Ninguna campaña real (`K` intermedios, ej. 5/10/20) tiene sentido reportar antes de verificar estas dos puertas con un `--n` chico (64 o 128).

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

## Qué falta

- **`run_gemm_chained.sbatch`**: ✅ hecho — con barrido de `ANCHOR_LIST`, ver el propio `.sbatch` de esta carpeta.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py` reconstruye `anchor_every` por fila leyendo la línea de configuración que el binario imprime al arrancar cada corrida (el CSV en sí no trae esa columna todavía — ver `Fase_4/tools/README.md`).
- **Scripts de gate** (K=0/K=1, automatizados): todavía no existen para GEMM — ver `Fase_4/tools/README.md`, sección "Qué falta". Mientras tanto, correr las dos puertas de la sección "Validación" arriba a mano, con `--export=ALL,ANCHOR_LIST="0 1"`.
- **Campaña real en PACCA**: compilado y verificado con `--n` chico en GPU Ampere+; falta correr el barrido de tamaños y valores de K que promete el plan.
