# Fase 4 — Stencil: ancla FP64

Extiende `Fase_3/Stencil/stencil_tensor_activation.cu` (copiado como base — ver la nota de cabecera del archivo) con el mecanismo de **ancla FP64**: cada `K` iteraciones, el paso que normalmente calcularía el kernel WMMA se recalcula completo en FP64 y el resultado reemplaza al de la ruta rápida. Especificado y corregido (dos rondas de revisión crítica) en el documento **Plan de Precisión Mixta**, secciones 01 y 02 — esa es la referencia normativa; si algo aquí no coincide con lo que el plan describe, el plan tiene razón.

## Por qué existe

La compensación (Kahan local, compensación espacial — ver `Fase_3/Stencil/README.md`) tiene un techo: como máximo recupera precisión cercana a FP32, porque no puede compensar más información de la que el Tensor Core realmente calculó. El ancla es el mecanismo para acercarse más a FP64 sin pagar su costo completo — recalculando exactamente en FP64 solo *una de cada K* iteraciones, en vez de todas.

## Qué corrige y qué no

- **Sí corrige**: en el paso de ancla no se introduce ningún error de redondeo por almacenamiento nuevo.
- **No corrige**: el drift acumulado en las iteraciones anteriores al ancla sigue presente en el campo de entrada — el ancla no reconstruye la trayectoria FP64 verdadera desde el inicio (eso costaría lo mismo que correr todo en FP64). Si frenar la introducción de error nuevo en cada ancla basta para contener el crecimiento total, o si el error heredado domina igual, es una pregunta empírica — para eso está el barrido de `K`, no para asumir la respuesta.

## Uso

```
--anchor-every K
```

Entero ≥ 0, por defecto `0` (deshabilitado — comportamiento idéntico a Fase 3). `K > 0` activa el ancla.

**Requisitos** (validados en `parse_args`, el binario aborta con un mensaje explicativo si no se cumplen):
- `--spatial-comp on` — el residuo de compensación se re-siembra en `double`, y esa reconstrucción solo está implementada sobre la convención de `CompMode::Spatial` (`Q(v) + comp = v`).
- `--execution-mode normal` (el default) — incompatible con `--execution-mode graph`: los grafos capturan de antemano una secuencia fija de lanzamientos WMMA, y el ancla necesita decidir en cada iteración si toca FP64 o WMMA. Soportarlo exigiría un tercer grafo dedicado, fuera de alcance de esta primera implementación.

Ejemplo:

```bash
./stencil_tc --nx 4096 --ny 4096 --iters 40 --tc fp16 --spatial-comp on --anchor-every 8
```

## Validación — correr esto ANTES de confiar en cualquier resultado

Estos tres gates están descritos en detalle en la sección 01/02 del documento de plan. No están automatizados todavía en un script (`gate3_ancla.sbatch` queda como trabajo pendiente, ver "Qué falta" más abajo) — corrégelos a mano hasta que ese script exista:

1. **`--anchor-every 1`** (ancla en cada iteración): la ruta debe converger a ser numéricamente indistinguible de `GPU_FP64` (comparar `rel_l2`/`rel_linf` contra esa ruta — deberían caer a nivel de ruido de punto flotante). Si no, hay un bug — probablemente el mismo tipo de error de truncamiento a `float` que ya se corrigió una vez en el diseño (ver el aviso "BUG YA ENCONTRADO" en la cabecera del `.cu`).
2. **`--anchor-every 0`**: la ruta debe ser **bit-idéntica** a correr el mismo comando sin ese flag (o, equivalentemente, a `Fase_3/Stencil/stencil_tc` con los mismos parámetros). Compara los CSV byte a byte.
3. **Barrido de `K` intermedios**: una vez 1 y 2 pasan, recién ahí tienen sentido los resultados de `K` intermedios (2, 4, 8, 16, ...) — son los que responden la pregunta real del objetivo 4.

## Costo de memoria — léelo antes de dimensionar una campaña

Habilitar el ancla ensancha el residuo de compensación de `float` (4 bytes/celda) a `double` (8 bytes/celda). El par `(T, comp)` con ancla activo pasa a ocupar **10 bytes/celda en FP16** — más que los 8 bytes de FP64 puro. No es un descuido: en cuanto se usa el ancla, la ventaja deja de ser de ancho de banda de memoria y pasa a ser puramente de throughput de cómputo (el término principal se sigue calculando con Tensor Cores). Vale la pena medir si esa ventaja de cómputo sigue compensando el costo de memoria adicional — es exactamente el tipo de resultado que debe salir del barrido de `K`, no asumirse de antemano.

## Diseño experimental (barrido de K)

Ver sección 01/02 del plan para el detalle completo. En resumen: escala logarítmica relativa al horizonte de overflow sin ancla de cada formato, nunca por encima de él (un `K` mayor que el horizonte nunca llega a dispararse antes del overflow). Ejemplo orientativo: `K ∈ {2, 4, 8, 16, ∞}` para FP16 (horizonte ≈29); `K ∈ {2, 4, 8, 16, 32, 64, ∞}` para BF16 (horizonte ≈142 con compensación espacial). **Calibra `K` con el costo real de un paso FP64 vs. un paso Tensor Core medido en este kernel** — no asumas que la escala de otro kernel (GEMM, Convolución) transfiere directamente; la brecha FP64/TC es muy distinta según si el kernel es memory-bound (Stencil) o compute-bound.

## Cómo lanzar la campaña

`run_stencil_tc.sbatch` en esta carpeta ya expone `ANCHOR_LIST` (valores de `K` separados por espacio, default `"0"` = deshabilitado) además de todos los parámetros heredados de Fase 3 (`NX`/`NY`/`ITERS_LIST`/`TC_FORMAT`/`SPATIAL_COMP`/etc.). Valida `ANCHOR_LIST` contra `SPATIAL_COMP` al arrancar (falla rápido con un mensaje claro si se pide `K>0` sin `--spatial-comp on`, en vez de dejar que el binario aborte a mitad de una campaña larga). El perfilado NCU no se repite por cada valor de `K` — el ancla no modifica el kernel WMMA que NCU perfila (ver "Sobre qué base de código se construyó" abajo), así que perfilarlo más de una vez no agrega información.

```bash
sbatch run_stencil_tc.sbatch                                      # ANCHOR_LIST=0, igual que Fase 3
sbatch --export=ALL,ANCHOR_LIST="0 1" run_stencil_tc.sbatch       # gates K=0 / K=1 (correr primero)
sbatch --export=ALL,ANCHOR_LIST="2 4 8 16" run_stencil_tc.sbatch  # barrido real
```

## Qué falta (no construido todavía en esta sesión)

- **`gate3_ancla.sbatch`**: automatizar la *comparación* de los gates de arriba (diff numérico contra `GPU_FP64` para K=1, diff bit a bit contra Fase 3 para K=0) como un script standalone, siguiendo el mismo patrón que `old/Fase_4/Stencil/gate1_regresion.sbatch`/`gate2_checkpoints.sbatch`. Las corridas en sí ya se pueden lanzar con `ANCHOR_LIST="0 1"` (ver arriba); falta automatizar la comparación de sus salidas.
- **Columna `anchor_every` en el CSV**: el mecanismo funciona y se puede invocar por CLI/sbatch, pero el esquema de columnas de `CSV_DRIFT`/`CSV_SUMMARY`/`CSV_ENERGY` (heredado de Fase 3) todavía no tiene una columna dedicada a `anchor_every` — hoy hay que llevar la cuenta de qué `K` generó cada archivo por fuera (nombre de archivo, carpeta de resultados), no confiar en que el propio CSV lo declare.
- **Frente de Pareto 3D y estadística** (etapas 7-9 del plan): dependen de que la campaña de datos con ancla ya exista.

## Sobre qué base de código se construyó

Este archivo parte de `Fase_3/Stencil/stencil_tensor_activation.cu`, que a su vez migra la versión más avanzada del kernel (rama histórica `fase4-estadistica-variabilidad`, con la reformulación `Y = X·H + V·X`, tiling ensanchado, `ldmatrix`/`mma.sync` explícito y CUDA Graphs). Busca el marcador `ANCLA FP64:` en el `.cu` para ubicar cada pieza del mecanismo — son las únicas secciones que difieren de `Fase_3/Stencil/stencil_tensor_activation.cu`.
