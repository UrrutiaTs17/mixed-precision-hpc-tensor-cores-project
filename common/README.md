# common/

Código compartido por los tres kernels (GEMM, Convolución, Stencil) y las cuatro fases. Nada específico de un kernel vive aquí — si algo empieza a necesitar `#ifdef GEMM` o similar, es señal de que no pertenece a esta carpeta.

## Archivos

### `cuda_checks.cuh`
Macros `CHECK_CUDA`, `CHECK_CUBLAS`, `CHECK_CUDNN` para validar llamadas a la API y abortar con un mensaje claro (archivo:línea + error) si algo falla. `CHECK_CUBLAS`/`CHECK_CUDNN` solo quedan definidas si el `.cu` que incluye este header ya incluyó `cublas_v2.h`/`cudnn.h` antes — por eso Stencil, que no usa ninguna de las dos librerías, simplemente no las tiene disponibles.

**Incluir dentro del bloque `namespace { ... }` anónimo de cada `.cu`**, no a nivel de archivo (ver el comentario de cabecera del propio header para el porqué).

### `metrics.cuh`
Cronómetro de eventos CUDA (`CudaEventTimer`) y las funciones de comparación numérica (`compare_fp64_ref_vs_fp32`, `compare_float_vectors`, `compare_double_vectors`) que producen `ErrorMetrics` (L2 relativo, L∞ relativo, finitud de referencia/solución). Las tres funciones de comparación comparten una sola implementación genérica (`metrics_detail::compare_sequences`) — en el código anterior estaban triplicadas casi idénticas; aquí es una plantilla, un solo lugar para corregir un bug de esta lógica.

Depende de `cuda_checks.cuh` (usa `CHECK_CUDA` en `CudaEventTimer`).

### `power_sampling.h`
Telemetría de energía GPU (NVML, contador monotónico de dos lecturas) y CPU (RAPL). Migrado sin cambios de lógica desde el código anterior (`old/Fase_3/Stencil/tools/power_sampling.h`) — es código ya probado en el clúster, con comentarios extensos sobre por qué el contador NVML se lee por delta de dos puntos y no por muestreo periódico (el sensor onboard refresca cada ~20-25 ms; un hilo de muestreo dentro de una ventana más corta que eso solo produce NaN). Léelo antes de tocar cualquier cosa relacionada con energía — la mayoría de las decisiones raras que parecen bugs a primera vista están explicadas ahí.

Opt-in por macro `-DUSE_NVML_TELEMETRY`: sin ella, el binario compila y corre igual, pero la energía GPU siempre sale `NaN` — útil para desarrollo local sin GPU de centro de datos.

### `wmma_gemm.cuh`
Kernel GEMM con Tensor Cores (API WMMA + pipeline `cp.async` de 3 etapas, Ampere `sm_80+`): `wmma_gemm_kernel<T>`, sus constantes de tiling (`kWmmaM/N/K`, `kBlockTileM/N`, `kNumStages`, etc.), y las conversiones escalares `float_to_tc_scalar<T>`/`tc_scalar_to_float<T>`. Extraído de `Fase_2/GEMM/gemm_tensor_activation.cu` (código ya migrado y verificado, sin cambios de lógica) porque Fase 3/4 lo necesitan también para el GEMM encadenado — antes de esta extracción, usarlo en más de un archivo habría significado duplicar ~185 líneas de CUDA con pipeline `cp.async`, exactamente el tipo de duplicación que esta carpeta existe para evitar.

`tc_scalar_to_float<T>` es la única pieza que no existía en el original: Fase 2 nunca necesitaba reconstruir un valor T de vuelta a `float` (el acumulador WMMA ya sale en `float`); Fase 3/4 sí, para reconstruir `Q(v) + comp` al encadenar iteraciones.

### `chained_precision.cuh`
**Código nuevo, no presente en el proyecto anterior — y hoy NO USADO por ningún `.cu` del repositorio.** Se escribió temprano en la reconstrucción como motor genérico compartido del mecanismo de ancla FP64 (plantillas parametrizadas por `low_precision_step`/`fp64_step`, para no triplicar la lógica de branching anchor/normal entre Stencil, GEMM y Convolución), tal como lo recomienda el documento *Plan de Precisión Mixta*, secciones 01/02.

**Las tres implementaciones reales del ancla (`Fase_4/Stencil/stencil_tensor_activation.cu`, `Fase_4/GEMM/gemm_chained.cu`, `Fase_4/Convolution/conv_chained.cu`) NO incluyen este header.** Cada una terminó escribiendo sus propios kernels locales de reconstrucción/reseed (`reconstruct_exact_double_kernel`, `reseed_double_from_fp64_kernel`, `widen_comp_to_double_kernel` — mismos nombres, misma lógica, repetidos en los tres archivos), en vez de instanciar la plantilla genérica de aquí. Motivo: la plantilla de este header pasa `ToFloatFn`/`FromFloatFn` como funtores de tipo `__device__` a través de un parámetro de template — funciona en el mismo archivo, pero complica el contrato de compilación (requiere `--extended-lambda` de forma consistente en cada traducción) sin ninguna ganancia real, dado que cada kernel de todas formas necesita su propia orquestación en el bucle principal (`run_chained_route()`/el bucle de Stencil) para decidir qué buffers viven y en qué orden se hace swap. Repetir ~15 líneas de kernel trivial tres veces resultó más simple y más fácil de auditar que depurar una plantilla genérica sin poder compilar en este entorno.

**Este archivo queda como referencia del diseño original, no como dependencia real.** Si en algún momento se decide consolidar los tres kernels locales en uno solo, este es el punto de partida — pero mientras eso no pase, no asumas que un cambio aquí afecta a Stencil/GEMM/Convolución: no los afecta. Nunca se ha compilado ni ejecutado.

## Por qué existe esta carpeta

El código anterior duplicaba `common.cuh` (CHECK_*, CudaEventTimer, Metrics, ErrorMetrics, compare_*) copiado dentro de cada `.cu`, y tenía dos implementaciones de telemetría de energía con APIs distintas (`EnergyProbe`/`RaplReader` en `Fase_2/telemetry.cuh`, `PowerBuffer`/`EnergyMeasurement` en `Fase_3/Stencil/tools/power_sampling.h`) que hacían básicamente lo mismo. `common/` consolida esto en un solo lugar por responsabilidad, para que corregir un bug de medición no signifique buscarlo en tres archivos.
