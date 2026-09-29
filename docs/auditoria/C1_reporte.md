# Auditoría C1: Tabla `tab:conv-mixta` (Convolución 2D, Fase 2)

Hallazgo C1 (crítico) de la revisión del informe final (28/09/2026). Este
reporte rastrea cada celda de `tab:conv-mixta` hasta una línea de log con job
ID, fija los criterios de aceptación **antes** de lanzar cualquier job nuevo y
registra el resultado pasa/falla de la re-ejecución.

- Rama: `auditoria-c1-conv-fase2`.
- Configuración implícita de la tabla: `N=1, C=K=1024, H=W=256, R=S=3`,
  padding 1, stride 1, dilation 1. `conv_flops = 2·N·K·outH·outW·C·R·S =
  1 236 950 581 248 FLOP` (1,23695e12).
- Referencia de la corrección de TF32: commit `d6614ce` (2026-08-05 20:15 -0500),
  `conv_tensor_activation.cu:629-634` (`cudnnSetConvolutionMathType(convDesc,
  CUDNN_FMA_MATH)`). Los binarios posteriores imprimen
  `GPU cuDNN FP32 escalar      : math type CUDNN_FMA_MATH (TF32 desactivado)`;
  los anteriores imprimen `GPU cuDNN clasico - tiempo`.
- Zonas horarias: la línea `Fecha:` de los logs está en EDT (-04); `sacct` y
  los commits están en -05. 4613 empezó a las 18:46:48 -05 (19:46:48 EDT),
  29 minutos **antes** del commit `d6614ce`.

## 1. Inventario de evidencia

### 1.1 Fuentes buscadas

| Lugar | Qué se buscó | Resultado |
| - | - | - |
| PACCA `~/mixed-precision-hpc-tensor-cores-project/Fase_2/Convolution/logs/` | `mixed_precision_conv_tc_*.out/.err` | 5 jobs (6872, 6876, 6880, 6882, 6884), todos del 2026-09-10 y con `C=K=64` |
| PACCA `~` (profundidad 6, sin `hyperion-results`) | `mixed_precision_conv_tc_*`, `*conv_tc*.out` | Solo los 5 anteriores |
| PACCA `sacct -u latorresn -S 2026-07-01` | jobs `mixed_precision_conv_tc` | Además: 4061, 4064, 4065, 4098, 4115, 4118, 4592, 4598, 4613, 4622, 4626, 6860. Sin log conservado, salvo 4613 (copia local) |
| Local `~/Documentos/Resultados_PACCA/` (respaldo 2026-08-30) | `RESULTADOS CONV 2D`, `conv_tc` | Nada de Fase 2 Convolución |
| Local `~/Documentos/Proyecto_de_Grado/Pruebas/` y todos los `.zip` de `~/Documentos` y `~/Descargas` | ídem, y los valores de la tabla (`153.959`, `149195`, `79.145`, `358.355`) | `Resultados_05_08_Fase_2/` (y los zip `Resultados_05_08 (1).zip`, `Resultados_05_08_Fase_2.zip`, idénticos por md5): jobs 4609, 4610, 4612, 4613 |
| Repositorio (árbol e historial `git log -S`) | valores de la tabla | Nada |

Job 4626 (`COMPLETED`, 2026-08-05 21:36 -05, 10:49 min, directorio
`Fase_2/Convolution`) es posterior al fix y dura lo mismo que 4613, por lo que
probablemente fue la corrida `C=1024` corregida, pero **su log no se conservó
en ninguna parte**: no es evidencia.

### 1.2 Logs disponibles

Ruta local de los cuatro primeros: `~/Documentos/Proyecto_de_Grado/Pruebas/Resultados_05_08_Fase_2/`.
Los 68xx están en PACCA (`Fase_2/Convolution/logs/`); sha256 verificado igual en PACCA y en la copia local.

| Job | Fecha (log) | Binario | Configuración | Rutas | Línea `CUDNN_FMA_MATH` | Estado | sha256 (prefijo) |
| - | - | - | - | - | - | - | - |
| 4609 | 2026-08-05 18:06 EDT | Fase 1 GEMM (`fase1_gemm_baseline`) | GEMM M=N=K=12288 y 32768, FP32/FP64 | CPU BLAS, cuBLAS | n/a | OK | `6eb877aa` |
| 4610 | 2026-08-05 18:34 EDT | Fase 1 Conv (`cudnn_conv_balanced`, `IMPLICIT_GEMM` forzado, antes de `d6614ce`) | Conv N=1, C=K=1024 y 2048, H=W=256, R=S=3, FP32 | CPU OpenBLAS, cuDNN FP32 | n/a (Fase 1) | OK, pero error vs CPU 1,414 (bug de layout BLAS de la referencia CPU, corregido en `d6614ce`) | `61798d2e` |
| 4612 | 2026-08-05 18:38 EDT | Fase 2 GEMM (`mixed_precision_gemm_tc`) | GEMM M=N=K=12288 y 32768 | CPU, cuBLAS clásico FP32, cuBLAS TC, WMMA | n/a | OK | `eac9b085` |
| **4613** | 2026-08-05 19:46 EDT | Fase 2 Conv, **antes de `d6614ce`** | **N=1, C=K=1024, H=W=256, R=S=3**; y C=K=2048 (ITERS=5) | CPU FP32, cuDNN FP32 (DEFAULT_MATH), cuDNN TC FP16, WMMA FP16; **sin BF16** (el sbatch no pasaba `--tc-format`) | **No** (imprime `GPU cuDNN clasico`) | OK | `0636601c` |
| 6872 | 2026-09-10 17:53 EDT | Fase 2 Conv, actual | C=K=64, HW 64..512 | ninguna | — | FAILED: `libopenblas.so.0` no encontrada en runtime | `b0f22c6f` |
| 6876 | 2026-09-10 17:55 EDT | ídem | ídem | ninguna | — | FAILED: `nvcc fatal: Unknown option '-Wl,-rpath,...'` | `5fccb3a0` |
| 6880 | 2026-09-10 18:02 EDT | ídem | ídem | ninguna | — | FAILED: `--cutlass` sin CUTLASS en el include path | `07fd926c` |
| 6882 | 2026-09-10 18:06 EDT | ídem | C=K=64, HW 64/128/256/512 | CPU, cuDNN FP32, TC FP16, TC BF16, WMMA FP16 | Sí | OK | `dd8e8b56` |
| 6884 | 2026-09-10 18:30 EDT | ídem | C=K=64, HW 64/128/256/512 | ídem + CUTLASS FP16/BF16 | Sí | OK | `3a4f2418` |

Ningún log contiene una ejecución `--double` (cuDNN FP64) de Convolución:
`run_conv_tc.sbatch` nunca la invocó.

### 1.3 Mediciones de Convolución con C=K=1024 (única fuente: job 4613, pasada de benchmark)

Se excluyen las pasadas bajo Nsight Compute (la segunda invocación de cada
tamaño), cuyo tiempo WMMA está perturbado por el perfilador.

| Job | Fecha | Entrada (N,C,H,W) ; filtro (K,C,R,S) | FMA_MATH | Ruta | Tiempo (ms) | TFLOP/s | Err. máx. abs. vs FP64 | L2 rel. vs FP64 | Línea del log |
| - | - | - | - | - | - | - | - | - | - |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 1024, 256, 256 ; 1024, 1024, 3, 3 | no | CPU FP32 | 10317.009687 | 0.119894 | 0.000346 | 0.000000 | 63 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 1024, 256, 256 ; 1024, 1024, 3, 3 | no | cuDNN FP32 (sin FMA_MATH) | 15.629005 | 79.144549 | 0.272615 | 0.000545 | 68 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 1024, 256, 256 ; 1024, 1024, 3, 3 | no | cuDNN TC FP16 | 8.290816 | 149.195281 | 0.337585 | 0.000574 | 76 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 1024, 256, 256 ; 1024, 1024, 3, 3 | no | WMMA FP16 | 42.800641 | 28.900282 | 0.270692 | 0.000538 | 85 |

El inventario completo por ruta (4613 con C=2048 y los jobs 6882/6884 con
C=K=64) está en el Anexo A.

## 2. Veredicto por celda de `tab:conv-mixta`

Comprobación de FLOP implícito (tiempo × rendimiento) frente a 1,23695e12:

| Fila | ms × TFLOP/s | Desviación |
| - | - | - |
| cuDNN (FP64) | 3,71077e12 | +199,99 % (= 2·12288³, un GEMM) |
| cuDNN clásico (FP32) | 1,21851e13 | +885,09 % |
| Tensor Core (FP16) | 1,23698e12 | +0,002 % |
| WMMA custom | 1,23695e12 | 0,000 % |

| Fila | Celda | Valor en la tabla | Veredicto | Origen / valor correcto |
| - | - | - | - | - |
| FP64 | Tiempo | 358,355 ms | **Errónea** | Es GEMM FP32 N=12288: job 4609 línea 52 (`GPU cuBLAS- tiempo medio : 358.3551432 ms`, corrida `fp32`) y job 4612 línea 64 (`GPU cuBLAS clasico - tiempo : 358.354309 ms`). No existe medición cuDNN FP64 de Convolución: **sin respaldo** hasta la re-ejecución |
| FP64 | TFLOP/s | 10,355 | **Errónea** | Job 4612 línea 65 (`10.355259 TFLOP/s`, cuBLAS FP32 N=12288) |
| FP64 | Speedup, errores | — | Correcta como referencia | — |
| FP64 | Tensor Cores | No (FP64) | **Sin respaldo** | No hay medición; si cuDNN FP64 supera 9,7 TFLOP/s habría que considerar DMMA |
| FP32 | Tiempo | 153,959 ms | **Errónea** | Job **4610** línea 55 (Fase 1, `GPU cuDNN - tiempo medio : 153.959424 ms`, `IMPLICIT_GEMM` forzado, 8,034 TFLOP/s), otro binario y otro algoritmo. El tiempo de la corrida de Fase 2 (4613 línea 68) es 15,629005 ms |
| FP32 | TFLOP/s | 79,145 | **Errónea como FP32 escalar** | Job 4613 línea 69 (`79.144549 TFLOP/s`): TF32 en Tensor Cores (`CUDNN_DEFAULT_MATH`, sin la línea `CUDNN_FMA_MATH`), 4 veces el pico FP32 escalar |
| FP32 | Err. máx. abs. | 0,272615 | **Errónea como FP32 escalar** | Job 4613 línea 71, mismo cómputo TF32 |
| FP32 | L2 rel. | 0,000545 | **Errónea como FP32 escalar** | Job 4613 línea 72, firma de TF32 (FP32 escalar da `0.000000` en 6882/6884) |
| FP32 | Speedup | 2,33× | **Errónea** | 358,355 / 153,959: cociente de dos valores erróneos |
| FP32 | Tensor Cores | No (FP32) | **Errónea para 4613** (usaba TF32); correcta para una corrida posterior al fix | — |
| FP16 (cuDNN TC) | Tiempo | 8,291 ms | **Confirmada** | Job 4613 línea 76 (`8.290816 ms`) |
| FP16 (cuDNN TC) | TFLOP/s | 149,195 | **Confirmada** | Job 4613 línea 77 (`149.195281`) |
| FP16 (cuDNN TC) | Err. máx. abs. | 0,337585 | **Confirmada** | Job 4613 línea 80 |
| FP16 (cuDNN TC) | L2 rel. | 0,000574 | **Confirmada** | Job 4613 línea 81 |
| FP16 (cuDNN TC) | Speedup | 43,23× | **Errónea** | 358,355 / 8,291: el denominador es correcto, el numerador no es FP64 de Convolución |
| FP16 (cuDNN TC) | Tensor Cores | Sí (HMMA) | **Sin respaldo directo** | El perfil NCU de 4613 solo captura `wmma_gemm_kernel`; 149 TFLOP/s > 19,5 lo implica, pero no es un perfil |
| WMMA | Etiqueta | (BF16) | **Errónea** | `wmma_gemm_kernel` usa fragmentos `__half` (`conv_tensor_activation.cu:1238`) e `im2col_fp16_kernel`; no existe ruta WMMA BF16 y 4613 ni siquiera corrió BF16. Correcta: **(FP16)** |
| WMMA | Tiempo | 42,801 ms | **Confirmada** | Job 4613 línea 85 (`42.800641 ms`) |
| WMMA | TFLOP/s | 28,900 | **Confirmada** | Job 4613 línea 86 (`28.900282`) |
| WMMA | Err. máx. abs. | 0,270692 | **Confirmada** | Job 4613 línea 89 |
| WMMA | L2 rel. | 0,000538 | **Confirmada** | Job 4613 línea 90 |
| WMMA | Speedup | 8,37× | **Errónea** | 358,355 / 42,801 |
| WMMA | Tensor Cores | Sí (HMMA) | **Confirmada (binario)** | 4613: `Instrucciones HMMA detectadas en binario: 4` y reporte NCU `ncu_wmma_conv_N1_C1024_..._job4613.ncu-rep` del kernel WMMA |

Texto asociado: el cociente cuDNN/WMMA `5,16` (42,801 / 8,291) es correcto
respecto a 4613; los `43,23×` y `18,57×` (= 153,959 / 8,291) de la Síntesis,
la Discusión y las Conclusiones son erróneos.

### 2.1 Hipótesis de la fila FP32 mezclada

**Confirmada en lo esencial y corregida en el detalle.** La fila sí mezcla el
tiempo de una corrida con el rendimiento y el error de otra, pero el tiempo
**no** viene de una corrida posterior al fix:

- 153,959 ms viene del job **4610**, línea base de **Fase 1**
  (`cudnn_conv_balanced.cu`, algoritmo `IMPLICIT_GEMM` forzado), ejecutado a
  las 18:34 EDT del 2026-08-05, también **antes** de `d6614ce`. Su rendimiento
  (8,034 TFLOP/s) es FP32 escalar, pero de un algoritmo que la heurística de
  Fase 2 no elige, y su error frente a la CPU (1,414) está contaminado por el
  bug de layout BLAS corregido en el mismo commit.
- 79,145 TFLOP/s, 0,272615 y 0,000545 vienen del job **4613** (Fase 2, antes
  del fix, TF32 activo), cuyo tiempo real fue 15,629 ms.
- No existe ningún log conservado de una corrida posterior al fix con
  C=K=1024 (4626 pudo serlo, pero su log se perdió).

## 3. Pre-registro de criterios de aceptación

Fijados en este commit, **antes** de cualquier `sbatch`. Los resultados se
evalúan solo contra estos umbrales; no se ajustan a posteriori.

**Job a lanzar** (uno solo, desde `Fase_2/Convolution` en PACCA):

```
sbatch --export=ALL,C=1024,K=1024,H=256,W=256,TC_FORMAT=both,RUN_CUTLASS=0,ITERS=10,RUN_DOUBLE=1 run_conv_tc.sbatch
```

`RUN_DOUBLE=1` (opción nueva, default 0) agrega, por cada HW, una invocación
`./conv_tc --double` con los mismos flags. N=1, R=S=3, padding/stride/dilation
1 son los defaults del sbatch.

**Reglas de procedencia (P0):**

- Todas las filas de la tabla salen del **mismo job**: FP32, FP16 y WMMA de la
  pasada de benchmark de la invocación FP32 (no de la pasada bajo Nsight
  Compute); FP64 de la invocación `--double` (`GPU cuDNN clasico` en el
  bloque `RESULTADOS CONV 2D FP64`).
- El bloque debe declarar `Entrada (N,C,H,W) : 1, 1024, 256, 256` y
  `Filtro (K,C,R,S) : 1024, 1024, 3, 3`, y la invocación FP32 debe contener la
  línea `math type CUDNN_FMA_MATH (TF32 desactivado)`.
- Redondeo para la tabla: tiempo y TFLOP/s a 3 decimales, errores a los 6
  decimales impresos, speedups `t_FP64 / t_fila` a 2 decimales.

**Umbrales:**

| ID | Criterio | Umbral | Regla de decisión |
| - | - | - | - |
| U1 | FLOP implícito (ms × TFLOP/s) en todas las filas | 1,23695e12 ± 0,1 % | Falla la fila que se salga |
| U2 | FP32 escalar (cuDNN `CUDNN_FMA_MATH`) | TFLOP/s ≤ 19,5 **y** L2 rel. vs FP64 ≤ 1e-6 | El log imprime 6 decimales: pasa solo con `0.000000` (< 5e-7); `0.000001` es indeterminado y cuenta como falla |
| U3 | cuDNN FP64 | TFLOP/s ≤ 19,5 | Si además supera 9,7 TFLOP/s se reporta como posible DMMA, **sin corregir** la celda ni la columna Tensor Cores |
| U4 | WMMA FP16 | L2 del mismo orden que cuDNN TC FP16 | Pasa si 0,1 ≤ L2_WMMA / L2_TC-FP16 ≤ 10 |
| U5 | cuDNN TC BF16 (solo coherencia de formatos; no va a la tabla) | L2_BF16 > L2_FP16 | Si falla, se reporta aquí y la tabla no se toca por ello |

**Qué pasa si algo falla:**

- Fila que falla U1–U4: no se escribe en la tabla; la celda conserva su valor
  actual y el comentario de C1 pasa a `% [C1-PENDIENTE]` con el motivo.
- Falla de infraestructura (compilación, biblioteca, OOM) sin resultados: se
  diagnostica, se corrige y se relanza el mismo comando; los umbrales no
  cambian y el relanzamiento se documenta en la sección 4.
- FP16 y WMMA de la nueva corrida reemplazan a los de 4613 aunque difieran
  poco, para que todos los cocientes provengan de una sola sesión.

## 3.1 Adenda al inventario (antes del resultado de 7785; umbrales sin cambios)

Revisión completa de `~/Documentos/Resultados_PACCA/` (carpetas `Fase_2/`,
`ncu_fase2_20260913/`, `campana_final_20260912/`, `campana_holder_20260917/`,
`superadas/`, `Fase_3/`, `Fase_4/` y el `Resultados_PACCA.zip`), buscando
`RESULTADOS CONV 2D`, `C=1024`, `1, 1024, 256, 256`, `K=1024` y los valores de
la tabla:

- `Resultados_PACCA/Fase_2/` solo contiene `GEMM_logs/` (jobs 6596-6599). De
  Convolución de Fase 2 solo hay dos reportes Nsight Compute
  (`ncu_fase2_20260913/Fase_2/Convolution/logs/`, jobs 6882 y 6884, C=K=64).
  **Ningún log de Fase 2 Convolución con C=K=1024 está en `Resultados_PACCA`.**
  El respaldo es del 2026-08-30 y para entonces `Fase_2/Convolution/logs/`
  ya no tenía los logs de agosto (su `INVENTARIO.md` no lista Convolución).
- La carpeta de "Fase 2" que sí contiene los valores de la tabla es
  `~/Documentos/Proyecto_de_Grado/Pruebas/Resultados_05_08_Fase_2/Fase_2/Convolution/mixed_precision_conv_tc_4613.out`
  (sección 1), fuera de `Resultados_PACCA`.
- `campana_holder_20260917/logs_holder_run/F1_Convolucion.log` es Fase 1 con
  C=K=64 (FP32 y FP64); no aplica a C=1024.

Evidencia nueva encontrada en PACCA (no está en `Resultados_PACCA`):

| Job | Fecha (log) | Binario | Configuración | Ruta | Tiempo (ms) | TFLOP/s | Algoritmo cuDNN | Err. máx. / L2 vs CPU | Línea |
| - | - | - | - | - | - | - | - | - | - |
| 7709 | 2026-09-27 14:29 EDT | Fase 1 (`fase1_conv_baseline`, nvcc 13.1), con `CUDNN_FMA_MATH` | N=1, C=K=1024, H=W=256, R=S=3, ITERS=10 | cuDNN FP32 | 41.282355 | 29.963179 | 6 (WINOGRAD), ws 100 MiB | 0.000267 / 0.000001 | 61-65 |
| 7709 | ídem | ídem | ídem | cuDNN FP64 | 164.147302 | 7.535613 | 1 (IMPLICIT_PRECOMP_GEMM) | 0.000000 / 0.000000 | 92-96 |

`Fase_1/Convolution/logs/fase1_conv_baseline_7709.out` en PACCA, sha256 `75c99565`.

Consecuencias, **sin modificar** los umbrales de la sección 3:

1. 7709 es una medición cuDNN FP64 real con C=K=1024, H=W=256 (7,54 TFLOP/s,
   por debajo de 9,7: sin indicio de DMMA). No cumple P0 (otro binario, otro
   job), así que no reemplaza a 7785 como fuente de la tabla; queda como
   corroboración independiente.
2. **Riesgo previsible para U2.** Con `CUDNN_FMA_MATH`, la heurística de cuDNN
   eligió Winograd para FP32. Winograd hace menos multiplicaciones que las que
   cuenta `conv_flops`, así que su rendimiento "efectivo" (29,96 TFLOP/s) supera
   el pico escalar de 19,5 sin usar Tensor Cores, y su error es mayor que el de
   un GEMM directo (L2 `0.000001` frente a la CPU FP32). Si el binario de Fase 2
   elige el mismo algoritmo en 7785, la fila FP32 **fallará U2 por construcción
   del umbral**, no por TF32. En ese caso se aplica la regla ya fijada (la fila no
   se escribe, `% [C1-PENDIENTE]`) y la decisión de redefinir U2 queda para los
   autores.

## 4. Resultados de la re-ejecución

Job **7785** (`sbatch --export=ALL,C=1024,K=1024,H=256,W=256,TC_FORMAT=both,RUN_CUTLASS=0,ITERS=10,RUN_DOUBLE=1 run_conv_tc.sbatch`),
COMPLETED, ExitCode 0:0, 2026-09-29 00:18:50 -> 00:25:22 EDT (6:36 min). Log
`Fase_2/Convolution/logs/mixed_precision_conv_tc_7785.out`, sha256
`a5360434`. Mismas reglas de procedencia que la sección 3: FP32/FP16/WMMA de
la pasada de benchmark (líneas 69-112, antes de `Perfilando`), FP64 de la
única invocación `--double` (líneas 141-149).

### 4.1 Evaluación contra los umbrales

| ID | Fila | Medición | Umbral | Resultado |
| - | - | - | - | - |
| U1 | FP64 | 164,094873 ms × 7,538021 TFLOP/s = 1,236951e12 | 1,23695e12 ± 0,1 % | **Pasa** (desviación 0,0000 %) |
| U1 | FP32 escalar | 39,949927 ms × 30,962524 TFLOP/s = 1,236951e12 | ídem | Pasa (informativo; la fila falla por U2) |
| U1 | TC FP16 | 5,846528 ms × 211,570117 TFLOP/s = 1,236951e12 | ídem | **Pasa** |
| U1 | WMMA | 25,889587 ms × 47,777918 TFLOP/s = 1,236951e12 | ídem | **Pasa** |
| U2 | FP32 escalar | 30,962524 TFLOP/s; L2 vs FP64 = 0,000001 | TFLOP/s ≤ 19,5 y L2 = 0,000000 | **Falla** (ambas condiciones) |
| U3 | FP64 | 7,538021 TFLOP/s | ≤ 19,5 (aviso DMMA si > 9,7) | **Pasa**, sin indicio de DMMA |
| U4 | WMMA FP16 | L2 = 0,000538; TC FP16 L2 = 0,000574; razón 0,937 | razón en [0,1 ; 10] | **Pasa** |
| U5 | TC BF16 (no va a la tabla) | L2 = 0,002862 > L2 FP16 = 0,000574 | BF16 > FP16 | **Pasa** |

**U2 falla, y no por TF32.** La línea `GPU cuDNN FP32 escalar : math type
CUDNN_FMA_MATH (TF32 desactivado)` está presente en el log (línea 75): TF32
está apagado. El job 7709 (Fase 1, mismo tamaño C=K=1024, H=W=256, corrido el
2026-09-27) imprime explícitamente `cuDNN algoritmo elegido : 6` para FP32, y
6 es `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` en la enumeración de
cuDNN. El binario de Fase 2 (`conv_tensor_activation.cu`) no imprime el
algoritmo elegido, pero su rendimiento (30,96 TFLOP/s) es consistente con el
mismo algoritmo Winograd de 7709 (29,96 TFLOP/s), y muy distinto del FP32
escalar verificado a C=K=64 en 6882/6884 (2-18 TFLOP/s, L2 = 0,000000).
Winograd reduce el conteo real de multiplicaciones respecto al conteo directo
que usa `conv_flops()`, así que su TFLOP/s "nominal" excede el pico escalar
sin que haya Tensor Cores involucrados, y su error (transformada Winograd,
no acumulación FP32 directa) es mayor que el de un GEMM directo. **El umbral
U2 fue diseñado para detectar TF32, no Winograd; aquí detecta un fenómeno
real pero distinto.** Esto excede lo que este reporte puede decidir por su
cuenta (ver sección 3, regla "falla de infraestructura" vs "hallazgo
numérico/de diseño": esto es lo segundo). Queda para los autores: (a) forzar
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` para un baseline FP32 escalar
directo y comparable a las rutas TC, o (b) aceptar Winograd como línea base
FP32 legítima y redefinir U2.

### 4.2 Valores para `tab:conv-mixta` (de este job)

| Fila | Tiempo (ms) | TFLOP/s | Speedup vs FP64 | Err. máx. abs. | L2 rel. |
| - | - | - | - | - | - |
| FP64 [Referencia] | 164,095 | 7,538 | --- | --- | --- |
| FP32 escalar | **sin escribir** (falla U2) | | pendiente | | |
| TC FP16 | 5,847 | 211,570 | 28,07× | 0,337585 | 0,000574 |
| WMMA (relabel **FP16**, no BF16) | 25,890 | 47,778 | 6,34× | 0,270692 | 0,000538 |

Cociente cuDNN TC / WMMA (párrafo posterior a la tabla): 25,889587 / 5,846528
= 4,43× (antes 5,16×, con los tiempos de 4613).

Nota de reproducibilidad: los errores de TC FP16 y WMMA son bit a bit
idénticos a los del job 4613 (kernel determinista); solo cambiaron tiempo y
TFLOP/s, consistente con el cambio de toolchain (nvcc 13.1 vs HPC SDK 23.1) y
no con un cambio de comportamiento numérico.

## 5. Estado final

- FP64, TC FP16 y WMMA (relabel FP16): **resueltos**, con log de origen 7785.
- FP32 escalar: **abierto**. No se escribe en la tabla; el comentario pasa a
  `% [C1-PENDIENTE]` con el motivo de la sección 4.1.
- Las propagaciones "43,23×" pasan a "28,07×" en los tres sitios. Las
  propagaciones "18,57×" (dependen de FP32 escalar) quedan marcadas como
  pendientes en los tres sitios, no se reemplazan por un número.

## Anexo A. Inventario completo por ruta (pasadas de benchmark)

| Job | Fecha | Entrada ; filtro | FMA_MATH | Ruta | Tiempo (ms) | TFLOP/s | Err. máx. abs. vs FP64 | L2 rel. vs FP64 | Línea |
| - | - | - | - | - | - | - | - | - | - |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 2048, 256, 256 ; 2048, 2048, 3, 3 | no | CPU FP32 | 23576.797131 | 0.209859 | 0.000554 | 0.000000 | 196 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 2048, 256, 256 ; 2048, 2048, 3, 3 | no | cuDNN FP32 (sin FMA_MATH) | 94.836530 | 52.171904 | 0.545794 | 0.000563 | 201 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 2048, 256, 256 ; 2048, 2048, 3, 3 | no | cuDNN TC FP16 | 31.730688 | 155.931137 | 0.675970 | 0.000579 | 209 |
| 4613 | Wed Aug  5 19:46:48 EDT 2026 | 1, 2048, 256, 256 ; 2048, 2048, 3, 3 | no | WMMA FP16 | 150.333032 | 32.912277 | 0.538775 | 0.000548 | 218 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | CPU FP32 | 9.475668 | 0.031870 | 0.000066 | 0.000000 | 69 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.045158 | 6.687347 | 0.000055 | 0.000000 | 75 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.051507 | 5.863062 | 0.069398 | 0.000486 | 83 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.050995 | 5.921928 | 0.391202 | 0.002878 | 92 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.072294 | 4.177224 | 0.040032 | 0.000436 | 101 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | CPU FP32 | 113.470744 | 0.010646 | 0.000045 | 0.000000 | 145 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.099328 | 12.161320 | 0.000051 | 0.000000 | 151 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.058982 | 20.479999 | 0.067098 | 0.000499 | 159 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.059290 | 20.373887 | 0.275201 | 0.002965 | 168 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.221696 | 5.448721 | 0.039223 | 0.000453 | 177 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | CPU FP32 | 1141.002821 | 0.004235 | 0.000049 | 0.000000 | 221 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.301363 | 16.033272 | 0.000050 | 0.000000 | 227 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.179507 | 26.917240 | 0.042548 | 0.000390 | 235 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.178995 | 26.994233 | 0.290399 | 0.002415 | 244 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.804352 | 6.007119 | 0.030898 | 0.000328 | 253 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | CPU FP32 | 5437.256850 | 0.003555 | 0.000058 | 0.000000 | 297 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 1.059328 | 18.244919 | 0.000049 | 0.000000 | 303 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.385843 | 50.091209 | 0.050699 | 0.000423 | 311 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.383590 | 50.385394 | 0.413601 | 0.004553 | 320 |
| 6882 | Thu Sep 10 18:06:54 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | WMMA FP16 | 3.145523 | 6.144400 | 0.028796 | 0.000366 | 329 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | CPU FP32 | 26.961304 | 0.011201 | 0.000066 | 0.000000 | 69 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.117248 | 2.575651 | 0.000055 | 0.000000 | 75 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.187699 | 1.608903 | 0.069398 | 0.000486 | 83 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.188006 | 1.606275 | 0.391202 | 0.002878 | 92 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.072294 | 4.177224 | 0.040032 | 0.000436 | 101 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | CUTLASS FP16 | 0.025293 | 11.939758 | 0.040016 | 0.000436 | 113 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 64, 64 ; 64, 64, 3, 3 | si | CUTLASS BF16 | 0.025395 | 11.891613 | 0.186182 | 0.002379 | 123 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | CPU FP32 | 167.497094 | 0.007212 | 0.000045 | 0.000000 | 165 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.099840 | 12.098954 | 0.000051 | 0.000000 | 171 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.059597 | 20.268866 | 0.067098 | 0.000499 | 179 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.059597 | 20.268866 | 0.275201 | 0.002965 | 188 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.222208 | 5.436166 | 0.039223 | 0.000453 | 197 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | CUTLASS FP16 | 0.041267 | 29.271662 | 0.039207 | 0.000454 | 209 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 128, 128 ; 64, 64, 3, 3 | si | CUTLASS BF16 | 0.041267 | 29.271662 | 0.190012 | 0.002466 | 219 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | CPU FP32 | 1119.655299 | 0.004315 | 0.000049 | 0.000000 | 261 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 0.300646 | 16.071499 | 0.000050 | 0.000000 | 267 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.181146 | 26.673782 | 0.042548 | 0.000390 | 275 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.180122 | 26.825423 | 0.290399 | 0.002415 | 284 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | WMMA FP16 | 0.803226 | 6.015543 | 0.030898 | 0.000328 | 293 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | CUTLASS FP16 | 0.091853 | 52.604146 | 0.030868 | 0.000328 | 305 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 256, 256 ; 64, 64, 3, 3 | si | CUTLASS BF16 | 0.091750 | 52.662856 | 0.169053 | 0.001767 | 315 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | CPU FP32 | 5345.038329 | 0.003616 | 0.000058 | 0.000000 | 357 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN FP32 (FMA_MATH) | 1.059738 | 18.237867 | 0.000049 | 0.000000 | 363 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN TC FP16 | 0.384922 | 50.211141 | 0.050699 | 0.000423 | 371 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | cuDNN TC BF16 | 0.328294 | 58.872016 | 0.413601 | 0.004553 | 380 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | WMMA FP16 | 3.144499 | 6.146401 | 0.028796 | 0.000366 | 389 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | CUTLASS FP16 | 0.274534 | 70.400479 | 0.028792 | 0.000366 | 401 |
| 6884 | Thu Sep 10 18:30:43 EDT 2026 | 1, 64, 512, 512 ; 64, 64, 3, 3 | si | CUTLASS BF16 | 0.274330 | 70.453037 | 0.283045 | 0.004228 | 411 |

Extraído con un parser sobre los `.out` (solo bloques que no están precedidos
por `Perfilando` dentro de la misma corrida). Las pasadas NCU de 4613, 6882 y
6884 se omiten.
