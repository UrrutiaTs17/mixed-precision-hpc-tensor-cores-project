# Auditoría C2: fila "GPU WMMA custom (BF16)" en `tab:gemm-mixta-12288`

Hallazgo C2 (crítico) de la revisión del informe final (28/09/2026). Diagnóstico de
código ya confirmado por el director (no se repite aquí): `wmma_gemm_kernel<T>` es un
template, `T=__half` y `T=__nv_bfloat16` son dos kernels distintos que comparten el
nombre demangled base `wmma_gemm_kernel`.

- Rama: `auditoria-c2-gemm-fase2` (basada en `main`, independiente de la auditoría C1).
- **Ningún job se relanzó.** Todo el análisis usa resultados ya descargados localmente.
- Herramienta usada: `ncu` local (`/opt/nvidia/nsight-compute/2026.3.1/ncu`), disponible
  en esta máquina; se usó solo para **leer** (`--import`) los `.ncu-rep` ya descargados,
  no para perfilar nada nuevo.

## 1. Inventario de evidencia (Fase_2/GEMM, N=12288 y controles)

| Archivo | Ubicación local | Fecha | M=N=K | `TC_FORMAT` en el log | Kernel WMMA (nombre NCU) |
| - | - | - | - | - | - |
| `mixed_precision_gemm_tc_4612.out` | `Pruebas/Resultados_05_08_Fase_2/Fase_2/GEMM/` | 2026-08-05 18:38 | 12288 y 32768 | **no existe la variable** (sbatch pre-`--tc-format`) | — (ver ncu-rep) |
| `ncu_wmma_m12288_n12288_k12288_job4612.ncu-rep` | ídem | ídem | 12288 | — | **`wmma_gemm_kernel<__half>`** |
| `ncu_wmma_m32768_n32768_k32768_job4612.ncu-rep` | ídem | ídem | 32768 | — | no inspeccionado (tab:gemm-mixta-32768 no está bajo sospecha) |
| `mixed_precision_gemm_tc_6598.out` | `Resultados_PACCA/Fase_2/GEMM_logs/logs/` | 2026-08-26 23:29 | 12288 y 32768 | **no existe la variable** (mismo sbatch pre-`--tc-format`) | — |
| `ncu_wmma_m12288_n12288_k12288_job6598.ncu-rep` | ídem (y copia en `ncu_fase2_20260913/Fase_2/GEMM/logs/`) | ídem | 12288 | — | **`wmma_gemm_kernel<__half>`** |
| `mixed_precision_gemm_tc_6599.out` | ídem | 2026-08-26 23:50 | 1024 | **no existe la variable** | — |
| `ncu_wmma_m1024_n1024_k1024_job6599.ncu-rep` | ídem | ídem | 1024 | — | no inspeccionado (tamaño de control, no aparece en ninguna tabla) |
| `ncu_wmma_m{1024,2048,4096,8192}_..._job6881.ncu-rep` | `Resultados_PACCA/ncu_fase2_20260913/Fase_2/GEMM/logs/` | (sin `.out` descargado) | 1024–8192 | indeterminado (falta el `.out`) | `wmma_gemm_kernel<__half>` en m=1024 (único inspeccionado) |
| `ncu_wmma_m{1024,2048,4096,8192}_..._job6883.ncu-rep` | ídem | (sin `.out` descargado) | 1024–8192 | indeterminado | `wmma_gemm_kernel<__half>` en m=1024 (único inspeccionado) |

**No existe ningún `.out` ni `.ncu-rep` descargado localmente para GEMM con `TC_FORMAT`
real (`fp16`/`bf16`/`both`) a M=N=K=12288.** Los dos únicos logs con esa configuración
(4612 y 6598) son ambos anteriores a que `run_gemm_tc.sbatch` tuviera la variable
`TC_FORMAT` (confirmado por su ausencia total en el `.out`: ni se exporta ni se imprime,
a diferencia del sbatch actual, que hace `echo "TC_FORMAT=${TC_FORMAT}"` — ver
`Fase_2/GEMM/run_gemm_tc.sbatch:268`). El binario de esa fecha corría la ruta WMMA
únicamente en FP16 por defecto, sin alternativa BF16 invocable (`Fase_2/GEMM/README.md`,
sección "BF16", ya documenta este vacío histórico).

## 2. Confirmación de la hipótesis principal

**Confirmada con evidencia directa, más fuerte que la hipótesis de skip/warmup
propuesta en el contexto.** No hizo falta invocar el mecanismo de
`NCU_LAUNCH_SKIP=3`/`NCU_LAUNCH_COUNT=1` cayendo en los warmups de FP16: el nombre de
kernel que Nsight Compute registra ya incluye el parámetro de template resuelto.
Importando `ncu_wmma_m12288_n12288_k12288_job4612.ncu-rep` con
`ncu --import ... --print-summary per-kernel`:

```
void <unnamed>::wmma_gemm_kernel<__half>(const T1 *, const T1 *, float *, int, int, int)
    (192, 192, 1)x(512, 1, 1), Device 0, CC 8.0, Invocations 1
    sm__inst_executed_pipe_tensor_op_hmma.sum ......... 905.969.664
    sm__ops_path_tensor_src_fp16_dst_fp32.sum .......... 3.710.851.743.744
    sm__ops_path_tensor_src_bf16_dst_fp32.sum .......... 0
    sm__pipe_tensor_cycles_active.avg.pct_of_peak ...... 22,50 %
    sm__throughput.avg.pct_of_peak_sustained_elapsed ... 64,12 %
    gpu__dram_throughput.avg.pct_of_peak_sustained ..... 39,15 %
    dram__bytes_read.sum ................................ 58,75 GB
    dram__bytes_write.sum .............................. 605,64 MB
    launch__registers_per_thread ....................... 34
```

**Estos valores son exactamente los de `tab:ncu-tensor` en `main.tex`** (905 969 664
HMMA; 3,71×10¹² fp16→fp32 al 100 %; 0 bf16→fp32; 64,12 % SM; 74,77 % *warps* activos;
39,15 % DRAM; 58,75 GB leídos; 605,64 MB escritos; 34 registros/hilo): confirma que
`tab:ncu-tensor` se construyó sobre este mismo `.ncu-rep`, y que el kernel perfilado es
literalmente `<__half>`, no `<__nv_bfloat16>`.

Se repitió el mismo `--print-summary per-kernel` sobre `job6598` (mismo tamaño,
corrida del 26/08) y sobre `job6881`/`job6883` a m=1024 (los únicos tamaños de ese par
inspeccionados): **las cuatro corridas devuelven `wmma_gemm_kernel<__half>`, ninguna
`<__nv_bfloat16>`.** Cero perfiles de BF16 existen en el conjunto de `.ncu-rep`
descargados para GEMM, a ningún tamaño.

**Veredicto:** la fila "WMMA custom (BF16)" de `tab:gemm-mixta-12288` son los datos
FP16 con etiqueta incorrecta. BF16 para GEMM WMMA **nunca se ejecutó** a N=12288 en
ningún resultado descargado — no por un problema de captura (skip/warmup), sino porque
el binario/sbatch de esa fecha no tenía ruta BF16 invocable en absoluto. No hay
corrupción de datos ni aliasing de buffers.

## 3. Mecanismo para `tab:ncu-tensor`

Confirmado directamente (sección 2), sin necesidad de invocar la hipótesis de
`NCU_LAUNCH_SKIP`/warmup: el job 4612 es anterior a `--tc-format`, así que no había
"orden FP16 antes que BF16" que perfilar por accidente — simplemente no existía una
invocación BF16 en ese binario. La hipótesis de skip/warmup sigue siendo plausible como
explicación general para corridas *posteriores* que sí usan `TC_FORMAT=both` (jobs 6881/6883
parecen ser un par de esa naturaleza, dado que ambos con m=1024 dan también `<__half>`),
pero **no se puede confirmar para 6881/6883 específicamente**: no hay `.out` descargado
que declare su `TC_FORMAT`. Se deja constancia de esto como indeterminado, sin inferirlo.

## 4. Corrección aplicada al `.tex`

- `tab:gemm-mixta-12288`: fila reetiquetada "GPU WMMA custom (BF16)" → "GPU WMMA custom
  (FP16)". Ningún valor numérico cambia (los datos siempre fueron de la ruta FP16).
- Párrafo previo a la tabla: "WMMA personalizada en BF16" → "WMMA personalizada en FP16".
- `subsec:res-ncu`: "sobre el kernel `wmma_gemm_kernel`" → "sobre la instanciación FP16
  del kernel `wmma_gemm_kernel`", para no dar a entender que se caracterizó el kernel de
  forma genérica (aplica igual a BF16).
- No se agregó fila BF16 a ninguna tabla: no hay datos propios que la respalden (sección 1).
- El comentario `% [REVISIÓN - CRÍTICO]` se reemplazó por `% [C2-RESUELTO]` con el detalle
  de esta auditoría. La última frase de ese comentario ("Además, la columna 'Tensor
  Cores' marca 'No (FP64)'...", que corresponde a C10) **se dejó intacta, verbatim**, tal
  como exigía el alcance de esta tarea.
- `tab:gemm-mixta-32768` no se tocó: ya etiqueta "(FP16)" correctamente en ambas rutas.

## 5. Estado final

- Hallazgo C2: **resuelto** con evidencia directa (nombre de kernel en NCU), sin
  necesidad de datos nuevos.
- Ningún job relanzado; no fue necesario.
- Cada cifra corregida es rastreable: `.tex` → este reporte → `job4612` /
  `ncu_wmma_m12288_n12288_k12288_job4612.ncu-rep` (rutas locales en la sección 1).
- Compilación verificada: pdflatex+bibtex+2×pdflatex, 0 errores nuevos, 89 páginas,
  sin referencias `??`.
