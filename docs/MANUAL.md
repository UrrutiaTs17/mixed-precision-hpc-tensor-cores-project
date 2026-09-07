# Manual del proyecto

Guía completa para alguien que se une al proyecto sin haber visto el código antes: qué es cada archivo, qué ejecuta, qué datos produce, cómo se analizan, y cómo correr tanto una prueba individual como una campaña completa.

**Estado de este manual**: el proyecto se está reconstruyendo siguiendo el documento *Plan de Precisión Mixta*. Fase 1, Fase 2, Fase 3 y Fase 4 tienen código para los tres kernels (GEMM, Convolución, Stencil), incluyendo la estadística inferencial y el Frente de Pareto 3D (`Fase_4/tools/run_statistics.py`, `Fase_4/tools/pareto_front.py`) y dos orquestadores de principio a fin (`run_full_pipeline.sh` para una máquina propia, `run_full_pipeline_pacca.sh` para un clúster con SLURM). **Ninguna campaña real se ha corrido todavía.**

Lo que **sí** está verificado contra hardware real (GPU Ampere `sm_86`, 2026-09-06), y que hasta la auditoría de esa fecha no existía:

- La **verificación del orden de operandos** de `cublasDgemm` en GEMM y Convolución (`Fase_3/tools/verificar_orden_operandos_*.py`) — el "punto de mayor riesgo de error silencioso" que los propios `.cu` señalaban. **Pasa en los dos kernels.** En Convolución cubre además la indexación del `im2col`, el padding "SAME" y la estructura bloque-diagonal del filtro.
- Los **gates K=0/K=1 del ancla FP64** (`Fase_4/tools/gate3_ancla.py` + un `gate3_ancla.sbatch` por kernel). Probados positiva y negativamente para GEMM y Convolución; Stencil se probará en PACCA (su `.cu` rechaza Windows a propósito por un `static_assert` de 64 bits).
- La columna **`anchor_every`** ya es una columna real del CSV en los seis binarios, no algo reconstruido desde una línea de texto.
- La **medición de tiempo y energía por ruta** en GEMM y Convolución, que hasta esa auditoría no distinguía una ruta de otra ni excluía el costo de la referencia FP64 (ver la advertencia de la sección 7). Corregida y vigilada por `Fase_4/tools/gate4_medicion.py`. Los dos kernels publican ahora la ruta `GPU_FP64` como punto de Pareto propio, igual que Stencil.

Los scripts de Python de análisis siguen probados solo contra datos sintéticos (ver `Fase_4/tools/README.md`). Donde una pieza todavía no existe, este manual lo dice explícitamente en vez de describir algo que no vas a encontrar en el repositorio. Si encuentras una sección desactualizada, es más confiable el `README.md` de la carpeta específica que este manual — actualízalo si notas la diferencia.

---

## 1. Antes de empezar

### 1.1 Qué necesitas

Lee `REQUIREMENTS.md` en la raíz del repositorio para el detalle completo. En resumen: la compilación y ejecución de CUDA corre en **cualquier máquina con GPU NVIDIA Ampere o más nueva** (`sm_80+` — ver REQUIREMENTS.md sobre por qué ese piso, no "cualquier GPU con Tensor Cores") y el entorno conda de `environment.yml` activo. PACCA es una opción de ejecución (vía SLURM, `sbatch`), no un requisito — cada `.sbatch` corre igual con `bash archivo.sbatch` directo, sin SLURM, en tu propia máquina/servidor.

### 1.2 Cómo está organizado el proyecto

```
common/      → código compartido por los tres kernels y las cuatro fases
Fase_1/      → línea base FP64/FP32, sin Tensor Cores
Fase_2/      → Tensor Cores activados (FP16/BF16), sin encadenar iteraciones
Fase_3/      → encadenamiento genuino + drift + compensación
Fase_4/      → ancla FP64 + energía completa + Pareto 3D + estadística
tools/       → utilidades de perfilamiento compartidas
docs/        → este manual
old/         → código anterior a la reconstrucción, de referencia
```

Cada fase tiene tres subcarpetas: `GEMM/`, `Convolution/`, `Stencil/` — un kernel HPC distinto cada una. Dentro de cada una hay un `README.md` con el detalle específico de ese kernel en esa fase; este manual da la vista general y el flujo de trabajo, los README de cada carpeta dan el detalle de flags y columnas de CSV.

### 1.3 El vocabulario mínimo para no perderte

- **Kernel**: en este proyecto, no es "el núcleo del sistema operativo" — es una rutina de cómputo que corre en la GPU (GEMM, Convolución, Stencil).
- **Ruta** (*route*): una de las variantes de un kernel según qué biblioteca/precisión usa (p. ej. "cuBLAS FP32", "WMMA propio FP16 con compensación espacial").
- **Ground truth**: el resultado en FP64, usado como referencia contra la que se mide el error de todas las demás rutas.
- **Drift**: cuánto se aleja una ruta de baja precisión del ground truth a medida que avanzan las iteraciones de un esquema encadenado (solo aplica a esquemas iterativos, ver Fase 3).
- **Horizonte de overflow**: la iteración en la que una ruta deja de ser numéricamente finita (`inf`/`NaN`). Se distingue "medido" (exacto, detectado en tiempo de ejecución) de "predicho" (estimado por regresión).
- **EDP** (*Energy-Delay Product*): energía consumida × tiempo de ejecución. Métrica que castiga tanto lo lento como lo derrochador.
- **Ancla FP64**: mecanismo (Fase 4) que recalcula un paso completo en FP64 cada `K` iteraciones para frenar el drift, sin correr todo en FP64. Ver la sección 5 de este manual y el documento *Plan de Precisión Mixta* para el detalle completo.

---

## 2. Fase 1 — Línea base FP64/FP32

**Objetivo**: establecer cuánto tardan y qué throughput dan GEMM, Convolución y Stencil en punto flotante estándar, sin Tensor Cores. Es la referencia de "cómo sería este cómputo sin precisión mixta".

| Kernel | Binario | Librería | Ruta CPU |
|---|---|---|---|
| GEMM | `Fase_1/GEMM/gemm_baseline.cu` | cuBLAS (Sgemm/Dgemm) | OpenBLAS |
| Convolución | `Fase_1/Convolution/conv_baseline.cu` | cuDNN (con TF32 desactivado explícitamente) | im2col + OpenBLAS |
| Stencil | `Fase_1/Stencil/stencil_baseline.cu` | kernel CUDA propio (no aplica librería) | CPU serial |

### Cómo correr una prueba individual

Cada binario se compila con el `.sbatch` de su carpeta (`sbatch run_gemm_fase1.sbatch`, etc.), que a su vez compila con `nvcc` y lanza el binario con los parámetros que le pases por variable de entorno. Ejemplo, para correr GEMM en un solo tamaño en vez del barrido por defecto:

```bash
sbatch --export=ALL,SIZES="1024" run_gemm_fase1.sbatch
```

Consulta el `README.md` de cada carpeta para la lista completa de variables aceptadas — todas tienen un valor por defecto documentado, así que `sbatch run_gemm_fase1.sbatch` sin argumentos ya corre algo razonable.

### Qué datos obtienes

Cada corrida imprime a `stdout` (redirigido por SLURM al log del job) líneas con tiempo (ms), throughput (GFLOPS/TFLOPS) y error contra la referencia de mayor precisión disponible en esa fase. Fase 1 no tiene post-procesamiento automático a CSV todavía — lee el log del job directamente, o revisa si el `README.md` de la carpeta específica ya documenta un formato `CSV_*` (si Fase 3/4 ya se construyeron para ese kernel, es probable que sí).

---

## 3. Fase 2 — Activación de Tensor Cores

**Objetivo**: activar Tensor Cores explícitamente (FP16/BF16) en los tres kernels, y verificar que la activación es real (no solo que el binario "debería" usarlos) — cada `.sbatch` de esta fase cuenta instrucciones HMMA en el binario compilado antes de perfilar, y aborta si no encuentra ninguna.

| Kernel | Rutas | Nota |
|---|---|---|
| GEMM | CPU, cuBLAS sin TC, cuBLAS con TC (`cublasGemmEx`), WMMA propio | 4 rutas — ver `Fase_2/GEMM/README.md`, sección "comparaciones justas" |
| Convolución | CPU, cuDNN sin TC, cuDNN con TC, WMMA propio (im2col en GPU) | 4 rutas — misma advertencia de comparaciones justas |
| Stencil | GPU FP32/FP64, WMMA propio | Solo 2 familias — Stencil no usa cuBLAS/cuDNN, así que no hay brecha librería-vs-WMMA que aclarar aquí |

### La advertencia más importante de esta fase: comparaciones justas

Para GEMM y Convolución, **nunca reportes un solo número de "aceleración por Tensor Cores"** mezclando la ruta de librería (cuBLAS/cuDNN con TC) con la ruta WMMA propia. Son comparaciones distintas:

- **Librería FP32 vs. librería con TC** → efecto puro de precisión, misma calidad de implementación en ambos lados. Esta es la comparación que responde "¿ayudan los Tensor Cores?".
- **Librería FP32 vs. WMMA propio** → mide además la brecha de calidad de implementación (un kernel de estudiante nunca va a igualar años de tuning de NVIDIA). Es una comparación válida y útil, pero responde una pregunta distinta ("¿qué tan cerca llega nuestro propio kernel a la librería?"), no "cuánto aceleran los Tensor Cores".

El `README.md` de `Fase_2/GEMM/` y `Fase_2/Convolution/` desarrolla esto con más detalle — léelo antes de escribir cualquier resultado de esta fase en el documento de tesis.

### Cómo correr

Igual que Fase 1: `sbatch run_gemm_tc.sbatch` (o `run_conv_tc.sbatch`, `run_stencil_tc.sbatch`), con variables de entorno para tamaño, formato (`fp16`/`bf16`/`both`) e iteraciones. El default de formato es `both` — corre FP16 y BF16 en la misma invocación, no hace falta lanzar dos jobs.

---

## 4. Fase 3 — Encadenamiento y drift

**Objetivo**: dejar de medir cada kernel como una llamada suelta y encadenarlo de verdad (salida de la iteración *n* → entrada de la iteración *n+1*), para poder medir cómo se degrada la exactitud numérica a lo largo del tiempo — no solo el error de un cómputo aislado.

### Stencil

`Fase_3/Stencil/stencil_tensor_activation.cu` — el kernel más grande y más importante del proyecto. Aplica repetidamente un operador de diferencias finitas (Laplaciano de 5 puntos, formulado internamente como producto de matrices `Y = X·H + V·X` para poder usar Tensor Cores) sobre una grilla, midiendo:

- **Drift acumulado** (norma L2 y L∞ relativas) contra una referencia FP64 que se calcula en paralelo.
- **Horizonte de overflow**: la iteración en la que cada formato deja de ser finito, medido exactamente y también estimado por regresión (el código se niega a imprimir la estimación si no hay suficientes puntos de ajuste — si ves "N/D" en esa columna, es honestidad del código, no un bug).
- **Energía** (NVML/RAPL) por ventana, con un umbral de confiabilidad explícito — ver `common/README.md` sobre `power_sampling.h`.

Tres políticas de compensación disponibles vía `--kahan`/`--spatial-comp`:

| Política | Qué hace | Resultado |
|---|---|---|
| `none` | Sin compensación | Baseline |
| `kahan_local` (`--kahan on`) | Suma de Kahan clásica, por celda | **Depende del operador** (`--op-mode`): bajo el operador de estrés (el que usa la campaña principal) es indistinguible de `none`, incluso algo peor (+6% a +9%) — el residuo de una celda nunca lo relee ninguna de sus 4 vecinas, que son las que sufren el error real. Bajo un operador difusivo sí ayuda (-24% a -26%), porque ahí el residuo queda correlacionado temporalmente, que es lo que Kahan necesita para funcionar. |
| `spatial` (`--spatial-comp on`) | Compensación que explota la linealidad del operador | Reduce el error 5-6 órdenes de magnitud bajo el operador de estrés, con menor costo que `kahan_local` — y no depende de la correlación temporal del residuo, así que es la opción robusta en ambos modos de operador. |

Si no sabes cuál usar: `spatial` es la opción robusta. `kahan_local` se conserva por completitud experimental — bajo el operador de estrés, el hallazgo de que casi no ayuda es en sí mismo parte del resultado del proyecto, no lo borres pensando que es un error. Ver `Fase_3/Stencil/README.md` para el detalle numérico por modo de operador.

### GEMM y Convolución

Construido (`Fase_3/GEMM/gemm_chained.cu`, `Fase_3/Convolution/conv_chained.cu`) — mismo problema que Fase 3 ya resolvió para Stencil (el código anterior medía GEMM y Convolución como llamadas sueltas, ver Fase 2, lo cual no representa cómo se usan en HPC clásico: solucionadores iterativos, iteración de potencias, capas apiladas), resuelto siguiendo la sección 02 del documento **Plan de Precisión Mixta**. **Nada de esto se ha compilado ni ejecutado todavía** (este entorno de desarrollo no tiene GPU ni `nvcc`) — antes de confiar en cualquier resultado, corre el smoke test que documenta cada README.

- **GEMM**: `X(n+1) = X(n) · A`, con `A = c·H` un operador fijo construido a partir de una matriz de Hadamard de Sylvester (entradas exactamente `±1`, sin error de redondeo del operador en ningún formato — a diferencia de una matriz ortogonal genérica). Analogía directa con la iteración de potencias.
- **Convolución**: `X(n+1) = conv(X(n), W)`, con `W` el mismo Laplaciano de 5 puntos de Stencil expresado como filtro 3×3, replicado **bloque-diagonal** en 64 canales (`C=K=64`, cada canal evoluciona independiente — no es un capricho: con `C=K=1` el GEMM subyacente del `im2col` desperdiciaría 63 de cada 64 filas del tile WMMA de `common/wmma_gemm.cuh`, dando cifras de rendimiento engañosas).

Ambos usan compensación por linealidad (`--comp on`, `comp` en `float`, sin ancla — eso es Fase 4) y una referencia FP64 encadenada vía cuBLAS. **El punto de mayor riesgo silencioso de los dos archivos**: cuBLAS es *column-major*, el resto del archivo es *row-major* — la llamada a `cublasDgemm` invierte el orden "natural" de los operandos para que ambas convenciones calculen la misma operación matemática sobre el mismo buffer. Si estuviera al revés, el binario compilaría y correría igual, comparando peras con manzanas sin ningún error visible.

**Esa verificación ya existe y ya pasó** (GPU Ampere real, 2026-09-06):

```bash
python3 Fase_3/tools/verificar_orden_operandos_gemm.py --n 32
python3 Fase_3/tools/verificar_orden_operandos_conv.py --hw 64
```

Los dos scripts **extraen** del `.cu` el texto de las funciones bajo prueba —no las reimplementan, para verificar el código que corre en la campaña y no una copia—, lo compilan en un binario mínimo que expone el resultado crudo, y lo comparan contra NumPy. El de Convolución cubre además la indexación del `im2col` (con un filtro asimétrico, la única forma de detectar una transposición `r↔s` o un filtro volteado: el Laplaciano es simétrico y los oculta), el padding "SAME" —reportando el error del borde por separado— y la estructura bloque-diagonal del filtro. Ver la derivación completa junto a `gpu_fp64_step()`/`gpu_fp64_conv_step()` en cada `.cu`, y `Fase_3/tools/README.md`.

---

## 5. Fase 4 — Ancla FP64, energía completa y Frente de Pareto

**Objetivo**: cerrar la brecha hacia FP64 con el mecanismo de ancla, ampliar la telemetría de energía a los tres kernels, y sintetizar todo en un Frente de Pareto 3D (Tiempo, Energía, Error) por kernel, con el respaldo estadístico (t-test/ANOVA) que exige el objetivo 3 del plan de tesis.

**Estado: el ancla FP64 está construida en los tres kernels** (`Fase_4/Stencil/stencil_tensor_activation.cu`, `Fase_4/GEMM/gemm_chained.cu`, `Fase_4/Convolution/conv_chained.cu`), cada uno con su `.sbatch` y post-proceso de CSV, y la estadística (ANOVA + Tukey HSD) y el Frente de Pareto 3D también (`Fase_4/tools/run_statistics.py`, `Fase_4/tools/pareto_front.py` — Etapas 7 y 9 del plan). **Ninguna campaña real se ha corrido todavía**; los dos scripts de análisis sí se ejecutaron contra datos sintéticos, pero eso valida que el código corre, no que las conclusiones tengan sentido.

Los **gates K=0/K=1 sí están automatizados** y hay que correrlos antes de cualquier campaña con `--anchor-every > 0`:

```bash
cd Fase_4/GEMM && sbatch gate3_ancla.sbatch      # ídem Convolution/ y Stencil/
```

Cada uno compila los dos binarios (Fase 3 y Fase 4), corre las tres pasadas necesarias y le pasa los logs a `Fase_4/tools/gate3_ancla.py`, que decide y sale con `0`/`1`/`2` (`2` = no evaluable, que **no** es un "pasa"). `tools/validacion_preliminar.sbatch` los corre para los tres kernels de un tirón.

### El mecanismo de ancla, en corto

Cada `K` iteraciones (parámetro `--anchor-every K`), el paso se recalcula completo en FP64 en vez de en baja precisión, y el resultado reemplaza al de la ruta rápida. Corrige el error de *ese* paso — no reconstruye el drift ya acumulado antes del ancla, porque eso costaría lo mismo que correr todo en FP64. El costo extra es proporcional a `1/K`. `K=0` (deshabilitado) debe ser bit-idéntico al comportamiento de Fase 3.

**`K=1` (ancla en cada iteración): cuidado con el criterio ingenuo.** El *estado interno* del mecanismo sí converge exactamente a la trayectoria FP64 — con K=1 se puede demostrar que `dequant(T) + comp64 == out64` bit a bit. Pero lo que `CSV_DRIFT` publica **no es ese estado**: en GEMM y Convolución compara contra el buffer `T` tal cual se guarda, ya cuantizado a 16 bits. Así que el piso que ese CSV puede reportar está acotado por la cuantización del formato, no por la exactitud del ancla, y pedir "`rel_l2` a nivel de ruido de punto flotante (~`1e-16`)" es imposible por construcción.

El criterio correcto es una cota derivada: para redondeo al más cercano con `p` bits de significando, `rel_l2 ≤ 2^-p` y `rel_linf ≤ 2^-p`, con `p=11` en FP16 (`4.883e-04`) y `p=8` en BF16 (`3.906e-03`). Medido en GPU real, los dos formatos caen al **0.77** de su cota respectiva — el mismo factor en ambos, que es la confirmación de que el modelo es el correcto.

En Stencil el gate es distinto porque `CSV_DRIFT` mide otro objeto (el acumulador FP32 *antes* del redondeo de almacenamiento, no el buffer de 16 bits): allí se comprueba (a) que la ruta anclada alcance el nivel de `GPU_FP64` de la misma corrida y (b) que `rel_l2_prop`/`rel_linf_prop` —que sí miden el estado propagado— cumplan la misma cota `2^-p`. Todo esto está implementado y explicado en `Fase_4/tools/gate3_ancla.py`.

**No hay un motor compartido entre los tres kernels.** Se consideró escribirlo como plantilla genérica (`common/chained_precision.cuh`, todavía existe en el repo) pero terminó sin usarse: cada uno de los tres `.cu` de Fase 4 implementa sus propios kernels locales de reconstrucción/reseed (mismo nombre, misma lógica, repetidos tres veces) porque pasar los conversores `float↔T` como funtores de template entre traducciones agregaba complejidad de compilación sin ganancia real — la orquestación del bucle (qué buffers viven, en qué orden se hace swap) de todas formas es específica de cada kernel. Ver `common/README.md`, sección `chained_precision.cuh`, para el detalle de esta decisión.

### Frente de Pareto: uno por kernel, no uno combinado

GEMM, Convolución y Stencil resuelven problemas distintos — no tiene sentido "elegir el kernel que minimiza el EDP" (nadie elige entre correr una convolución o un stencil según cuál gasta menos energía; usa el que su problema necesita). El Frente de Pareto se calcula **por separado para cada kernel**, sobre el espacio (tiempo, energía, error) de sus propias rutas — la salida es una directriz distinta por kernel: "para esta tolerancia de error, en Stencil conviene BF16+spatial+ancla(K=8); en GEMM conviene FP16 con Tensor Cores de librería", etc.

### Estadística: qué correr y sobre qué datos

t-test/ANOVA factorial sobre los datos de variabilidad (réplicas por celda del diseño: formato × tratamiento × tamaño). Dos detalles de diseño que no son opcionales:

1. **Aleatorizar o intercalar el orden de envío de las réplicas** entre celdas al lanzarlas al clúster — si el aislamiento térmico no se logra (`nvidia-smi -lgc` puede estar denegado en PACCA), la deriva térmica entre trabajos consecutivos queda confundida con la celda del diseño si se envían en bloque, violando independencia.
2. **Tratar `K` (anchor_every) como variable ordinal**, no como categoría plana — la pregunta es cómo cambia el resultado a medida que `K` crece, no solo si los niveles difieren entre sí.

---

## 6. Cómo lanzar una campaña completa

Una "campaña" es un barrido sistemático (todos los tamaños × todos los formatos × todos los tratamientos × réplicas) pensado para producir el dataset final, no una prueba de humo. El flujo general:

1. **Correr la validación preliminar**, un job corto y barato que hace todo lo que hay que confirmar antes de comprometer horas de cola:

   ```bash
   sbatch tools/validacion_preliminar.sbatch     # o: bash tools/validacion_preliminar.sbatch
   ```

   Verifica, de lo más barato a lo más caro: (a) el **orden de operandos** de `cublasDgemm` en GEMM y Convolución —y de paso, en Convolución, la indexación del `im2col`, el padding "SAME" y el filtro bloque-diagonal—; (b) una **prueba de humo** de los tres kernels, que confirma que compilan, corren y emiten los CSV con `anchor_every` variando por ruta; (c) los **gates K=0/K=1** del ancla en los tres kernels; y (d) el **gate de medición** en GEMM y Convolución, que confirma que `t_iter_ms` y `energy_gpu_j` distinguen una ruta de otra y que la referencia FP64 se reporta aparte. Sale con código `0` solo si todo pasa, y corre todos los pasos aunque uno falle, para que un solo job diga todo lo que hay que arreglar.

2. **Lanzar el barrido completo**. En un clúster con SLURM, lo recomendado es:

   ```bash
   bash run_full_pipeline_pacca.sh          # DRY_RUN=1 imprime el grafo sin enviar
   ```

   Envía cada fase como un job independiente con `sbatch --parsable` y las encadena con `--dependency=afterok`: la validación preliminar primero, las campañas colgando de ella, Fase 4 de cada kernel dependiendo solo de su propia Fase 3 (los tres kernels en paralelo), y el post-proceso dependiendo de los seis a la vez. Ver la sección 6.1.

   Fuera de un clúster, `run_full_pipeline.sh` corre las mismas fases con `bash` en secuencia. **Su default es la campaña completa, no una prueba de humo** — para eso está `PIPELINE_MODE=smoke`. También se puede lanzar cada `.sbatch` por separado, con las variables documentadas en su README (`N_LIST`/`HW_LIST`/`NX_LIST`/`NY_LIST`, `ITERS_LIST`, `TC_FORMAT`, `COMP_LIST`, `ANCHOR_LIST`); todas aceptan `SMOKE_TEST=1`.

3. **Extraer los CSV**: `extract_csv.py` para Stencil, `extract_csv_chained.py` para GEMM/Convolución (esquema de columnas distinto — ver `Fase_3/tools/README.md`). Los `.sbatch` ya lo invocan automáticamente al terminar.

4. **Correr el análisis**: `python3 Fase_4/tools/run_statistics.py --results-dir results/` (ANOVA + Tukey HSD, Etapa 7) y `python3 Fase_4/tools/pareto_front.py --results-dir results/` (Frente de Pareto 3D, Etapa 9). En el flujo de PACCA esto es el job final `tools/postproceso.sbatch`, que **no pide GPU** a propósito: no hay una sola línea de CUDA en el post-proceso, y ocupar una GPU compartida durante la hora larga que puede tardar el Pareto sería desperdiciarla.

**Antes de reportar tiempo o energía de GEMM/Convolución, lee la advertencia de la sección 7** sobre `t_iter_ms` y `energy_gpu_j` en esos dos kernels.

### 6.1 El grafo de dependencias de `run_full_pipeline_pacca.sh`

```
                      +--> Fase 1 / Fase 2 (opcionales, sin dependientes)
  validación          |
  preliminar ---------+--> F3 GEMM    --> F4 GEMM    --+
  (los tres           +--> F3 Conv    --> F4 Conv    --+--> post-proceso
   kernels)           +--> F3 Stencil --> F4 Stencil --+     (sin GPU)
```

Por qué así, y no un solo proceso largo: `run_full_pipeline.sh` invoca cada `.sbatch` con `bash` en secuencia dentro de **un** proceso, lo que exige tener la GPU reservada de principio a fin. Con las campañas ampliadas (cuatro tamaños × tres barridos de iteraciones × cuatro valores de `K`, en tres kernels) eso son fácilmente decenas de horas seguidas: en un clúster compartido no hay forma de pedir esa reserva, y si el job se cae en la hora 30 se pierde todo.

Dos detalles del grafo que no son arbitrarios:

- **Fase 1 y Fase 2 no están en la cadena de Fase 3/4.** No producen `CSV_*` que el post-proceso consuma, así que meterlas en la cadena solo lograría que un fallo suyo —por ejemplo, CUTLASS sin clonar, que es *opcional*— bloqueara una campaña que no las necesita.
- **El post-proceso usa `afterok`, no `afterany`.** Si cualquiera de los seis jobs de campaña falla, el ANOVA no arranca. Un análisis estadístico sobre una campaña incompleta es peor que no tenerlo, porque parece un resultado.

---

## 7. Cómo analizar los resultados

- **Columnas de CSV**: cada `README.md` de fase documenta el esquema exacto de columnas que produce esa fase (`CSV_DRIFT`, `CSV_SUMMARY`, `CSV_ENERGY`, `CSV_HORIZON`, según la fase).
- **⚠️ Los números de tiempo y energía de GEMM y Convolución anteriores a 2026-09-06 no son utilizables — vuelve a correr esas campañas.** Estaban mal de dos formas independientes: (a) un solo cronómetro envolvía las tres trayectorias de cada iteración y el mismo número se imprimía en las filas `_none` y `_comp`, con los dos `PowerBuffer` cubriendo el mismo intervalo sobre un contador NVML **de todo el dispositivo**, y ambos incluyendo el costo de la referencia FP64; (b) la pausa de checkpoint se abría *antes* de sincronizar, así que el `cudaMemcpy` D2H bloqueaba esperando la cola asíncrona, esa espera caía dentro de la pausa y se restaba del total — el cómputo que se quería medir se descontaba del tiempo medido. Medido en la misma GPU, el segundo defecto inflaba `gflops` entre 6× y 16× (`BF16_none` llegó a reportar 16 440 GFLOPS en una RTX 3050). **El eje de error (`rel_l2`/`rel_linf`) nunca estuvo afectado.** Stencil nunca tuvo ninguno de los dos problemas.

  **Ya está corregido**: los tres tramos se miden en fases separadas, cada una con su cronómetro y su ventana de energía, y la trayectoria de referencia se publica como ruta propia `GPU_FP64` — que además es el punto de Pareto "todo en FP64" que a esos dos kernels les faltaba. Lo vigila `Fase_4/tools/gate4_medicion.py`, dentro de `tools/validacion_preliminar.sbatch`. Ver `Fase_3/GEMM/README.md` para la tabla de antes/después.
- **`anchor_every` significa dos cosas distintas según el kernel.** En GEMM/Convolución varía **por fila** dentro de la misma corrida (`_none` = `0`, `_comp` = `K`); en Stencil es una constante de **toda la invocación**, que comparten hasta las filas de `GPU_FP64`/`CPU_FP64` que nunca ejecutan el ancla. Al agrupar o filtrar por esta columna, no asumas la semántica del otro kernel — ver `Fase_4/tools/README.md`.
- **`energy_gpu_j_per_iter`** es la cantidad comparable entre corridas con distinto número de iteraciones (útil para comparar entre configuraciones de la campaña de ancla, donde el error y la energía a veces necesitan regímenes de `ITERS` distintos — ver la sección 5 y la nota de `tools/README.md` de Fase 3).
- **`energy_window_reliable`**: descarta (no promedies) filas donde valga `0` — la ventana de medición fue demasiado corta para que el contador NVML sea confiable (ver `common/power_sampling.h`).
- **Horizonte "medido" vs. "predicho"**: nunca reportes el predicho si el medido está disponible para el mismo punto — el predicho es una extrapolación para casos donde no se puede correr suficientes iteraciones, no una alternativa igual de buena.

---

## 8. Preguntas frecuentes

**¿Por qué Stencil no tiene "ruta de librería" como GEMM/Convolución?**
Porque no existe una biblioteca de NVIDIA para "aplicar un stencil" — cuBLAS es para álgebra lineal, cuDNN para convolución. El Laplaciano de Stencil se implementa como kernel CUDA propio en todas sus rutas (FP32, FP64, y el WMMA con Tensor Cores), así que la comparación "con TC" vs. "sin TC" en Stencil es limpia desde el principio (mismo nivel de esfuerzo de optimización en ambos lados) — no hace falta la aclaración de "comparaciones justas" que sí aplica a GEMM/Convolución.

**¿Por qué el ancla no corrige todo el drift si K=1 corre en FP64 completo?**
Sí lo hace — con K=1, cada iteración se recalcula en FP64, así que el estado interno converge a la trayectoria FP64 pura. La limitación real es con K>1: el ancla evita que la iteración de ancla introduzca *nuevo* error, pero no reconstruye el drift que ya se acumuló en las iteraciones anteriores. Es exactamente el punto intermedio entre "nunca anclar" (rápido, drift libre) y "anclar siempre" (K=1, exacto, tan lento como FP64) que el barrido de K está diseñado para caracterizar.

**Corrí K=1 y `rel_l2` se quedó en `2e-4`, no bajó a `1e-16`. ¿Está roto el ancla?**
No. Es lo esperado, y hay una cota que lo predice. `CSV_DRIFT` de GEMM/Convolución compara la referencia FP64 contra el buffer `T` **ya cuantizado a 16 bits**, nunca contra `T+comp`, así que ese CSV no puede reportar nada mejor que la precisión del formato — por exacto que sea el mecanismo interno. Con FP16 la cota es `2^-11 = 4.9e-04` y con BF16 `2^-8 = 3.9e-03`; un `2e-4` en FP16 está justo donde debe estar. Lo que sí sería un bug es **superar** esa cota: eso es lo que verifica el gate K=1. Ver la sección 5.

**Encontré algo en el código que no coincide con este manual o con un README de carpeta.**
Confía primero en el `README.md` de la carpeta específica, después en el documento *Plan de Precisión Mixta*, y reporta la discrepancia — este manual es más propenso a quedar desactualizado que el código o su README inmediato.

**No tengo acceso a PACCA todavía / estoy revisando el código en mi laptop, ¿puedo compilar algo?**
No — no hay flujo de compilación local documentado (ver `REQUIREMENTS.md`). Puedes leer, editar y revisar el código sin GPU; para compilar y correr necesitas una sesión en PACCA.
