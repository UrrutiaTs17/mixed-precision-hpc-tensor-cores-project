# Evaluación Experimental del Impacto Numérico y Energético de la Computación en Precisión Mixta mediante Tensor Cores en GPUs NVIDIA para Kernels HPC

Trabajo de grado — Escuela de Ingeniería de Sistemas e Informática, Universidad Industrial de Santander (UIS).

## De qué trata este proyecto

La computación científica de alto rendimiento (HPC) ha dependido históricamente de la doble precisión (FP64) para garantizar estabilidad numérica. Las GPUs NVIDIA modernas, en cambio, dedican una porción enorme de su silicio a **Tensor Cores**: unidades especializadas que multiplican y acumulan matrices a velocidades muy superiores, pero usando formatos de menor precisión (FP16, BF16). Usarlos en kernels clásicos de HPC —GEMM, Convolución, Stencil— promete velocidad y ahorro energético, al costo de introducir error de redondeo.

Este proyecto **no busca demostrar que Tensor Cores "funcionan"** — busca **cuantificar exactamente cuándo y cuánto conviene usarlos**: para cada kernel, ¿qué combinación de formato y técnica de corrección da el mejor compromiso entre tiempo de ejecución, energía consumida y error numérico, y a partir de qué punto la ganancia de velocidad deja de valer la pérdida de exactitud?

La pregunta de investigación completa está en el documento *Plan de Trabajo de Grado* (raíz del repositorio). En términos operativos, el proyecto:

1. Implementa GEMM, Convolución 2D y Stencil 2D en FP64, FP32, FP16 y BF16, activando Tensor Cores explícitamente (cuBLAS/cuDNN y kernels WMMA propios).
2. Mide rendimiento (latencia, TFLOPS) y energía (NVML para GPU, RAPL para CPU host, Energy-Delay Product) de cada combinación.
3. Mide cómo se degrada la exactitud numérica en esquemas iterativos, y evalúa técnicas de compensación (suma de Kahan, compensación espacial, y el mecanismo de ancla FP64 descrito más abajo) contra una referencia FP64 (ground truth).
4. Construye un Frente de Pareto en el espacio (Tiempo, Energía, Error) por cada kernel, para traducir los datos en una directriz de ingeniería: qué precisión usar según la tolerancia al error de la aplicación.

## Qué corrige realmente el proyecto (y qué no)

Es fácil malinterpretar el objetivo como "usar Tensor Cores y arreglar el error con Kahan hasta llegar a FP64". No es así, y vale la pena ser precisos:

- **FP64 es siempre la referencia**, nunca el objetivo a igualar. Todo lo demás (FP32, FP16, BF16, con o sin compensación) se mide *contra* FP64, no se fuerza *a* FP64.
- Hay **dos capas de corrección de error, no una**. Los Tensor Cores ya acumulan internamente en FP32 aunque los operandos sean FP16/BF16 — eso es hardware, siempre activo, gratis. Las técnicas de compensación (Kahan, compensación espacial) atacan un problema distinto: el redondeo de **guardar** el estado en FP16/BF16 entre iteraciones de un esquema encadenado.
- La suma de Kahan clásica, aplicada célula por célula, tiene un problema estructural: el residuo de una celda nunca lo relee ninguna de sus vecinas, que son las que sufren el error real de propagación. Bajo el operador de estrés que usa la campaña principal, esto la vuelve **indistinguible de no compensar** (incluso algo peor: +6% a +9% de error). Bajo un operador más difusivo, en cambio, sí ayuda (-24% a -26%) — su compensación depende de que el residuo esté correlacionado temporalmente, algo que el operador de estrés no garantiza. La **compensación espacial** (explota la linealidad del operador) funciona en ambos casos y de forma más consistente, y es una contribución propia del proyecto. Ver `Fase_3/Stencil/README.md` para el detalle completo por modo de operador.
- El techo alcanzable con compensación por almacenamiento es **precisión cercana a FP32**, no a FP64 — es un límite de información, no de esfuerzo de ingeniería: no se puede recuperar con una técnica de compensación lo que el hardware nunca calculó. Para acercarse más a FP64 sin pagar su costo completo, el proyecto incorpora un mecanismo adicional: el **ancla FP64** (ver abajo).

## El mecanismo de ancla FP64

Cada `K` iteraciones, en vez de calcular ese paso en baja precisión, se calcula **una vez, completo, en FP64**, y el resultado reemplaza al de la ruta rápida. El costo extra es una evaluación FP64 por cada `K` pasos — no una repetición de las `K` iteraciones anteriores (eso costaría igual que correr todo en FP64). Corrige el error de *ese* paso; no reconstruye el drift ya acumulado antes del ancla. Cuánto ayuda en la práctica —y a partir de qué `K` deja de valer la pena frente a su costo— es una pregunta empírica que el proyecto mide, no asume.

La especificación completa (pseudocódigo, gates de validación, diseño experimental) está en el documento **Plan de Precisión Mixta**, secciones 01 (Stencil) y 02 (extensión a GEMM/Convolución, que primero necesitan encadenarse igual que Stencil).

## Estructura del repositorio

```
mixed-precision-hpc-tensor-cores-project/
├── common/              # Código compartido: validación CUDA/cuBLAS/cuDNN, métricas de
│                         # tiempo/error, telemetría de energía, motor de ancla+compensación.
├── Fase_1/               # Línea base FP64/FP32, sin Tensor Cores.
│   ├── GEMM/
│   ├── Convolution/
│   └── Stencil/
├── Fase_2/               # Activación de Tensor Cores (FP16/BF16), sin encadenar iteraciones.
│   ├── GEMM/
│   ├── Convolution/
│   └── Stencil/
├── Fase_3/               # Encadenamiento genuino (salida(i) → entrada(i+1)), drift,
│   ├── GEMM/              # compensación. Los tres kernels, no solo Stencil.
│   ├── Convolution/
│   ├── Stencil/
│   └── tools/            # Post-proceso de CSV (extracción, resumen).
├── Fase_4/               # Ancla FP64, diseño factorial, telemetría de energía
│   ├── GEMM/              # completa, Frente de Pareto 3D, análisis estadístico.
│   ├── Convolution/
│   ├── Stencil/
│   └── tools/
├── tools/                # Utilidades compartidas: perfilamiento (Nsight Compute),
│                          # detección de toolchain, validación preliminar y
│                          # post-proceso sin GPU.
├── docs/
│   └── MANUAL.md         # Manual del estudiante: qué es cada archivo, cómo correrlo,
│                          # qué datos produce, cómo se analizan.
├── REQUIREMENTS.md       # Software y entorno necesarios para compilar y correr.
├── run_full_pipeline.sh        # Orquestador para una máquina propia con GPU.
├── run_full_pipeline_pacca.sh  # Orquestador para un clúster con SLURM (jobs + dependencias).
├── Documento Plan Proyecto de Grado.docx.pdf   # Plan de tesis oficial.
└── README.md             # Este archivo.
```

Cada carpeta de fase, y cada subcarpeta de kernel dentro de ella, tiene su propio `README.md` con el detalle de qué hace, qué parámetros acepta y qué produce. El manual en `docs/MANUAL.md` es la puerta de entrada recomendada si es la primera vez que trabajas en el proyecto — enlaza a todo lo demás en el orden en que conviene leerlo.

## Kernels evaluados

- **GEMM**: matrices cuadradas `N = 1024, 2048, 4096, 8192`.
- **Convolución 2D**: dominios cuadrados `H=W = 64, 128, 256, 512`, con `C=K=64` canales y filtro `3×3`.
- **Stencil 2D**: dominios cuadrados `NX=NY = 4096, 8192, 16384`.

Estos son los tamaños de las campañas reportables en las cuatro fases. Los
tamaños reducidos que aparecen en pruebas de humo y gates se usan únicamente
para validar compilación y correctitud; sus resultados no forman parte del
dataset experimental.

## Formatos de precisión

FP64 (referencia), FP32, FP16 (Tensor Cores), BF16 (Tensor Cores). FP8/INT8 quedan explícitamente fuera de alcance (ver la sección de Limitaciones del plan de tesis) — su rango dinámico es insuficiente para variables físicas de simulaciones continuas.

## Herramientas

- **Compilador**: NVIDIA `nvcc`.
- **Bibliotecas**: cuBLAS, cuDNN, OpenBLAS (referencia de CPU).
- **Profiling**: NVIDIA Nsight Compute — verificación real de activación de Tensor Cores (conteo de instrucciones HMMA), no solo aspiracional.
- **Telemetría**: NVML (GPU), RAPL (CPU host).
- **Post-procesamiento**: Python 3 (`scipy`, `statsmodels`, `pandas`, `numpy`, `matplotlib`) — ver `REQUIREMENTS.md`.
- **Ejecución**: cualquier máquina con GPU NVIDIA Ampere+ (`sm_80+`) y el entorno conda de `environment.yml` — ver `REQUIREMENTS.md`. PACCA (vía SLURM/`sbatch`) es una opción, no un requisito: todo `.sbatch` del proyecto corre igual con `bash archivo.sbatch` directo, sin SLURM.

## Por dónde empezar

1. Lee `REQUIREMENTS.md` para el entorno necesario — `conda env create -f environment.yml` deja todo listo (toolchain CUDA + análisis en Python) en cualquier máquina con GPU Ampere+.
2. Lee `docs/MANUAL.md` — es la guía completa, pensada para quien se une al proyecto sin haber visto el código antes: qué es cada archivo, qué ejecuta, qué datos obtiene, cómo se analizan, y cómo lanzar tanto una corrida individual como una campaña completa.
3. **Antes de cualquier campaña**, corre la validación preliminar: `bash tools/validacion_preliminar.sbatch`. Es un job corto que verifica el orden de operandos de `cublasDgemm` en GEMM y Convolución, hace una prueba de humo de los tres kernels y corre los gates K=0/K=1 del ancla FP64. Sale con `0` solo si todo pasa.
4. Para lanzar la campaña:
   - En un clúster con SLURM: `bash run_full_pipeline_pacca.sh` — envía cada fase como un job independiente encadenado con `--dependency=afterok` (`DRY_RUN=1` imprime el grafo sin enviar nada). Es el camino recomendado: no exige tener la GPU reservada durante decenas de horas seguidas.
   - En una máquina propia: `bash run_full_pipeline.sh` corre las cuatro fases en secuencia y termina con la estadística y el Frente de Pareto 3D. **Su default es la campaña completa**; para una prueba rápida, `PIPELINE_MODE=smoke`.
5. Si vas a modificar el mecanismo de ancla o compensación, lee primero el documento **Plan de Precisión Mixta** (comparte el link con tu director/compañeros si no lo tienes) — es la especificación normativa; el código debe seguirla, no al revés.

## Equipo

**Director**
- Gilberto Javier Díaz Toro, Ph.D. — Escuela de Ingeniería de Sistemas e Informática, UIS

**Autores / Investigadores**
- Karen Dayana Mateus Gomez (2212765) — Escuela de Ingeniería de Sistemas e Informática, UIS
- William Andrés Urrutia Torres (2220058) — Escuela de Ingeniería de Sistemas e Informática, UIS

**Institución**: Universidad Industrial de Santander (UIS), Facultad de Ingenierías Fisicomecánicas, Escuela de Ingeniería de Sistemas e Informática.

**Modalidad**: Trabajo de investigación.
