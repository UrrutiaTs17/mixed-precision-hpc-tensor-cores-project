# Requisitos de software

Este proyecto corre en **cualquier máquina con una GPU NVIDIA Ampere o más nueva** (ver la sección de hardware abajo — no es "cualquier GPU con Tensor Cores", es una restricción real, léela antes de asumir que tu GPU sirve), con el entorno conda de `environment.yml` activo. **PACCA es una opción de ejecución, no un requisito** — todos los `.sbatch` del proyecto corren igual con `sbatch archivo.sbatch` bajo SLURM (PACCA) o con `bash archivo.sbatch` directo en tu propia máquina/servidor, sin SLURM de por medio (ver "Ejecutar sin SLURM" más abajo).

## Requisito de hardware: por qué Ampere+ y no "cualquier GPU con Tensor Cores"

Volta (`sm_70`, V100) y Turing (`sm_75`, RTX 20xx) **tienen** Tensor Cores, pero **no sirven para este proyecto**: `common/wmma_gemm.cuh` (el kernel WMMA que reutilizan GEMM, Convolución y Stencil en Fase 2-4) usa un pipeline de 3 etapas con `cp.async` (`cuda_pipeline_primitives.h`), una instrucción que **solo existe desde Ampere (`sm_80`)**. Compilar para `sm_70`/`sm_75` con este archivo falla en tiempo de compilación, no en tiempo de ejecución — no hay una ruta de degradación silenciosa.

- **Mínimo real: `sm_80`** (A100, RTX 30xx). `sm_86` (RTX 30xx de consumo), `sm_89` (Ada, RTX 40xx) y `sm_90` (Hopper, H100) también funcionan — `tools/detect_toolchain.sh` detecta la arquitectura real de tu GPU con `nvidia-smi` y compila para ella automáticamente (no hace falta fijar `CUDA_ARCH` a mano salvo que quieras forzar una arquitectura distinta a la GPU 0 detectada).
- BF16 (`--tc bf16` / `--tc-format bf16`) y la ruta 5 (CUTLASS) de GEMM/Convolución **también** exigen `sm_80+` — mismo piso, no es una restricción adicional.
- Si tu única GPU es Volta/Turing, las rutas *sin* Tensor Cores (Fase 1: FP64/FP32 clásico) sí compilan y corren — pero Fase 2-4 (el objeto real del proyecto) no.

## Instalación (recomendada): conda

```bash
conda env create -f environment.yml
conda activate mixed-precision-hpc
nvcc --version   # confirma que quedo en el PATH del entorno
```

`environment.yml` instala: `nvcc`/cuBLAS (canal `nvidia::cuda-toolkit`), `cuDNN` (`conda-forge::cudnn`), OpenBLAS (referencia de CPU de Fase 1-2), y todo el lado de Python (`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`) para el post-proceso de CSV, la estadística inferencial (Etapa 7 del plan) y el Frente de Pareto 3D (Etapa 9).

**Requisito previo, fuera de conda**: el **driver NVIDIA** debe estar instalado en el sistema (conda no lo instala — es un componente de kernel/sistema operativo, no de espacio de usuario). Verifica con `nvidia-smi`; la columna "CUDA Version" de su salida es la versión **máxima** de CUDA que ese driver soporta — el `cuda-toolkit` de `environment.yml` (12.4) debe ser igual o menor a eso, si tu driver es más viejo, edita `environment.yml` para fijar una versión de `cuda-toolkit` compatible antes de crear el entorno.

### Qué NO cubre conda

- **NVML** (`libnvidia-ml.so`) — viene con el driver NVIDIA, no con el toolkit. Ya está presente si `nvidia-smi` funciona. Opt-in de telemetría de energía GPU vía `-DUSE_NVML_TELEMETRY` en la compilación (cada `.sbatch` lo detecta automáticamente — ver `common/power_sampling.h`).
- **CUTLASS** (ruta 5 opcional de GEMM/Convolución, serie **2.x** — no 3.x/CuTe) — header-only, no es paquete conda, se clona aparte:
  ```bash
  git clone --branch v2.11.0 https://github.com/NVIDIA/cutlass "$HOME/cutlass"
  ```
  Por defecto, `run_gemm_tc.sbatch`/`run_conv_tc.sbatch` buscan el checkout en `$HOME/cutlass/include` (variable `CUTLASS_DIR`, sobreescribible con `--export=ALL,CUTLASS_DIR=/otra/ruta` o `CUTLASS_DIR=/otra/ruta bash run_gemm_tc.sbatch` sin SLURM). Si no está presente, el binario compila igual con las 4 rutas restantes (detección vía `__has_include`) y `--cutlass` termina con un mensaje explicando cómo habilitarlo.

  **Parche necesario con compiladores host recientes (GCC 13+)**: CUTLASS v2.11.0 (2022) tiene un typo real en `include/cutlass/matrix.h` — cuatro llamadas a `m.set_slice3x3(...)` cuando el método que existe se llama `set_slice_3x3` (con guion bajo). Con GCC viejo esto a veces se toleraba; con GCC moderno es un error de compilación (`has no member named 'set_slice3x3'`). Está en un utilitario de matrices de rotación/reflexión 3D que este proyecto nunca usa — se llega ahí solo por inclusión transitiva. Arreglo de una línea, verificado en este proyecto (GCC 16.2, CUDA 13.3, `sm_89`):
  ```bash
  sed -i 's/m\.set_slice3x3(/m.set_slice_3x3(/g' "$HOME/cutlass/include/cutlass/matrix.h"
  ```
  Sin este parche, **tanto** `Fase_2/GEMM` como `Fase_2/Convolution` fallan en compilación con `--cutlass`/`CUTLASS_DIR` configurado.

  **Verificado en GPU real** (RTX 4060 Laptop, `sm_89`, 2026-09-03): las 5 rutas de GEMM y las 5 de Convolución compilan y corren limpio con el parche de arriba. La ruta 5 de Convolución tenía además un bug propio (no de CUTLASS): `problem_size.output_size()` se asumía que devolvía un `cutlass::Tensor4DCoord` (la forma 4D de salida); en realidad devuelve `int64_t` (el conteo total de elementos N·P·Q·K). Ya corregido en `conv_tensor_activation.cu` — construye el `Tensor4DCoord` directamente desde los campos `N`/`P`/`Q`/`K` de `Conv2dProblemSize`. La ruta 5 de GEMM no tenía bugs propios, solo necesitaba el parche del typo de arriba.
- **Nsight Compute** (`ncu`, perfilado opcional vía `RUN_NCU=1`) — el paquete conda no publica builds confiables para todas las plataformas, así que no está en `environment.yml`. Instálalo aparte (https://developer.nvidia.com/nsight-compute) si lo necesitas; si no está, `tools/detect_toolchain.sh` lo detecta y fuerza `RUN_NCU=0` con un aviso, sin romper la compilación ni la corrida.

## Alternativa: pip (solo el lado de Python)

Si ya tienes el CUDA Toolkit instalado por tu cuenta (o estás en un clúster con `module load cuda`) y solo necesitas el post-proceso/estadística:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar sin SLURM

Cada `.sbatch` del proyecto es, fuera de un clúster, un script de bash normal — las líneas `#SBATCH` son comentarios. Con el entorno conda activo:

```bash
cd Fase_4/GEMM
bash run_gemm_chained.sbatch
# o, para parametrizar (equivalente a --export=ALL,VAR=valor bajo SLURM):
N_LIST="1024 2048 4096 8192" COMP_LIST="off on" ANCHOR_LIST="0 1 5" bash run_gemm_chained.sbatch
```

`tools/detect_toolchain.sh` (que cada `.sbatch` source-ea) resuelve `nvcc`/`ncu`/`CUDA_ARCH` buscando primero en tu `PATH` (el entorno conda los deja ahí) y solo cae a las rutas fijas históricas de PACCA como último recurso — no hace falta editar ningún `.sbatch` para correr fuera del clúster. `SLURM_SUBMIT_DIR`/`SLURM_JOB_ID` (usados para nombrar logs/resultados) tienen default a `$(pwd)`/`manual` cuando no existen.

## En el clúster (PACCA) — sigue funcionando igual

- **SLURM** (`sbatch`) — opcional, no requerido por el proyecto en sí (ver arriba).
- **RAPL** (`/sys/class/powercap/intel-rapl*`) legible por el usuario del job, para telemetría de energía CPU. Puede requerir un permiso explícito del administrador (mitigación PLATYPUS en kernels Linux recientes restringe la lectura a root o a usuarios con `CAP_PERFMON`/regla `udev`) — si no está disponible, la energía CPU sale `NaN` sin romper el resto de la medición.
- Permiso de **`nvidia-smi -lgc`** (o modo persistente) si se va a intentar el aislamiento térmico de la Etapa 5 del plan — al momento de escribir esto, está denegado en el nodo de PACCA; ver la sección de Limitaciones del documento de tesis.

## Qué NO hace falta

- **No se necesita GPU en la máquina donde se edita o revisa el código** (este entorno de desarrollo, por ejemplo, no tiene ninguna) — solo hace falta para compilar y correr.
- **No hay contenedor Docker/Singularity documentado.** `environment.yml` cubre el mismo objetivo (reproducibilidad del entorno) sin necesitar acceso privilegiado al host; si tu entorno de destino sí exige un contenedor, agrégalo aquí antes de depender de él en un script.
