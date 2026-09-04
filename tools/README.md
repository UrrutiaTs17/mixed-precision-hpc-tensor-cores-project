# tools/

Utilidades compartidas entre fases y kernels — no pertenece a ningún kernel específico, por eso vive fuera de `Fase_N/`.

## `detect_toolchain.sh`

Detección **portable** de `nvcc`/`ncu`/arquitectura de GPU (`detect_nvcc`, `detect_ncu`, `detect_cuda_arch`) — todos los `.sbatch` del proyecto (Fase 1-4) lo source-ean para no tener hardcodeada la ruta de PACCA. Con el entorno de `environment.yml` activo, cualquier `.sbatch` corre igual con `sbatch archivo.sbatch` (PACCA) o `bash archivo.sbatch` directo en cualquier otra máquina con GPU Ampere+ — ver `REQUIREMENTS.md`, sección "Ejecutar sin SLURM", y el comentario de cabecera del propio script.

## `common_ncu.sh`

Definiciones que se cargan con `source tools/common_ncu.sh` desde los `.sbatch` que perfilan con Nsight Compute (`ncu`):

- `NCU_QUICK_METRICS`: métricas rápidas para confirmar que un kernel realmente activa Tensor Cores (instrucciones HMMA, por qué ruta de precisión) más ocupación y ancho de banda, sin el costo de `--set full`. Usadas por los tres kernels.
- `NCU_WARP_STALL_METRICS` / `NCU_COALESCING_METRICS`: métricas adicionales, específicas de Stencil, para diagnosticar por qué la compensación espacial resulta más rápida que Kahan local pese a leer más datos por celda — solo los `.sbatch` de Stencil las concatenan; agregarlas a `NCU_QUICK_METRICS` encarecería innecesariamente el perfilado de GEMM/Convolución.
- `ncu_run()`: envoltorio que marca `NCU_PROFILING=1` en el entorno del proceso perfilado, para que el propio binario sepa que sus tiempos van a salir inflados por el overhead de `ncu` y los etiquete como tales en el CSV en vez de mezclarlos con una corrida limpia.

Antes de perfilar cualquier kernel nuevo, este es el primer lugar a mirar — no dupliques la lista de métricas dentro de un `.sbatch` de fase.
