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

## `validacion_preliminar.sbatch`

La puerta que hay que pasar **antes** de comprometer horas de cola en una campaña. Corre, de lo más barato a lo más caro:

1. **Orden de operandos** de `cublasDgemm` en GEMM y Convolución (`Fase_3/tools/verificar_orden_operandos_*.py`). Es lo más importante y lo más barato — `N=32` y `hw=64`, segundos. Si esto falla, ninguna columna `rel_l2`/`rel_linf` de esos dos kernels es válida.
2. **Prueba de humo** de los tres kernels (`SMOKE_TEST=1` sobre el `.sbatch` de Fase 4 de cada uno): compila, corre el tamaño más chico y extrae los CSV. Además comprueba que la columna `anchor_every` existe y que en GEMM/Convolución **varía por ruta** (`_none`=0, `_comp`=K) dentro de la misma corrida.
3. **Gates K=0 y K=1** del ancla FP64 en los tres kernels (`Fase_4/<kernel>/gate3_ancla.sbatch`).

No usa `set -e` a propósito: corre **todos** los pasos aunque uno falle, para que un solo envío diga todo lo que hay que arreglar en vez de una cosa por vez. Sale con `0` si todo pasa, o con el número de pasos fallidos. Usa un `RESULTS_DIR` propio (`results/humo_preliminar`) para no contaminar los resultados de campaña — los `.sbatch` hacen `tee -a` sobre `results/run_<job>.log`, y fuera de SLURM el id de job es la constante `manual`.

`run_full_pipeline_pacca.sh` lo envía primero y cuelga las campañas de él con `afterok`, así que un fallo aquí impide que ninguna campaña arranque. `SKIP_VALIDACION=1` lo salta (no recomendado).

## `postproceso.sbatch`

Estadística inferencial (Etapa 7) y Frente de Pareto 3D (Etapa 9) sobre todos los `results/` de Fase 3 y Fase 4 de los tres kernels.

**No pide GPU a propósito**: no hay una sola línea de CUDA aquí, solo `pandas`/`statsmodels`/`matplotlib` sobre CSV ya extraídos. Por eso el archivo no lleva `#SBATCH --gres=gpu:1` ni `--partition` — la partición la elige quien lo envía (`run_full_pipeline_pacca.sh` lo hace con `POST_PARTITION`), y sin ese flag cae en la partición por defecto del clúster. Pasar un nombre de partición inexistente hace que `sbatch` **rechace** el envío, así que no se adivina ninguno.

Se envía normalmente como último eslabón de una cadena (`--dependency=afterok:J1:J2:...`), pero también corre solo sobre lo que ya haya en `results/`.
