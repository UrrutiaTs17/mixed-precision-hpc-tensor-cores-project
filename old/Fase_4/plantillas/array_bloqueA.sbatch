#!/bin/bash
#SBATCH --job-name=f4var_A_none_kahan
#SBATCH --output=logs/f4var_A_%A_%a.out
#SBATCH --error=logs/f4var_A_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
# --time y --array los fija el lanzador en la linea de sbatch (walltime
# calibrado por bloque, --array=1-<REPLICAS>%<THROTTLE>). Las directivas de
# recursos replican las de Fase_3/Stencil/run_stencil_tc.sbatch: mismo nodo,
# mismos CPUs y memoria, y --exclusive porque compartir nodo contaminaria
# RAPL, potencia y tiempo, que son justo las variables bajo medicion.

# ---------------------------------------------------------------------------
# ARRAY 1 (bloque A): tratamientos "none" y "kahan_local".
#
#   SPATIAL_COMP=off  +  KAHAN_LIST="off on"
#
# SPATIAL_COMP=off es la UNICA via para obtener el barrido factorial de Kahan
# local (run_stencil_tc.sbatch fuerza KAHAN_LIST=off cuando SPATIAL_COMP=on).
# Este array NUNCA pone spatial=on, asi que kahan=on + spatial=on es imposible
# por construccion: el tratamiento espacial vive en el ARRAY 2, disjunto.
#
# Por tarea (una replica): 2 valores de KAHAN x 2 valores de ITERS = 4
# invocaciones del binario; cada una con --tc both -> FP16 y BF16 en la misma
# invocacion => 8 observaciones WMMA por tarea.
# ---------------------------------------------------------------------------
BLOQUE=A
TRATAMIENTOS="none+kahan_local"
SPATIAL_COMP=off
KAHAN_LIST="off on"
INVOCACIONES=4
OBS_WMMA=8

# shellcheck source=/dev/null
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cuerpo_tarea.sh"
