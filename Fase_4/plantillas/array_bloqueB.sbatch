#!/bin/bash
#SBATCH --job-name=f4var_B_spatial
#SBATCH --output=logs/f4var_B_%A_%a.out
#SBATCH --error=logs/f4var_B_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
# --time y --array los fija el lanzador en la linea de sbatch. Recursos
# identicos a los del bloque A y a los de Fase_3/Stencil/run_stencil_tc.sbatch:
# --exclusive es obligatorio, compartir nodo contaminaria RAPL, potencia y
# tiempo.

# ---------------------------------------------------------------------------
# ARRAY 2 (bloque B): tratamiento "spatial".
#
#   SPATIAL_COMP=on  +  KAHAN_LIST="off"
#
# Con SPATIAL_COMP=on, run_stencil_tc.sbatch fuerza ademas KAHAN_LIST=off por
# su cuenta. Este array NUNCA pide kahan=on: la pareja kahan=on + spatial=on no
# existe en el diseno y no puede producirse desde aqui.
#
# Las rutas WMMA de este bloque salen etiquetadas WMMA_FP16_SP / WMMA_BF16_SP
# en TODAS sus filas CSV, asi que conviven sin ambiguedad con las del bloque A.
#
# Por tarea (una replica): 2 valores de ITERS = 2 invocaciones del binario,
# cada una con --tc both => 4 observaciones WMMA por tarea.
# ---------------------------------------------------------------------------
BLOQUE=B
TRATAMIENTOS="spatial"
SPATIAL_COMP=on
KAHAN_LIST="off"
INVOCACIONES=2
OBS_WMMA=4

# shellcheck source=/dev/null
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cuerpo_tarea.sh"
