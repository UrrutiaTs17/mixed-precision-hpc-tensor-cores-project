# common_ncu.sh — definiciones compartidas para el perfilado con Nsight Compute.
#
# Se declara UNA sola vez y se reutiliza entre run_gemm_tc.sbatch y
# run_conv_tc.sbatch (via `source`) para no duplicar la lista de metricas
# quick entre archivos. No es ejecutable por si mismo: solo exporta variables
# y funciones auxiliares.

# Metricas rapidas de validacion de Tensor Cores (modo NCU_MODE=quick).
# Objetivo: confirmar que el kernel realmente emite instrucciones HMMA y por
# que ruta de precision (fp16/bf16/tf32 -> fp32), sin el costo de --set full.
NCU_QUICK_METRICS="sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_tf32_dst_fp32.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
