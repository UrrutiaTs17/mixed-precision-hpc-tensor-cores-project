# common_ncu.sh — definiciones compartidas para el perfilado con Nsight Compute.
#
# Se declara UNA sola vez y se reutiliza entre run_gemm_tc.sbatch y
# run_conv_tc.sbatch (via `source`) para no duplicar la lista de metricas
# quick entre archivos. No es ejecutable por si mismo: solo exporta variables
# y funciones auxiliares.

# Metricas rapidas de validacion de Tensor Cores (modo NCU_MODE=quick).
# Objetivo: confirmar que el kernel realmente emite instrucciones HMMA y por
# que ruta de precision (fp16/bf16 -> fp32), mas ocupacion alcanzada y ancho
# de banda efectivo (DRAM/L1), sin el costo de --set full. Sin
# sm__ops_path_tensor_src_tf32_dst_fp32.sum (Stencil no usa TF32: siempre da 0
# y consume un paso de replay; si GEMM/Convolution necesitan ese contador,
# agreguenlo en su propio --metrics en vez de aqui).
NCU_QUICK_METRICS="sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

# Ejecuta el comando dado (tipicamente ncu) con NCU_PROFILING=1 en el entorno.
# El binario objetivo (lanzado por ncu como target-process, que hereda el
# entorno) usa esa variable para marcar sus tiempos como no validos y
# prefijar "NCU_" en el CSV -- los tiempos bajo perfilado quedan inflados
# frente a una corrida limpia en la misma configuracion.
ncu_run() {
    NCU_PROFILING=1 "$@"
}
