# common_ncu.sh — definiciones compartidas para el perfilado con Nsight Compute.
#
# Se declara UNA sola vez en tools/ (fuera de cualquier Fase_N, ver README.md)
# y se reutiliza via `source` entre los .sbatch que perfilan con ncu:
# run_gemm_tc/run_conv_tc/run_stencil_tc de Fase_2 y run_stencil_tc de Fase_3
# (run_stencil_horizon de Fase_3 no perfila con ncu, no lo necesita), para no
# duplicar la lista de metricas quick entre fases. No es ejecutable por si
# mismo: solo exporta variables y funciones auxiliares.

# Metricas rapidas de validacion de Tensor Cores (modo NCU_MODE=quick).
# Objetivo: confirmar que el kernel realmente emite instrucciones HMMA y por
# que ruta de precision (fp16/bf16 -> fp32), mas ocupacion alcanzada y ancho
# de banda efectivo (DRAM/L1), sin el costo de --set full. Sin
# sm__ops_path_tensor_src_tf32_dst_fp32.sum (Stencil no usa TF32: siempre da 0
# y consume un paso de replay; si GEMM/Convolution necesitan ese contador,
# agreguenlo en su propio --metrics en vez de aqui).
#
# 3 metricas agregadas para Kahan (Fase 3, Stencil): comp[] agrega 3 sumas
# FP32/celda y un buffer FP32 leido+escrito por iteracion (1.07 GB extra a
# 16384^2, ~1.06 -> ~3.2 GB de trafico DRAM), lo que en teoria mueve el kernel
# WMMA de latency-bound (20.0% pico DRAM, 61.1% ocupacion alcanzada hoy) a
# memory-bound. Sin estas 3 metricas eso no se puede demostrar con --kahan on:
#   launch__registers_per_thread                 : Kahan sube presion de
#     registros (hoy 32 reg/hilo, limitante de ocupacion es shared, no
#     registros); si --kahan on cruza el umbral de spill, la ocupacion cae.
#   sm__sass_thread_inst_executed_op_fadd_pred_on.sum : cuenta las 3 sumas
#     FP32 nuevas por celda (y = val - comp; comp_nuevo = s - y, mas la resta
#     implicita); debe subir con --kahan on y quedarse en la cuenta base con
#     --kahan off.
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct : eficiencia
#     de escritura global (analoga a la de lectura ya presente para _ld);
#     cubre el nuevo trafico de escritura de comp[] que _ld no captura.
#   launch__occupancy_limit_registers            : confirma si registros (y
#     no shared) pasa a ser el limitante de ocupacion con Kahan activo.
NCU_QUICK_METRICS="sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,launch__registers_per_thread,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,launch__occupancy_limit_registers"

# Ejecuta el comando dado (tipicamente ncu) con NCU_PROFILING=1 en el entorno.
# El binario objetivo (lanzado por ncu como target-process, que hereda el
# entorno) usa esa variable para marcar sus tiempos como no validos y
# prefijar "NCU_" en el CSV -- los tiempos bajo perfilado quedan inflados
# frente a una corrida limpia en la misma configuracion.
ncu_run() {
    NCU_PROFILING=1 "$@"
}
