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

# --------------------------------------------------------------------------
# Grupos ADICIONALES para discriminar la anomalia de rendimiento del stencil
# (Fase 3): la compensacion ESPACIAL, que hace 5 lecturas de comp[] por celda,
# es consistentemente MAS RAPIDA (~1.56x sobre la base) que la Kahan LOCAL, que
# hace 1 (~1.97x), en nx=4096/8192/16384. Dos hipotesis rivales:
#
#   (a) coalescencia: las 5 lecturas son de celdas vecinas contiguas y se
#       sirven en pocas transacciones, asi que el costo no escala con el
#       numero de lecturas.
#   (b) latencia serializada: la ruta local encadena leer comp -> calcular ->
#       escribir comp, y esa dependencia domina sobre el numero de accesos,
#       mientras la espacial emite lecturas independientes que se solapan.
#
# NO se agregan a NCU_QUICK_METRICS: eso encareceria tambien a GEMM y
# Convolution, que no tienen esta pregunta. Los .sbatch que las necesiten las
# concatenan explicitamente (ver run_stencil_tc.sbatch de Fase_3).

# Discrimina (b). Desglose de Warp State Statistics: si la ruta local esta
# limitada por una cadena de dependencias y no por volumen de trafico, sus
# ciclos de stall se concentraran en long_scoreboard (espera de dato de
# memoria global pendiente) y/o wait (dependencia de instruccion de latencia
# fija), con pocos warps elegibles por scheduler; si la espacial solapa sus 5
# lecturas independientes, el mismo stall por lectura se reparte entre mas
# accesos en vuelo y la razon por issue activo baja.
NCU_WARP_STALL_METRICS="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active"

# Discrimina (a). Sectores por request (la columna "Sectors/Req" de Memory
# Workload Analysis) mas los totales crudos de request y sector: si las 5
# lecturas vecinas se sirven coalescidas, sectors_per_request NO escalara con
# el numero de lecturas y el total de requests subira ~5x mientras el de
# sectores sube mucho menos. Si en cambio cada lectura vecina paga su propia
# transaccion, ambos escalan juntos y la hipotesis (a) queda descartada.
NCU_COALESCING_METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"

# Ejecuta el comando dado (tipicamente ncu) con NCU_PROFILING=1 en el entorno.
# El binario objetivo (lanzado por ncu como target-process, que hereda el
# entorno) usa esa variable para marcar sus tiempos como no validos y
# prefijar "NCU_" en el CSV -- los tiempos bajo perfilado quedan inflados
# frente a una corrida limpia en la misma configuracion.
ncu_run() {
    NCU_PROFILING=1 "$@"
}
