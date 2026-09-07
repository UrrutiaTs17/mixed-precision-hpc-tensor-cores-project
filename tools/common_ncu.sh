# tools/common_ncu.sh — definiciones compartidas para el perfilado con Nsight Compute.
#
# Se declara UNA sola vez en tools/ (fuera de cualquier Fase_N) y se reutiliza
# vía `source` desde los .sbatch que perfilan con ncu en las fases 2, 3 y 4,
# para no duplicar la lista de métricas entre kernels y fases. No es
# ejecutable por sí mismo: solo exporta variables y funciones auxiliares.
#
# Migrado desde old/tools/common_ncu.sh sin cambios de lógica.

# Métricas rápidas de validación de Tensor Cores (NCU_MODE=quick). Objetivo:
# confirmar que el kernel realmente emite instrucciones HMMA y por qué ruta
# de precisión (fp16/bf16 -> fp32), más ocupación alcanzada y ancho de banda
# efectivo (DRAM/L1), sin el costo de --set full. No incluye
# sm__ops_path_tensor_src_tf32_dst_fp32.sum (Stencil no usa TF32: siempre da
# 0 y consume un paso de replay; si GEMM/Convolución necesitan ese contador,
# agréguenlo en su propio --metrics, no aquí).
#
# Las últimas 3 métricas de la lista existen para poder demostrar, con
# --kahan on en Stencil, si la compensación mueve el kernel WMMA de
# latency-bound a memory-bound (comp[] agrega 3 sumas FP32/celda y un buffer
# FP32 leído+escrito por iteración -- ~1.07 GB extra a 16384², ~1.06 -> ~3.2 GB
# de tráfico DRAM en teoría):
#   launch__registers_per_thread: Kahan sube presión de registros; si cruza
#     el umbral de spill, la ocupación cae.
#   sm__sass_thread_inst_executed_op_fadd_pred_on.sum: cuenta las 3 sumas
#     FP32 nuevas por celda (y = val - comp; comp_nuevo = s - y, más la resta
#     implícita); debe subir con --kahan on y quedarse en la cuenta base con
#     --kahan off.
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct: eficiencia
#     de escritura global (análoga a la de lectura ya presente para _ld);
#     cubre el nuevo tráfico de escritura de comp[].
#   launch__occupancy_limit_registers: confirma si registros (y no shared)
#     pasa a ser el limitante de ocupación con Kahan activo.
NCU_QUICK_METRICS="sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,launch__registers_per_thread,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,launch__occupancy_limit_registers"

# --------------------------------------------------------------------------
# Grupos ADICIONALES, específicos de Stencil, para discriminar por qué la
# compensación ESPACIAL (5 lecturas de comp[] por celda) es consistentemente
# más rápida (~1.56x sobre la base) que Kahan LOCAL (1 lectura, ~1.97x) en
# nx=4096/8192/16384. Dos hipótesis rivales:
#   (a) coalescencia: las 5 lecturas son de celdas vecinas contiguas y se
#       sirven en pocas transacciones, así que el costo no escala con el
#       número de lecturas.
#   (b) latencia serializada: la ruta local encadena leer comp -> calcular ->
#       escribir comp, y esa dependencia domina sobre el número de accesos,
#       mientras la espacial emite lecturas independientes que se solapan.
#
# NO se agregan a NCU_QUICK_METRICS: eso encarecería también a GEMM y
# Convolución, que no tienen esta pregunta. Los .sbatch de Stencil que las
# necesiten las concatenan explícitamente.

# Discrimina (b): si la ruta local está limitada por una cadena de
# dependencias y no por volumen de tráfico, sus ciclos de stall se
# concentrarán en long_scoreboard (espera de dato de memoria global
# pendiente) y/o wait (dependencia de instrucción de latencia fija), con
# pocos warps elegibles por scheduler; si la espacial solapa sus 5 lecturas
# independientes, el mismo stall por lectura se reparte entre más accesos en
# vuelo y la razón por issue activo baja.
NCU_WARP_STALL_METRICS="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active"

# Discrimina (a): sectores por request ("Sectors/Req" de Memory Workload
# Analysis) más los totales crudos de request y sector -- si las 5 lecturas
# vecinas se sirven coalescidas, sectors_per_request NO escala con el número
# de lecturas y el total de requests sube ~5x mientras el de sectores sube
# mucho menos; si cada lectura paga su propia transacción, ambos escalan
# juntos y (a) queda descartada.
NCU_COALESCING_METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"

# Ejecuta el comando dado (típicamente ncu ...) con NCU_PROFILING=1 en el
# entorno. El binario objetivo (lanzado por ncu como target-process, que
# hereda el entorno) usa esa variable para marcar sus tiempos como no
# válidos y prefijar "NCU_" en el CSV -- los tiempos bajo perfilado quedan
# inflados frente a una corrida limpia en la misma configuración, y este
# marcado evita que se mezclen sin querer en el análisis.
ncu_run() {
    NCU_PROFILING=1 "$@"
}
