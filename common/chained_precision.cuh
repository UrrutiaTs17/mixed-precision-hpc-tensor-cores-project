// common/chained_precision.cuh
//
// Motor compartido de encadenamiento + compensacion + ancla FP64, usado por
// Stencil, GEMM y Convolucion en Fase 3/4. Implementa el mecanismo
// especificado y corregido en el documento "Plan de Precision Mixta"
// (secciones 01 y 02) -- ese documento es la referencia normativa de este
// archivo; si hay una discrepancia entre el comentario de una funcion aqui
// y lo que ese documento describe, el documento manda y este archivo tiene
// un bug.
//
// QUE RESUELVE. Cada kernel encadena un operador lineal L sobre un estado
// que vive en baja precision (T = __half o __nv_bfloat16):
//     Stencil:      u(n+1) = L(u(n))       -- Laplaciano de 5 puntos
//     GEMM:         X(n+1) = X(n) * A      -- A ortogonal escalada, exacta en T
//     Convolucion:  X(n+1) = conv(X(n), W) -- mismo Laplaciano, como filtro 3x3
// El estado se guarda como el par (T, comp): T es el valor en baja
// precision; comp es el residuo que permite reconstruir el valor exacto
// conocido hasta ahora (comp[i] = valor_exacto[i] - tc_to_float(T[i])).
//
// DOS MECANISMOS DISTINTOS, NO UNO:
//   1. Compensacion (--comp on): corrige el redondeo de ALMACENAR el estado
//      en T entre iteraciones. Se apoya en que L es lineal:
//      L(T+comp) = L(T) + L(comp). Ver low_precision_step() de cada kernel
//      (no vive aqui: cada kernel implementa su propio L de bajo costo).
//   2. Ancla FP64 (--anchor-every K): cada K iteraciones, el paso se
//      recalcula COMPLETO en FP64 en vez de en baja precision. Corrige el
//      error de UN paso, no el drift ya acumulado (ver seccion 01 del plan
//      para la discusion completa de que corrige y que no).
//
// BUG YA ENCONTRADO Y CORREGIDO EN EL DISENO (no lo repitas al modificar
// este archivo): si `comp` se guarda en `float`, la re-siembra de un paso
// de ancla trunca el resultado FP64 a FP32 ANTES de guardarlo, y entonces
// ni con --anchor-every 1 se alcanza la exactitud de FP64 -- el gate K=1 de
// la seccion 01 fallaria. Por eso comp se declara en `double` en todo este
// archivo. A cambio, el par (T, comp) con ancla activo pasa a ocupar 10
// bytes/celda en FP16 (2 + 8), mas que los 8 bytes de FP64 puro: en cuanto
// se usa el ancla, la ventaja deja de ser de ancho de banda de memoria y es
// puramente de throughput de computo. Es una consecuencia real del diseno,
// no un descuido -- documentala si la mides.
//
// DOS MODOS DE COMP, PARA PRESERVAR EL COMPORTAMIENTO ACTUAL CON EL ANCLA
// DESHABILITADA (--anchor-every 0): en un paso NORMAL, el computo se sigue
// haciendo en `float` exactamente igual que antes de que existiera el
// ancla -- solo el resultado final se ENSANCHA a double al guardarlo
// (widen_comp_to_double). Nunca se resta en double en un paso normal. Esto
// es lo que hace que el gate K=0 (bit-identico al comportamiento actual)
// sea alcanzable: double(float_valor) despues float(double_valor) recupera
// el float original exacto, asi que ensanchar y luego volver a angostar en
// la siguiente iteracion no cambia ningun bit frente a haber usado float
// todo el tiempo.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "cuda_checks.cuh"

namespace chained_precision {

// ---------------------------------------------------------------------
// Kernels elementales compartidos (agnosticos de la forma del problema:
// funcionan igual sobre una grilla de Stencil, una matriz de GEMM o un
// campo de Convolucion, porque solo miran indice a indice).
// ---------------------------------------------------------------------

// Reconstruye el valor exacto conocido hasta ahora para el elemento i:
// tc_to_float(T[i]) + comp[i] si la compensacion esta activa, o solo
// tc_to_float(T[i]) si no -- en double, para que el llamador (el paso de
// ancla) pueda usarlo como entrada de un calculo FP64 sin perder digitos
// en la propia reconstruccion.
template <typename T, typename ToFloatFn>
__global__ void reconstruct_exact_kernel(const T* __restrict__ t_in,
                                          const double* __restrict__ comp_in, bool comp_on,
                                          double* __restrict__ exact_out, int n,
                                          ToFloatFn tc_to_float) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double base = static_cast<double>(tc_to_float(t_in[i]));
  exact_out[i] = comp_on ? (base + comp_in[i]) : base;
}

// Paso de ancla: dado el resultado FP64 completo de este paso (out64,
// calculado por el kernel L_fp64 propio de cada operador), cuantiza a T y
// guarda el residuo SIN truncar a float -- comp_out[i] = out64[i] -
// double(tc_to_float(T_out[i])). Es la operacion inversa de
// reconstruct_exact_kernel, y junto con ella es lo que hace exacto el gate
// K=1 (ver el aviso de "BUG YA ENCONTRADO" arriba: la version anterior
// truncaba aqui a float y por eso fallaba).
template <typename T, typename FromFloatFn, typename ToFloatFn>
__global__ void reseed_from_fp64_kernel(const double* __restrict__ out64, T* __restrict__ t_out,
                                         double* __restrict__ comp_out, int n,
                                         FromFloatFn float_to_tc, ToFloatFn tc_to_float) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const T q = float_to_tc(static_cast<float>(out64[i]));
  t_out[i] = q;
  comp_out[i] = out64[i] - static_cast<double>(tc_to_float(q));
}

// Ensancha a double el comp calculado en un paso NORMAL (en float, por el
// low_precision_step propio de cada kernel) -- ver la nota "DOS MODOS DE
// COMP" arriba sobre por que esto preserva el gate K=0.
__global__ void widen_comp_to_double_kernel(const float* __restrict__ comp_out_f,
                                             double* __restrict__ comp_out_d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  comp_out_d[i] = static_cast<double>(comp_out_f[i]);
}

inline int grid_size_for(int n, int block_size = 256) { return (n + block_size - 1) / block_size; }

// ---------------------------------------------------------------------
// Orquestador del bucle. Cada kernel (Stencil/GEMM/Convolucion) le pasa dos
// callables:
//
//   low_precision_step(T* t_in, const double* comp_in, bool comp_on,
//                       T* t_out, float* comp_out_f)
//     Un paso NORMAL: aplica el operador L de bajo costo (Tensor Cores)
//     sobre t_in (y sobre comp_in truncado a T si comp_on, explotando
//     linealidad -- ver el comentario de cabecera). Escribe t_out y
//     comp_out_f EN FLOAT (nunca en double: eso rompe el gate K=0).
//
//   fp64_step(const double* exact_in, double* out64)
//     Un paso de ANCLA: aplica el MISMO operador L, pero completo en
//     double (sin Tensor Cores). exact_in ya viene reconstruido por
//     reconstruct_exact_kernel.
//
// anchor_every = 0 deshabilita el ancla (comportamiento actual). Con
// anchor_every > 0, la SIEMBRA INICIAL de comp (antes de la primera
// llamada a este bucle) tambien debe hacerse en double genuino si se
// quiere que K=1 sea exacto desde la iteracion 1 -- eso es responsabilidad
// de cada kernel (su propio seed_comp_from_conversion_kernel generalizado,
// ver seccion 01 del plan), no de este orquestador.
// ---------------------------------------------------------------------
template <typename T, typename LowPrecisionStepFn, typename Fp64StepFn, typename ToFloatFn>
void run_chained_precision_loop(int n, int iters, int anchor_every, bool comp_on, T* t_in,
                                 T* t_out, double* comp_in, double* comp_out,
                                 double* exact_scratch, double* out64_scratch,
                                 LowPrecisionStepFn low_precision_step, Fp64StepFn fp64_step,
                                 ToFloatFn tc_to_float, cudaStream_t stream = 0) {
  const int grid = grid_size_for(n);
  const int block = 256;

  for (int iter = 1; iter <= iters; ++iter) {
    const bool is_anchor = (anchor_every > 0) && (iter % anchor_every == 0);

    if (is_anchor) {
      // PASO DE ANCLA (seccion 01/02 del plan): reconstruir sin truncar,
      // evaluar L completo en FP64, re-sembrar sin truncar.
      reconstruct_exact_kernel<T><<<grid, block, 0, stream>>>(t_in, comp_in, comp_on,
                                                                exact_scratch, n, tc_to_float);
      CHECK_CUDA(cudaGetLastError());

      fp64_step(exact_scratch, out64_scratch);

      reseed_from_fp64_kernel<T><<<grid, block, 0, stream>>>(
          out64_scratch, t_out, comp_out, n,
          [] __device__(float v) { return T(v); },  // float_to_tc<T> -- cada kernel puede
                                                       // sustituir por su propia conversion
                                                       // si necesita redondeo especifico.
          tc_to_float);
      CHECK_CUDA(cudaGetLastError());
    } else {
      // PASO NORMAL: sin cambios respecto al comportamiento antes del
      // ancla. comp se calcula en float y solo se ensancha al guardar.
      float* comp_out_f = reinterpret_cast<float*>(out64_scratch);  // reuso de scratch, ver nota abajo
      low_precision_step(t_in, comp_in, comp_on, t_out, comp_out_f);
      widen_comp_to_double_kernel<<<grid, block, 0, stream>>>(comp_out_f, comp_out, n);
      CHECK_CUDA(cudaGetLastError());
    }

    // Ping-pong de punteros host-side (no de memoria): el llamador es
    // dueno de los cuatro buffers y decide si reutiliza t_in/t_out o
    // asigna dos pares fijos y los intercambia aqui.
    T* t_tmp = t_in;
    t_in = t_out;
    t_out = t_tmp;
    double* comp_tmp = comp_in;
    comp_in = comp_out;
    comp_out = comp_tmp;
  }
}

}  // namespace chained_precision

// ---------------------------------------------------------------------
// NOTA DE IMPLEMENTACION (leer antes de compilar):
//
// 1. `comp_out_f` reutiliza `out64_scratch` reinterpretado como float* en
//    el paso normal. Es valido en tamano (out64_scratch tiene n doubles =
//    2n floats, y solo se necesitan n floats) pero es una decision de
//    ahorro de memoria que vale la pena revisar: si per formance de un
//    kernel se ve rara, es el primer sitio a sospechar. La alternativa mas
//    simple (y mas facil de razonar) es un buffer float dedicado de n
//    elementos -- cambialo si la claridad importa mas que los bytes en tu
//    caso.
// 2. `float_to_tc<T>` se pasa como lambda `[] __device__(float v){return T(v);}`
//    en reseed_from_fp64_kernel. Si el tipo T (__half / __nv_bfloat16) no
//    tiene un constructor implicito valido desde float en tu version del
//    toolkit, reemplazalo por __float2half / __float2bfloat16 segun T.
// 3. Este archivo NO se ha compilado ni ejecutado (este entorno no tiene
//    GPU/nvcc). Antes de usarlo en cualquier campana, correr los tres
//    gates de la seccion 01/02 del plan (K=1 igual a FP64, K=0 bit-identico
//    al comportamiento sin ancla, y el gate de encadenamiento contra la
//    ruta 4 de un solo disparo). Si alguno falla, este archivo es el
//    primer sospechoso.
// ---------------------------------------------------------------------
