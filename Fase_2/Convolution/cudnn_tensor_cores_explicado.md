# Convolucion 2D con cuDNN y Tensor Cores: explicacion desde cero

## Tabla de contenido

1. [La operacion matematica de convolucion 2D](#1-la-operacion-matematica-de-convolucion-2d)
2. [De convolucion a multiplicacion de matrices: im2col](#2-de-convolucion-a-multiplicacion-de-matrices-im2col)
3. [Que es cuDNN y que problema resuelve](#3-que-es-cudnn-y-que-problema-resuelve)
4. [Descriptores: el lenguaje de cuDNN](#4-descriptores-el-lenguaje-de-cudnn)
5. [Algoritmos de convolucion en cuDNN](#5-algoritmos-de-convolucion-en-cudnn)
6. [El workspace: memoria temporal de trabajo](#6-el-workspace-memoria-temporal-de-trabajo)
7. [Arquitectura de Tensor Cores](#7-arquitectura-de-tensor-cores)
8. [Precision mixta en cuDNN: FP16 entrada, FP32 acumulacion](#8-precision-mixta-en-cudnn-fp16-entrada-fp32-acumulacion)
9. [Los tres pasos para activar Tensor Cores en cuDNN](#9-los-tres-pasos-para-activar-tensor-cores-en-cudnn)
10. [Formato NCHW vs NHWC y su impacto en Tensor Cores](#10-formato-nchw-vs-nhwc-y-su-impacto-en-tensor-cores)
11. [Flujo completo de la API paso a paso](#11-flujo-completo-de-la-api-paso-a-paso)
12. [Error numerico esperado con precision mixta](#12-error-numerico-esperado-con-precision-mixta)
13. [Verificacion: como saber si los Tensor Cores se usaron](#13-verificacion-como-saber-si-los-tensor-cores-se-usaron)

---

## 1. La operacion matematica de convolucion 2D

Una convolucion 2D hacia adelante (*forward pass*) calcula, para cada imagen del batch, cada filtro de salida y cada posicion espacial, una suma ponderada sobre una ventana de la imagen de entrada:

```
Y[n, k, oh, ow] = sum_{c=0}^{C-1} sum_{r=0}^{R-1} sum_{s=0}^{S-1}
                    X[n, c, oh*stride_h + r*dil_h - pad_h,
                           ow*stride_w + s*dil_w - pad_w]
                  * W[k, c, r, s]
```

Donde:

| Simbolo | Significado                                      |
|---------|--------------------------------------------------|
| N       | Tamano del batch (numero de imagenes)            |
| C       | Canales de entrada (e.g. RGB = 3)                |
| H, W    | Alto y ancho de la imagen de entrada             |
| K       | Numero de filtros (canales de salida)            |
| R, S    | Alto y ancho de cada filtro                      |
| pad     | Relleno con ceros en los bordes                  |
| stride  | Paso del desplazamiento de la ventana            |
| dilation| Espacio entre elementos del filtro (atrous conv) |
| outH, outW | Dimensiones de la salida (calculadas abajo) |

Las dimensiones de la salida se calculan con:

```
outH = floor((H + 2*pad_h - dil_h*(R-1) - 1) / stride_h) + 1
outW = floor((W + 2*pad_w - dil_w*(S-1) - 1) / stride_w) + 1
```

**Coste computacional**: cada posicion de salida (n, k, oh, ow) requiere C*R*S multiplicaciones y C*R*S sumas. El total de operaciones de punto flotante es:

```
FLOPs = 2 * N * K * outH * outW * C * R * S
```

El factor 2 cuenta multiplicacion y suma como operaciones separadas (convencion estandar en HPC).

---

## 2. De convolucion a multiplicacion de matrices: im2col

La convolucion directa tiene patrones de acceso a memoria irregulares que dificultan su paralelizacion eficiente. La transformacion **im2col** (*image to column*) resuelve esto: reescribe la convolucion como una multiplicacion de matrices densa (GEMM) con accesos secuenciales.

### La idea

Para cada imagen del batch, se construye una matriz `col` donde **cada columna contiene todos los elementos de la ventana receptiva** correspondientes a una posicion de salida (oh, ow):

```
col tiene forma: [C*R*S, outH*outW]
```

Cada columna de `col` es la "ventana aplanada" centrada en (oh, ow). Los pixels fuera del borde (padding) se rellenan con cero.

El filtro `W` se ve como una matriz de forma `[K, C*R*S]`: cada fila es un filtro aplanado.

La convolucion se convierte en:

```
Y[K, outH*outW] = W[K, C*R*S] * col[C*R*S, outH*outW]
```

Esto es exactamente una GEMM, que BLAS y cuBLAS ejecutan de forma altamente optimizada.

### Costo de im2col

- Crea una copia de los datos de entrada con factor de expansion `R*S` (para stride=1, pad=0).
- Requiere memoria adicional proporcional a `C*R*S*outH*outW` elementos.
- Para filtros grandes (R=S=11) el overhead de memoria puede ser significativo.

cuDNN implementa variantes de este enfoque *in-place* para evitar la copia explicita (algoritmos `IMPLICIT_GEMM` e `IMPLICIT_PRECOMP_GEMM`).

---

## 3. Que es cuDNN y que problema resuelve

**cuDNN** (CUDA Deep Neural Network library) es la biblioteca de NVIDIA que provee implementaciones altamente optimizadas de operaciones usadas en redes neuronales: convoluciones, pooling, normalizacion por batch, activaciones, etc.

### Por que no simplemente usar cuBLAS

Para GEMM pura, cuBLAS es suficiente. Para convolucion 2D, el problema es mas complejo:

1. **Multiples algoritmos con distintos trade-offs**: el rendimiento optimo depende de las dimensiones exactas (N, C, H, W, K, R, S), el hardware disponible y la memoria GPU. No existe un unico mejor algoritmo universal.

2. **Autotuning**: cuDNN puede probar varios algoritmos y elegir el mas rapido para una configuracion dada en tiempo de ejecucion.

3. **Soporte de formatos de tensor**: cuDNN abstrae los formatos de memoria (NCHW, NHWC) y los tipos de datos (FP32, FP16, FP64, INT8) separadamente de los algoritmos.

4. **Integracion con Tensor Cores**: cuDNN expone controles especificos para forzar o prohibir el uso de Tensor Cores via descriptores.

### El modelo conceptual de cuDNN

cuDNN funciona con **descriptores**: objetos opacos que describen la forma y tipo de cada tensor o parametro. La separacion entre *descripcion* (descriptores) y *ejecucion* (llamada a la convolucion) permite que cuDNN elija la implementacion optima internamente.

---

## 4. Descriptores: el lenguaje de cuDNN

Antes de lanzar cualquier operacion, hay que describir todos los tensores y parametros mediante descriptores. cuDNN tiene tres tipos principales para convolucion:

### 4.1 `cudnnTensorDescriptor_t`

Describe un tensor de datos (entrada o salida). Se configura con:

```c
cudnnTensorDescriptor_t xDesc;
cudnnCreateTensorDescriptor(&xDesc);
cudnnSetTensor4dDescriptor(
    xDesc,
    CUDNN_TENSOR_NCHW,   // formato de memoria
    CUDNN_DATA_FLOAT,    // tipo de dato
    N, C, H, W           // dimensiones
);
```

Los formatos disponibles son `CUDNN_TENSOR_NCHW` y `CUDNN_TENSOR_NHWC`. Ver seccion 10 para la diferencia.

### 4.2 `cudnnFilterDescriptor_t`

Describe el tensor de pesos (filtros). Similar al tensor pero con su propio tipo:

```c
cudnnFilterDescriptor_t wDesc;
cudnnCreateFilterDescriptor(&wDesc);
cudnnSetFilter4dDescriptor(
    wDesc,
    CUDNN_DATA_FLOAT,    // tipo de dato del filtro
    CUDNN_TENSOR_NCHW,   // formato
    K, C, R, S           // [filtros, canales, alto, ancho]
);
```

### 4.3 `cudnnConvolutionDescriptor_t`

Describe los hiperparametros de la convolucion: padding, stride, dilation, tipo de operacion y **tipo de computo**. Este ultimo es el mas importante para precision mixta:

```c
cudnnConvolutionDescriptor_t convDesc;
cudnnCreateConvolutionDescriptor(&convDesc);
cudnnSetConvolution2dDescriptor(
    convDesc,
    pad_h, pad_w,             // padding
    stride_h, stride_w,       // stride
    dilation_h, dilation_w,   // dilation
    CUDNN_CROSS_CORRELATION,  // tipo: cross-correlation (no convolucion matematica pura)
    CUDNN_DATA_FLOAT          // computeType: tipo de la acumulacion
);
```

**`CUDNN_CROSS_CORRELATION` vs `CUDNN_CONVOLUTION`**: en redes neuronales siempre se usa cross-correlation (el filtro NO se voltea). La convolucion matematica pura requeriria voltear el filtro 180 grados. cuDNN llama "convolucion" a ambas operaciones pero distingue la orientacion con este flag.

**`computeType`**: define la precision de los acumuladores internos. Para precision mixta FP16+FP32 se pone `CUDNN_DATA_FLOAT` aunque los datos de entrada sean FP16.

---

## 5. Algoritmos de convolucion en cuDNN

cuDNN ofrece varios algoritmos, cada uno con diferentes trade-offs de velocidad, uso de memoria y compatibilidad con Tensor Cores:

| Algoritmo | Descripcion | Workspace | TC compatible |
|-----------|-------------|-----------|--------------|
| `IMPLICIT_GEMM` | im2col implicito sin copia adicional | Minimo | Parcialmente |
| `IMPLICIT_PRECOMP_GEMM` | im2col con indices precomputados | Medio | Si |
| `GEMM` | im2col explicito en workspace | Grande | Si |
| `DIRECT` | Kernel directo sin transformacion | Ninguno | No |
| `FFT` | Convolucion via FFT (rapida para filtros grandes) | Muy grande | No |
| `FFT_TILING` | FFT por tiles (para tensores que no caben en mem) | Grande | No |
| `WINOGRAD` | Algoritmo de Winograd (reduce FLOPs para R=S=3) | Pequeno | Si |
| `WINOGRAD_NONFUSED` | Winograd con kernels separados | Medio | Si |

### Seleccion automatica del algoritmo

```c
cudnnConvolutionFwdAlgoPerf_t perf_results[8];
int algo_count = 0;
cudnnGetConvolutionForwardAlgorithm_v7(
    handle, xDesc, wDesc, convDesc, yDesc,
    8,             // maximo de algoritmos a evaluar
    &algo_count,   // cuantos encontro
    perf_results   // resultados ordenados por tiempo estimado
);
// El algoritmo recomendado es perf_results[0].algo
```

cuDNN devuelve los algoritmos ordenados por rendimiento estimado. Con `CUDNN_TENSOR_OP_MATH` activado (ver seccion 9), los algoritmos TC-compatibles tendran prioridad en el ranking.

---

## 6. El workspace: memoria temporal de trabajo

Varios algoritmos necesitan un buffer temporal en la GPU para almacenar datos intermedios (como la matriz `col` del im2col, o los resultados parciales de FFT). Este buffer se llama **workspace**.

```c
size_t ws_bytes = 0;
cudnnGetConvolutionForwardWorkspaceSize(
    handle, xDesc, wDesc, convDesc, yDesc,
    algo,        // el algoritmo elegido
    &ws_bytes    // tamano necesario en bytes
);

void* d_workspace = nullptr;
if (ws_bytes > 0) cudaMalloc(&d_workspace, ws_bytes);
```

El workspace se pasa como argumento a `cudnnConvolutionForward`. Si el algoritmo no necesita workspace, `ws_bytes` sera 0 y se puede pasar `nullptr`.

**Importante**: el workspace es reutilizable entre llamadas con la misma configuracion. En aplicaciones reales (frameworks como PyTorch) se mantiene un pool de workspace preasignado para evitar `cudaMalloc` en cada iteracion de entrenamiento.

---

## 7. Arquitectura de Tensor Cores

Los **Tensor Cores** son unidades de hardware especializadas presentes en GPUs NVIDIA desde Volta (2017). Cada Tensor Core realiza una operacion matricial D = A*B + C en un solo ciclo de reloj, donde:

- `A` es una submatriz de 4x4 (Volta) o 8x4 (Ampere y posteriores)
- `B` es una submatriz de 4x4 o 4x8
- `C` y `D` son acumuladores de 4x4

En Ampere (RTX 3xxx, A100), una operacion Tensor Core tipica es:

```
D[16x16] = A[16x16] (FP16) * B[16x16] (FP16) + C[16x16] (FP32)
```

Esto se llama operacion **warp-level matrix multiply-accumulate** (WMMA). Un warp entero (32 hilos) colabora para ejecutar esta operacion en paralelo.

### Por que son tan rapidos

Las FPUs (Floating Point Units) tradicionales hacen una operacion escalar por ciclo. Un Tensor Core hace 256 operaciones de punto flotante por ciclo (para matrices 16x16 FP16). El pico teorico de una A100 con Tensor Cores es ~312 TFLOP/s en FP16, comparado con ~19.5 TFLOP/s en FP64 sin TC.

### Restricciones de los Tensor Cores

1. **Dimensiones multiplo de 8 (o 16)**: los tiles deben alinearse con las dimensiones de la operacion WMMA. Para FP16, las dimensiones K, N deben ser multiplos de 8 para uso optimo.
2. **Formato de memoria alineado**: los datos deben estar alineados a 128 bytes (o al menos 16 bytes) en memoria.
3. **Solo para operaciones matriciales**: los TC no aceleran operaciones escalares arbitrarias.

---

## 8. Precision mixta en cuDNN: FP16 entrada, FP32 acumulacion

### El problema de solo usar FP16

FP16 tiene un rango dinamico estrecho (~6x10^-5 a ~65504) y epsilon de maquina ~9.77x10^-4. Si la acumulacion de productos parciales tambien ocurriera en FP16, los errores de redondeo se acumularian rapidamente:

- Una suma de 1000 valores FP16 de magnitud 1.0 puede tener error relativo >1%.
- Valores fuera del rango [~6x10^-5, ~65504] causan underflow o overflow.

### La solucion: precision mixta

La estrategia estandar es:
- **Entradas y pesos**: FP16 (mitad de memoria, mayor ancho de banda, TC-compatible).
- **Acumulacion interna**: FP32 (cada producto FP16*FP16 se suma a un acumulador de 32 bits).
- **Salida**: FP32 (se puede reconvertir a FP16 manualmente si el siguiente paso lo acepta).

Matematicamente, para cada elemento de la salida:

```
Y[k, oh, ow] = sum_i  float32( half16(X_i) * half16(W_i) )
                                ^-TC opera en FP16-^
               ^------------acumulador en FP32-----------^
```

Este esquema es el que usan PyTorch, TensorFlow y todos los frameworks modernos en modo `autocast` o `mixed precision`.

### Como cuDNN implementa esto internamente

Cuando se especifica `computeType = CUDNN_DATA_FLOAT` con tensores FP16, cuDNN:

1. Carga los datos X y W como FP16 desde VRAM (mitad del ancho de banda usado).
2. Los Tensor Cores multiplican tiles FP16 x FP16.
3. Los productos parciales se acumulan en registros FP32.
4. El resultado final en FP32 se escribe en el tensor de salida Y.

---

## 9. Los tres pasos para activar Tensor Cores en cuDNN

### Paso 1: descriptores de entrada y filtro con `CUDNN_DATA_HALF`

```c
// Descriptor de la activacion de entrada: FP16
cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, C, H, W);

// Descriptor del tensor de filtros: FP16
cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, K, C, R, S);
```

Esto le dice a cuDNN que los datos en GPU son de 16 bits. Los buffers de dispositivo deben ser `__half*`, no `float*`.

### Paso 2: `computeType = CUDNN_DATA_FLOAT` en el descriptor de convolucion

```c
cudnnSetConvolution2dDescriptor(
    convDesc,
    pad_h, pad_w,
    stride_h, stride_w,
    dilation_h, dilation_w,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT          // acumulacion en FP32
);
```

El `computeType` define la precision del acumulador, independientemente del tipo de los operandos. `CUDNN_DATA_FLOAT` aqui es la clave para la precision mixta: los TC multiplican en FP16 pero acumulan en FP32.

### Paso 3: `cudnnSetConvolutionMathType` con `CUDNN_TENSOR_OP_MATH`

```c
cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
```

Esta es la llamada mas importante y la que mas frecuentemente se omite por error. Sin ella, cuDNN puede ignorar los Tensor Cores incluso si los datos son FP16 y el hardware los soporta.

`CUDNN_TENSOR_OP_MATH` le comunica al planificador de cuDNN que:
- Tiene permiso de usar unidades Tensor Core para esta convolucion.
- Puede elegir algoritmos que requieran convertir o reordenar datos internamente si eso permite usar TC.
- Priorizara algoritmos TC-compatibles en el ranking de `cudnnGetConvolutionForwardAlgorithm_v7`.

El valor alternativo `CUDNN_DEFAULT_MATH` desactiva los TC (o los usa solo oportunisticamente en versiones antiguas). Desde cuDNN 8.x existe tambien `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` que permite a cuDNN convertir datos FP32 a FP16 internamente si eso activa TC.

### Descriptor de salida: FP32

```c
// La salida es FP32: la acumulacion ya se hizo en FP32, se escribe directo.
cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW);
```

El buffer de salida `d_y` en GPU es `float*`. cuDNN escribe el resultado del acumulador FP32 directamente ahi.

### Resumen visual

```
CPU (FP32) --> [kernel conversion FP32->FP16] --> GPU (FP16)
                                                      |
                                    X_fp16 --------> [Tensor Core]
                                    W_fp16 --------> [  FP16*FP16  ] --> acum FP32 --> Y_fp32
                                                      |
                                                   GPU (FP32) --> CPU (FP32)
```

---

## 10. Formato NCHW vs NHWC y su impacto en Tensor Cores

El formato de memoria define como se ordenan los bytes de un tensor 4D en memoria lineal.

### NCHW (channel-first)

El indice de canal varia mas lentamente. Para un pixel (n, c, h, w):

```
offset = n*(C*H*W) + c*(H*W) + h*W + w
```

Todos los valores de un mismo canal de una imagen son contiguos en memoria. Es el formato historico de cuDNN y el default en PyTorch en CPU.

### NHWC (channel-last)

El indice de canal varía más rapidamente. Para un pixel (n, c, h, w):

```
offset = n*(H*W*C) + h*(W*C) + w*C + c
```

Todos los canales de un mismo pixel son contiguos. Es el formato preferido por TensorFlow y por los Tensor Cores.

### Por que NHWC es mejor para Tensor Cores

Los Tensor Cores operan sobre tiles matriciales. En una convolucion implementada como GEMM, la dimension `K` (canales de entrada * elementos del filtro) debe ser contigua en memoria para que los tiles se carguen eficientemente en los registros de la GPU.

Con formato NHWC, los `C` canales de cada posicion espacial son contiguos, lo que alinea naturalmente con como los TC consumen los datos. Con NCHW, se necesitan transpociones o *strides* no unitarios que reducen la eficiencia del ancho de banda.

**En la practica**: para filtros 3x3 y dimensiones multiplo de 8, NHWC con TC puede ser 1.5x-3x mas rapido que NCHW con TC. Sin embargo, NCHW con TC sigue siendo mucho mas rapido que NCHW sin TC.

En el codigo de este proyecto se usa NCHW para mantener consistencia con la referencia CPU (im2col natural en NCHW). Para maximizar el rendimiento en produccion, se recomienda NHWC.

---

## 11. Flujo completo de la API paso a paso

A continuacion se muestra el flujo completo para una convolucion FP16->FP32 con Tensor Cores:

```c
// --- 1. Crear el handle de cuDNN (una vez por programa) ---
cudnnHandle_t handle;
cudnnCreate(&handle);

// --- 2. Preparar datos FP16 en GPU ---
// Primero copiar FP32 a GPU, luego convertir con kernel propio
__half* d_x_fp16 = upload_and_convert_to_half(x_host);
__half* d_w_fp16 = upload_and_convert_to_half(w_host);
float*  d_y;
cudaMalloc(&d_y, y_count * sizeof(float));

// --- 3. Crear y configurar descriptores ---
cudnnTensorDescriptor_t xDesc, yDesc;
cudnnFilterDescriptor_t wDesc;
cudnnConvolutionDescriptor_t convDesc;

cudnnCreateTensorDescriptor(&xDesc);
cudnnCreateTensorDescriptor(&yDesc);
cudnnCreateFilterDescriptor(&wDesc);
cudnnCreateConvolutionDescriptor(&convDesc);

// Entrada FP16, formato NCHW
cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, C, H, W);

// Filtro FP16, formato NCHW
cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, K, C, R, S);

// Convolucion: acumulacion FP32 (precision mixta)
cudnnSetConvolution2dDescriptor(convDesc,
    pad_h, pad_w, stride_h, stride_w, dil_h, dil_w,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT);           // <-- acumulador FP32

// ACTIVACION EXPLICITA DE TENSOR CORES
cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);

// Salida FP32
cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                            outN, outC, outH, outW);

// --- 4. Seleccionar algoritmo (cuDNN elige el mas rapido con TC) ---
cudnnConvolutionFwdAlgoPerf_t perf[8];
int count = 0;
cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc,
                                        8, &count, perf);
cudnnConvolutionFwdAlgo_t algo = perf[0].algo;

// --- 5. Reservar workspace ---
size_t ws_bytes = 0;
cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                         algo, &ws_bytes);
void* d_ws = nullptr;
if (ws_bytes > 0) cudaMalloc(&d_ws, ws_bytes);

// --- 6. Ejecutar la convolucion ---
float alpha = 1.0f, beta = 0.0f;
cudnnConvolutionForward(handle,
    &alpha,
    xDesc, d_x_fp16,        // entrada FP16
    wDesc, d_w_fp16,        // filtro FP16
    convDesc, algo,
    d_ws, ws_bytes,
    &beta,
    yDesc, d_y);             // salida FP32

// --- 7. Limpiar ---
cudaFree(d_ws);
cudaFree(d_x_fp16);
cudaFree(d_w_fp16);
cudaFree(d_y);
cudnnDestroyTensorDescriptor(xDesc);
cudnnDestroyTensorDescriptor(yDesc);
cudnnDestroyFilterDescriptor(wDesc);
cudnnDestroyConvolutionDescriptor(convDesc);
cudnnDestroy(handle);
```

---

## 12. Error numerico esperado con precision mixta

### Fuentes de error

Al usar FP16 para los operandos, se introducen dos tipos de error:

1. **Error de conversion (cuantizacion)**: cada valor FP32 que se convierte a FP16 sufre un truncamiento de mantisa. La epsilon de maquina de FP16 es ~9.77e-4, versus ~1.19e-7 para FP32. Esto significa que la representacion FP16 de un numero puede diferir del original en hasta ~0.1%.

2. **Error de acumulacion**: aunque el acumulador es FP32, cada producto `half*half` se suma a el con precision FP32. Los errores de cuantizacion en los operandos se propagan sumados.

### Magnitudes tipicas

Para convolucion 2D con valores de activacion en el rango [-2, 2] y pesos en [-1, 1]:

| Metrica | Valor tipico TC (FP16+FP32) vs CPU (FP32) |
|---------|------------------------------------------|
| Error maximo absoluto | 0.01 - 1.0 (depende de C*R*S) |
| Error relativo L2 | 1e-3 - 1e-2 |

El error crece aproximadamente como `sqrt(C*R*S) * epsilon_FP16` en el peor caso. Para C=64, R=S=3 esto da ~9e-3, consistente con los valores observados.

### Cuando es aceptable

En el contexto del proyecto de grado, el criterio a evaluar es si el error se mantiene dentro del dominio de aceptabilidad para la aplicacion:

- **Inferencia en DNN**: error relativo <1% generalmente aceptable.
- **Metodo iterativo HPC** (e.g. solucion de EDPs): depende de la tasa de convergencia. Un error de 1e-3 por paso puede acumularse a lo largo de miles de iteraciones.
- **Simulacion numerica de referencia**: generalmente requiere error <1e-6, por lo que FP16 puro no es viable. La precision mixta FP16+FP32 (con acumulador FP32) puede ser suficiente para ciertos algoritmos.

---

## 13. Verificacion: como saber si los Tensor Cores se usaron

### Metodo 1: Nsight Compute (ncu)

```bash
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed \
    ./conv_tc --N 1 --C 64 --H 224 --W 224 --K 64 --R 3 --S 3 --iters 1
```

Un porcentaje alto en `sm__inst_executed_pipe_tensor` confirma uso de Tensor Cores. Valores cercanos a 0% indican que no se usaron.

### Metodo 2: Nsight Systems (nsys)

```bash
nsys profile --trace=cuda,cudnn ./conv_tc
```

Abrir el perfil en la GUI de Nsight Systems. Los kernels que usan TC tienen nombres que contienen `h884` (Volta/Turing) o `hmma` (Ampere) en su nombre interno. Los kernels FP32 sin TC tendran nombres como `sgemm` o `conv2d`.

### Metodo 3: verificacion por speedup

Si el speedup de la ruta TC vs FP32 clasico es significativamente mayor a 1x (tipicamente 2x-4x para tamanos adecuados), es una indicacion fuerte de que los TC se activaron. Si ambas rutas tienen rendimiento similar, probablemente los TC no se usaron.

### Causas comunes de que los TC no se activen

1. **Dimensiones no alineadas**: C, K deben ser multiplos de 8 para FP16 (multiplos de 4 para TF32 en Ampere).
2. **Falta de `cudnnSetConvolutionMathType`**: sin esta llamada cuDNN puede usar el camino FP32 estandar.
3. **Algoritmo incompatible**: algunos algoritmos (como `DIRECT` o `FFT`) no usan TC. `cudnnGetConvolutionForwardAlgorithm_v7` debe elegir `IMPLICIT_PRECOMP_GEMM` o `WINOGRAD` para TC.
4. **Compute Capability < 7.0**: los TC requieren Volta (7.0) o superior. Las GPUs Pascal (6.x) y anteriores no los tienen.
5. **cuDNN version antigua**: versiones < 7.0 tienen soporte TC limitado.

### Consultar el algoritmo seleccionado

```c
cudnnConvolutionFwdAlgoPerf_t perf[8];
int count = 0;
cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc,
                                        8, &count, perf);
for (int i = 0; i < count; ++i) {
    printf("Algoritmo %d: tiempo=%.3f ms, mathType=%s\n",
           perf[i].algo,
           perf[i].time,
           perf[i].mathType == CUDNN_TENSOR_OP_MATH ? "TENSOR_OP" : "DEFAULT");
}
```

El campo `mathType` de cada `cudnnConvolutionFwdAlgoPerf_t` indica si ese algoritmo especifico usa Tensor Cores (`CUDNN_TENSOR_OP_MATH`) o no (`CUDNN_DEFAULT_MATH`).
