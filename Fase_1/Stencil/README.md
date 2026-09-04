# Fase_1/Stencil

Baseline del stencil 2D de 5 puntos: CPU serial vs. GPU CUDA clasico (sin Tensor Cores). Es el punto de partida numerico y de rendimiento contra el que se comparan las rutas Tensor Core de Fase 2.

Migrado desde `old/Fase_1/Stencil2D/stencil2d_baseline.cu` — mismo esquema numerico, sin cambios de logica. Ver las notas de migracion al inicio del `.cu` para el detalle de que cambio (nombres/organizacion) y que no (aritmetica).

## Que hace

`stencil_baseline.cu` compara, para un mismo tipo `T` (`float` o `double`, segun `--double`):

1. **CPU serial**: aplica el stencil `iters` veces sobre la misma entrada y promedia el tiempo.
2. **GPU CUDA clasico**: el mismo esquema, un hilo por celda, sin Tensor Cores.

El stencil en si es un Laplaciano discreto de 5 puntos:

```
salida(x,y) = 0.25 * (arriba + abajo + izquierda + derecha) - centro
```

Las celdas de borde se copian sin modificar (condicion de frontera "identidad"). Al final se reporta tiempo medio, GFLOP/s, speedup GPU/CPU, error maximo absoluto y error relativo L2 entre ambas rutas.

## Flags de CLI

```
./stencil_baseline [--nx NX] [--ny NY] [--iters I] [--double]
./stencil_baseline [nx] [ny] [iters]      (forma posicional equivalente)
```

| Flag | Default | Significado |
|---|---|---|
| `--nx NX` | 2048 | Ancho de la grilla |
| `--ny NY` | 2048 | Alto de la grilla |
| `--iters I` | 10 | Iteraciones promediadas (CPU y GPU) |
| `--double` | (ausente = FP32) | Usa `double` en vez de `float` |
| `--help`, `-h` | — | Imprime uso y termina |

Ejemplos:

```
./stencil_baseline --nx 1024 --ny 1024 --iters 20
./stencil_baseline --double --nx 2048 --ny 2048 --iters 20
./stencil_baseline 512 512 10
```

## Que produce

Salida por stdout: caracteristicas de la GPU activa, configuracion del experimento, y un bloque de resultados con tiempo medio, GFLOP/s, speedup GPU/CPU, error maximo absoluto (notacion cientifica) y error relativo L2. No escribe archivos: el `.sbatch` redirige stdout/stderr a `logs/` via `#SBATCH --output`/`--error`.

## Compilar

```
nvcc -std=c++17 -O3 stencil_baseline.cu -o stencil_baseline \
     -gencode arch=compute_80,code=sm_80
```

## El `.sbatch`: de un tamano fijo a un barrido parametrizado

El `.sbatch` historico de Fase 1 (`old/Fase_1/Stencil2D/run_stencil_fase1.sbatch`) corria un **unico tamano fijo, 4096x4096** — muy por encima del rango **512²-2048²** que el plan de tesis promete para esta fase (esa malla grande se eligio en su momento para que los tiempos fueran comparables con los `.sbatch` de Fase 2/Fase 3, que tambien la usaban).

`run_stencil_fase1.sbatch` (este directorio) reemplaza ese tamano unico por un **barrido parametrizado**: nada de tamanos, iteraciones o precisiones esta fijo en el script, todo se controla por variable de entorno con el patron `VAR="${VAR:-default}"` ya usado en el resto del proyecto.

| Variable | Default | Significado |
|---|---|---|
| `STENCIL_SIZES` | `"512 1024 2048"` | Tamanos NX=NY (grilla cuadrada) a barrer, separados por espacio |
| `STENCIL_ITERS` | `20` | Iteraciones promediadas por corrida |
| `STENCIL_PRECISIONS` | `"fp32 fp64"` | Precisiones a correr por cada tamano |
| `OUT_DIR` | `logs` | Directorio para logs/artefactos generados por el script |
| `CUDA_ARCH` | `80` | Arquitectura CUDA objetivo (A100=80, RTX 3050=86, V100=70) |
| `NVCC` | ruta del HPC SDK del cluster | Compilador a usar |
| `SMOKE_TEST` | `0` | Si es `1`, corre un unico tamano pequeno (512, fp32, 3 iters) para validar rapido que el binario compila y ejecuta |

Ejemplos:

```
sbatch run_stencil_fase1.sbatch
sbatch --export=ALL,SMOKE_TEST=1 run_stencil_fase1.sbatch
sbatch --export=ALL,STENCIL_SIZES="1024 2048",STENCIL_ITERS=30 run_stencil_fase1.sbatch
sbatch --export=ALL,STENCIL_PRECISIONS=fp32 run_stencil_fase1.sbatch
```

Si se quiere reproducir el tamano historico de 4096x4096 para comparar contra corridas previas, basta con `STENCIL_SIZES=4096`.
