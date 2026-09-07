# old/ — código anterior a la reconstrucción

Snapshot completo del proyecto antes de la reorganización descrita en el documento **Plan de Precisión Mixta**. Se conserva por dos razones:

1. **Referencia de código ya probado.** Fase 1 y Fase 2 (línea base y activación de Tensor Cores) funcionan correctamente según la auditoría previa — la reconstrucción en la raíz del repositorio los migra y limpia, no los reescribe desde cero. Si algo en la versión nueva se comporta distinto, este es el punto de comparación.
2. **Historial de Fase 3/4 de Stencil.** `old/Fase_3/Stencil` y `old/Fase_4/Stencil` contienen la implementación de drift, Kahan, compensación espacial y las optimizaciones de kernel (tiling, `ldmatrix`, swizzle, CUDA Graphs) ya validadas con gates de regresión — la base sobre la que se construye el ancla FP64 nueva.

## Origen de cada carpeta

- `old/Fase_1`, `old/Fase_2`, `old/Fase_3`, `old/tools`: movidos directamente desde la raíz del repositorio (rama `main`) con `git mv`, así que conservan su historial de commits completo.
- `old/Fase_4`: **no existía en `main`** — se desarrolló en la rama `fase4-estadistica-variabilidad`, nunca fusionada. Se extrajo aquí con `git archive origin/fase4-estadistica-variabilidad Fase_4`, así que el contenido es el snapshot final de esa rama, pero **no conserva el historial de commits** de esa rama (para eso, `git log origin/fase4-estadistica-variabilidad -- Fase_4` sigue funcionando directamente sobre la rama).

## Qué NO hacer con esta carpeta

No compilar ni ejecutar nada de aquí como si fuera el código vigente — es historial, no el punto de partida para trabajo nuevo. El código activo vive en `Fase_1/` a `Fase_4/` en la raíz del repositorio.
