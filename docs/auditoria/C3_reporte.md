# Auditoría C3: Pareto de Convolución 2D + C14 (conteo de criterios)

Hallazgo C3 (crítico) y C14 (menor) de la revisión del informe final. Rama:
`auditoria-c3-pareto-conv`, basada en `main` (ya incluye C1 y C2).

**Ningún job se relanzó.** Todo el análisis usa resultados ya descargados en
`Fase_4/analysis/out/` (tabla canónica generada por `build_canonical.py`,
job 7145) y las figuras/CSV de `fig6_pareto.py`/`fig9_pareto2d_color.py`.

## 1. Inventario textual (prosa viva, no comentarios)

| Línea aprox. | Sección | Afirmación |
| - | - | - |
| 1136-1146 | Alcances y Limitaciones | "El frente de Pareto ... se construyeron también sobre los tres [kernels], con una cobertura desigual: dos tamaños en GEMM, cuatro en Convolución~2D y dos mallas en Stencil~2D." |
| 2413-2417 | `subsubsec:res-criterios-pareto` | No cita ningún número de criterios (solo remite a la Tabla~\ref{tab:reglas-pareto}). |
| 2432-2446 (`tab:estratos-pareto`) | ídem | Conv.~2D: "Admitidos los cuatro; solo candidatas BF16 (FP16 no finita en el horizonte)". |
| 2497-2519 (`subsubsec:res-pareto-conv`, fig. `fig:f4-pareto-conv`) | ídem | "para los cuatro estratos admitidos"; "Convolución~2D admite cuatro estratos, el doble que GEMM". |
| 2706-2716 (Discusión general) | ídem | "El análisis multiobjetivo ... pudo construirse en los tres kernels ... En los tres, tiempo y energía resultaron casi colineales." |
| 2945-2955 (Conclusiones) | ídem | "En GEMM y Convolución~2D las rutas sin compensación ocupan el extremo barato del frente y las compensadas el caro." |

**Una sola versión en toda la prosa viva**: los tres kernels incluidos, Convolución~2D
con cuatro estratos admitidos y frente construido. No se encontró ninguna línea viva
con la versión contradictoria ("resultó viable solo en GEMM" / "todos sus estratos se
descartaron"); esa versión solo persiste dentro de los tres bloques de comentario
listados en el contexto (confirmado por grep, sección 5).

## 2. Verificación contra los resultados descargados

### 2.a Conteo real de estratos admitidos (Convolución 2D)

Fuente: `Fase_4/analysis/out/audit_report.md` (generado por `build_canonical.py`,
"Estado: todos los asserts pasan"), tabla "Conteos por kernel x size x ruta":

| size | route | filas | finitas |
| - | - | - | - |
| 64/128/256/512 | `BF16_comp` | 12 | **12** (todas finitas) |
| 64/128/256/512 | `BF16_none` | 3 | **3** (todas finitas) |
| 64/128/256/512 | `FP16_comp` | 12 | **0** (ninguna finita) |
| 64/128/256/512 | `FP16_none` | 3 | **0** (ninguna finita) |
| 64/128/256/512 | `GPU_FP64` | 3 | 0 (referencia, no compite) |

Esto se repite **idéntico en los cuatro tamaños**. Confirma, sin ambigüedad:
cuatro estratos (HW=64,128,256,512), y en los cuatro **todas las candidatas FP16
son no finitas y todas las BF16 son finitas** — exactamente "Admitidos los
cuatro; solo candidatas BF16 (FP16 no finita en el horizonte)" de
`tab:estratos-pareto`. `F6_exclusions.csv` corrobora: las 60 filas de exclusión
`non_finite` en `conv` son las `FP16_comp`/`FP16_none` de los cuatro tamaños
(15 por tamaño = 12+3), y ninguna fila BF16 aparece excluida por esa razón.

`Fase_4/analysis/out/tables/F6_points.csv` (los puntos que efectivamente entran
a la figura) confirma lo mismo desde el otro extremo: para `kernel=conv` solo
existen filas con `format` en `{BF16, FP64}` (24 filas: 5 BF16 + 1 FP64\_ref por
tamaño × 4 tamaños); **cero filas `format=FP16`**.

### 2.b Frente realmente construido, y misma campaña que la figura del documento

`fig6_pareto.py` (función `build_points`) construye el frente con
`pareto_mask`/`front_layers` sobre `[T_ms, E_J, err_obj]` por estrato, y
`EXPECTED[("conv", 40)] = {64: 5, 128: 5, 256: 5, 512: 5}` — cinco candidatas
por tamaño. `F6_points.csv` muestra que esas cinco son, para cada tamaño,
`BF16_comp` en $K\in\{0,1,5,20\}$ (cuatro periodicidades de anclaje) más
`BF16_none` (`K=0`, sin anclaje): exactamente las candidatas BF16 finitas de
2.a, ninguna FP16.

Los `.caption.txt` generados junto a las figuras (`F6_pareto_conv_h40_modoA/B`,
`F9_pareto2d_conv`) declaran explícitamente "Convolución, h=40, producción
(**job 7145**)" y "triples por tamaño: 64: 5, 128: 5, 256: 5, 512: 5 ... Modo A:
solo candidatos FP16/BF16" — la misma campaña, el mismo job, y coincide con el
conteo de 2.a.

**El PNG embebido en el documento** (`tesis/figuras/fase4/Pareto-Conv-F4.png`,
7030×1881 px) no coincide en sha256 ni en dimensiones con ningún archivo
generado en `Fase_4/analysis/out/figures/` (`F6_pareto_conv_h40_modoA/B.png`,
4109×5048; `F9_pareto2d_conv.png`, 4113×3381) ni con versiones más antiguas
encontradas localmente (`pareto_out/pareto_conv.png`,
`analisis_fase4/figuras/F7_...png`). Es más reciente que `F9_pareto2d_conv.png`
(timestamp posterior) y con un *layout* propio (cuatro paneles lado a lado, uno
por tamaño), distinto de los `F6`/`F9` archivados, así que es una exportación
separada (probablemente un script de figuras de tesis no incluido en
`Fase_4/analysis/`, no localizado). **Se inspeccionó visualmente el PNG
embebido** (herramienta de lectura de imágenes) en vez de solo su hash, y
muestra: cuatro paneles (N=64/128/256/512, con las iteraciones de la pasada de
energía: 37 000 o 2 500, tal como dice el párrafo en la línea ~2410),
únicamente marcadores BF16 (diamante = sin compensación, triángulo = con
compensación, **cero marcadores FP16**), una línea discontinua etiquetada
"Frente no dominado (3 objetivos)" conectando puntos reales, y una barra de
color de error $L_2$ relativo en el rango ~5×10⁻⁴–4,5×10⁻³ (consistente con los
valores de `rel_l2` de las filas BF16 en `F6_points.csv`, p. ej. `0,00165798`
para `BF16_comp K=0` en HW=64). **Conclusión:** el contenido de la figura
coincide, panel por panel, con los datos canónicos de 2.a/2.b; no se pudo
confirmar la procedencia exacta del archivo (qué script lo generó
literalmente), pero sí se confirmó, por inspección directa de sus datos
graficados, que corresponde a la misma campaña (Convolución 2D, cuatro
tamaños, solo BF16, job 7145) y no a otro operador o campaña.

### 2.c Inclusión en el análisis multiobjetivo final

Confirmado: `Fase_4/analysis/out/audit_report.md` procesa GEMM, Convolution y
Stencil con el mismo pipeline (`build_canonical.py`) y los mismos asserts
("PASS: conv: ..." aparece en paridad con "PASS: gemm: ..." y "PASS: stencil:
..." en todo el reporte), y `fig6_pareto.py`/`EXPECTED` incluye
`("conv", 40)` junto a `("gemm", 40)` y `("stencil", 50/10)` en el mismo
diccionario, ejecutado en una sola corrida (`build_points()` itera sobre los
tres). Convolución~2D **sí** se integró al análisis multiobjetivo final junto
con GEMM (y Stencil), como dice hoy el documento.

## 3. Verificación del conteo de criterios (C14)

`tab:reglas-pareto` tiene **7 filas por encima del `\midrule`** ("Identidad de
la configuración", "Horizonte común del error", "Tasas por iteración",
"Fiabilidad de la energía", "Validez del error terminal", "Referencias",
"Estratos comparables") más una fila "Dominancia" después del `\midrule`, que
es el paso de dominancia en sí, no un criterio de admisión. El texto que
antecede la tabla (línea 2364) **no menciona ningún número** ("aplica los
criterios de la Tabla~\ref{tab:reglas-pareto}"): no hay discrepancia que
corregir en la prosa viva; la mención a "nueve criterios" solo sobrevive dentro
de los comentarios ya marcados para retirar (sección 5).

Contraste contra el pipeline real (`build_canonical.py`, `fig6_pareto.py`):

| Criterio (fila de la tabla) | Mecanismo de código encontrado |
| - | - |
| Identidad de la configuración | `build_canonical.py`: clave física `['kernel','size','route','format','compensation','K_efectivo','iters_num']`, verificada por el assert "unicidad de la clave fisica" |
| Horizonte común del error | `fig6_pareto.py`: `df[(df.iters_num == h) & ...]`, mismo `h` para todas las candidatas del estrato |
| Tasas por iteración | `fig6_pareto.py`: `T_ms, E_J = t_iter_ms_energy, energy_gpu_j_per_iter` |
| Fiabilidad de la energía | `sub.energy_reliable` (calculado en `build_canonical.py` desde `window_reliable`/`energy_gpu_j>0`) |
| Validez del error terminal | `sub.fin` (`solution_finite`) y `sub.rel_l2 > 0` |
| Referencias | `refs = sub[sub.is_reference & ...]`, excluidas de `cand` (nunca compiten) |
| Estratos comparables (≥2 candidatas) | **No se encontró un `if len(cand) < 2` explícito** en los scripts revisados; el diccionario `EXPECTED` fija de antemano 4-10 candidatas por estrato (todas ≥4), consistente con la regla mencionada pero sin una rama de código que la ejercite en esta corrida (ningún estrato tuvo menos de 4 candidatas). **Se declara indeterminado si existe un guard explícito en otro lugar no revisado**, no se afirma que la regla esté ausente. |

**Veredicto C14:** las 7 filas de `tab:reglas-pareto` son una descripción fiel
(ni inflada ni recortada) del pipeline: 6 de 7 criterios tienen mecanismo de
código identificado directamente; el séptimo es consistente con los datos
observados pero no se pudo verificar como rama de código activa. Ningún texto
vivo afirma "nueve" ni ningún otro número que contradiga las 7 filas, así que
no se requiere edición de prosa por C14 más allá de retirar el comentario.

## 4. Veredicto

**La versión que quedó en el documento (cuatro estratos admitidos, frente
construido, Convolución~2D incluida en el multiobjetivo) está respaldada por
los resultados descargados**, con origen en `Fase_4/analysis/out/audit_report.md`,
`out/tables/F6_points.csv`, `out/tables/F6_exclusions.csv` y los `.caption.txt`
de `F6_pareto_conv_h40_modoA/B`/`F9_pareto2d_conv` (todos del job 7145). La
versión alternativa ("resultó viable solo en GEMM", "todos sus estratos se
descartaron") no tiene respaldo en ningún resultado descargado y no aparece en
ninguna prosa viva del documento actual.

## 5. Comentarios retirados

Los tres bloques señalados en el contexto (ubicados hoy, tras los merges de
C1/C2, en las líneas ~2420-2429, ~2717-2718 y ~2959-2961 de `main.tex`) se
reemplazan cada uno por una línea:
`% [C3-RESUELTO] Verificado contra Fase_4/analysis/out/audit_report.md -- ver docs/auditoria/C3_reporte.md.`

No se tocó ningún otro comentario `% [REVISIÓN]`/`% [INTEGRACIÓN]`.

## 6. Estado final

- C3: **resuelto**, con archivo de origen para cada cifra citada arriba.
- C14: **resuelto** en cuanto a coherencia texto/tabla (sin discrepancia viva);
  el mapeo 6/7 criterios-código quedó documentado, con el séptimo declarado
  indeterminado (no ausente, no confirmado) en vez de asumido.
- Ningún job relanzado; no fue necesario.
- Compilación verificada: pdflatex+bibtex+2×pdflatex, 0 errores nuevos, sin
  referencias `??`.
