#!/usr/bin/env python3
"""GATE 3 -- validacion automatizada del ancla FP64 (K=0 y K=1), tres kernels.

Es el script que `Fase_4/Stencil/README.md` nombra como `gate3_ancla.sbatch` y
que `Fase_4/GEMM/README.md` y `Fase_4/Convolution/README.md` piden con la misma
redaccion ("mientras tanto, correr las dos puertas de la seccion Validacion
arriba a mano"). Sigue el estilo de `old/Fase_4/Stencil/tools/comparar_gate1.py`:
separar columnas DETERMINISTAS de columnas de MEDICION, y decidir el veredicto
solo sobre las primeras, con codigo de salida 0/1.

UN SOLO SCRIPT PARA LOS TRES KERNELS -- por que
------------------------------------------------
Se evaluo escribir `Fase_4/{GEMM,Convolution,Stencil}/tools/gate3_ancla.py` por
separado. Se descarto: la LOGICA de los dos gates es literalmente la misma en
los tres kernels (K=0 = diff determinista contra Fase 3; K=1 = cota numerica
sobre la ruta anclada), y lo unico que cambia es el ESQUEMA DE COLUMNAS de las
lineas CSV_*, que aqui vive aislado en una sola tabla de datos (ESQUEMAS) por
kernel. Tres copias de la misma logica de comparacion habrian divergido a la
primera correccion. GEMM y Convolucion ya comparten extract_csv_chained.py por
la misma razon; Stencil trae su esquema propio, que es exactamente lo que la
tabla ESQUEMAS modela.

Vive en Fase_4/tools/ (junto a extract_csv*.py y common_analysis.py), que es
donde ya esta el resto de la herramienta transversal a kernels. Cada
`Fase_4/<kernel>/gate3_ancla.sbatch` lo invoca con su --kernel.

========================================================================
GATE K=0 -- el ancla deshabilitada no cambia NADA
========================================================================
Corre el binario de Fase 4 con `--anchor-every 0` y el de Fase 3 con los
mismos parametros, y exige que las columnas DETERMINISTAS de CSV_DRIFT /
CSV_SUMMARY (y del resto de tokens CSV_* en Stencil) sean identicas campo a
campo. Las columnas de MEDICION (tiempo, energia, GFLOPS, speedups) se
reportan como desviacion relativa pero NO deciden el veredicto: dos corridas
del MISMO binario ya difieren ahi por ruido de reloj/termico/NVML, y exigirles
igualdad byte a byte haria fallar el gate siempre por un motivo que no tiene
nada que ver con el ancla.

========================================================================
GATE K=1 -- el ancla en cada iteracion alcanza el piso que le corresponde
========================================================================
LEER ESTO ANTES DE CAMBIAR LAS TOLERANCIAS. La formulacion ingenua del gate
("con K=1, rel_l2 debe caer a nivel de ruido de punto flotante, ~1e-16")
NO PUEDE PASAR en GEMM ni en Convolucion, y no por un bug: por lo que esos
CSV comparan.

  * En GEMM/Convolucion, CSV_DRIFT compara la referencia FP64 contra el
    buffer T (FP16/BF16) TAL CUAL SE GUARDA -- nunca contra T+comp (esto ya
    lo documenta Fase_4/GEMM/README.md). Con K=1 se puede demostrar que la
    reconstruccion es EXACTA: comp64 = out64 - dequant(q) es una resta exacta
    (los dos operandos estan dentro de un factor 2, Sterbenz), y por lo tanto
    dequant(q) + comp64 == out64 exactamente. O sea que el estado interno SI
    reproduce la trayectoria FP64 bit a bit, y lo unico que separa a T de la
    referencia es la CUANTIZACION al formato de 16 bits.

    De ahi sale una cota EXACTA, no un umbral inventado: para redondeo al mas
    cercano con p bits de significando, |ref_i - T_i| <= 2^-p * |ref_i| para
    todo i, y por lo tanto

        rel_linf = max|ref-T| / max|ref|  <=  2^-p
        rel_l2   = ||ref-T||_2 / ||ref||_2 <= 2^-p

    con p = 11 en FP16 (10 bits explicitos + 1 implicito) y p = 8 en BF16
    (7 + 1). Es decir 4.883e-04 y 3.906e-03.

    VERIFICADO EN GPU REAL (RTX 3050, sm_86, conv_chained --hw 64 --iters 12
    --tc both --comp on --anchor-every 1): FP16 llego a rel_linf = 3.74e-04 y
    BF16 a 3.00e-03 -- ambos al 0.77 de su cota, el MISMO factor en los dos
    formatos, que es la confirmacion empirica de que el modelo es el correcto.

  * En Stencil, CSV_DRIFT compara contra `d_out_fp32`: el acumulador FP32
    SIN el redondeo de almacenamiento ("ancla de no-regresion", ver el
    comentario de reduced_to_float en el .cu). NO es el mismo objeto. El
    analogo real de la columna rel_l2 de GEMM/Conv es rel_l2_prop
    (CSV_SUMMARY), que si mide el estado propagado en 16 bits.

    Por eso en Stencil el gate K=1 tiene DOS criterios, cada uno sobre la
    columna que le corresponde:
      (a) rel_l2/rel_linf (acumulador FP32) de la ruta anclada debe alcanzar
          el nivel de la ruta GPU_FP64 de la MISMA corrida -- que es la
          formulacion que ya pide Fase_4/Stencil/README.md ("numericamente
          indistinguible de GPU_FP64"). Con K=1 el paso se sustituye entero
          por stencil2d_fp64_kernel, asi que la trayectoria anclada ES la de
          GPU_FP64; se admite un factor de holgura (--factor-gpu-fp64) en vez
          de exigir identidad bit a bit, para no hacer fallar el gate por el
          narrowing a float del readout.
      (b) rel_l2_prop/rel_linf_prop deben cumplir la MISMA cota 2^-p que
          GEMM/Convolucion, por el mismo argumento.

Aplicar el criterio de un kernel al otro es el error facil de cometer aqui, y
es la razon por la que este script no tiene un solo umbral global.

USO
---
    # las dos puertas de un tiron (lo habitual):
    python3 gate3_ancla.py --kernel gemm \\
        --gate0-base   fase3_k0.log \\
        --gate0-nuevo  fase4_k0.log \\
        --gate1        fase4_k1.log

    # solo una de las dos:
    python3 gate3_ancla.py --kernel stencil --gate1 fase4_k1.log

Codigos de salida:
    0  todas las puertas pedidas pasan.
    1  FALLA al menos una.
    2  no se pudo evaluar (falta un log, o un log no trae filas CSV_*).
       NO es un "pasa": se distingue a proposito del 1.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

# ---------------------------------------------------------------------------
# Cota de cuantizacion por formato: 2^-p, con p = bits de significando
# (incluido el implicito). Ver la derivacion en el docstring.
# ---------------------------------------------------------------------------
ULP_POR_FORMATO = {
    "fp16": 2.0 ** -11,   # 10 bits explicitos + 1 implicito -> 4.883e-04
    "bf16": 2.0 ** -8,    # 7 bits explicitos + 1 implicito  -> 3.906e-03
}

# ---------------------------------------------------------------------------
# Esquemas de las lineas CSV_* por kernel.
#
#   nombres  : columnas en orden, contando el token CSV_* como columna 0.
#   clave    : columnas que identifican la fila (no son un resultado).
#   medicion : columnas de tiempo/energia -- se comparan con tolerancia
#              relativa y NO deciden el veredicto del gate K=0.
#
# Todo lo que no sea clave ni medicion es DETERMINISTA y tiene que salir
# identico entre la corrida de Fase 3 y la de Fase 4 con --anchor-every 0.
# ---------------------------------------------------------------------------

_CHAINED_DRIFT = {
    "nombres": ["token", "route", "size", "iter", "rel_l2", "rel_linf",
                "solution_finite", "anchor_every"],
    "clave": ("route", "size", "iter"),
    "medicion": set(),
}
_CHAINED_SUMMARY = {
    "nombres": ["token", "route", "size", "iters", "t_iter_ms", "t_total_ms",
                "gflops", "energy_gpu_j", "window_reliable", "gpu_segments",
                "anchor_every"],
    "clave": ("route", "size", "iters"),
    # window_reliable depende del tiempo de pared (total_s >= 0.5s * tramos),
    # asi que es una columna de MEDICION aunque valga 0/1. gpu_segments, en
    # cambio, lo fija la cadencia de checkpoints: determinista.
    "medicion": {"t_iter_ms", "t_total_ms", "gflops", "energy_gpu_j",
                 "window_reliable"},
}

# Stencil: 29 columnas historicas + 8 de Fase 4 + anchor_every = 38.
_STENCIL_SUMMARY_NOMBRES = [
    "token", "route", "nx", "ny", "iters", "kahan",
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms",
    "rel_l2", "rel_linf", "max_abs", "rel_l2_prop", "rel_linf_prop",
    "first_nonfinite", "store_rel_norm", "store_rel_max_guarded",
    "store_excluded_count", "store_eval_iter",
    "energy_gpu_j", "energy_cpu_j", "energy_total_j", "edp_j_s",
    "joules_per_gflop",
    "op_mode", "alpha", "ci_mode", "ci_p", "cell_updates_per_s",
    "energy_per_cell_update_j", "reference_role", "execution_mode",
    "anchor_every",
]
_STENCIL_MEDICION = {
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms",
    "energy_gpu_j", "energy_cpu_j", "energy_total_j", "edp_j_s",
    "joules_per_gflop", "cell_updates_per_s", "energy_per_cell_update_j",
}

ESQUEMAS = {
    "gemm": {"CSV_DRIFT": _CHAINED_DRIFT, "CSV_SUMMARY": _CHAINED_SUMMARY},
    "conv": {"CSV_DRIFT": _CHAINED_DRIFT, "CSV_SUMMARY": _CHAINED_SUMMARY},
    "stencil": {
        "CSV_SUMMARY": {"nombres": _STENCIL_SUMMARY_NOMBRES,
                        "clave": ("route",), "medicion": _STENCIL_MEDICION},
        "CSV_DRIFT": {"nombres": ["token", "route", "iter", "ref_l2", "abs_l2",
                                  "rel_l2", "max_abs", "anchor_every"],
                      "clave": ("route", "iter"), "medicion": set()},
        "CSV_CKPT": {"nombres": ["token", "route", "iter", "rel_l2", "rel_linf",
                                 "max_abs"],
                     "clave": ("route", "iter"), "medicion": set()},
        "CSV_STORE": {"nombres": ["token", "route", "nx", "ny", "iters", "kahan",
                                  "store_rel_norm", "store_rel_max_guarded",
                                  "store_excluded_count", "store_eval_iter",
                                  "ulp_formato"],
                      "clave": ("route",), "medicion": set()},
        "CSV_HORIZON": {"nombres": ["token", "format", "nx", "ny", "iters", "kahan",
                                    "h_predicho", "h_medido", "lambda", "r2",
                                    "n_puntos_fit", "semilla_A", "nyquist_u0",
                                    "piso_siembra", "fit_status"],
                        "clave": ("format",), "medicion": set()},
        "CSV_ONSET": {"nombres": ["token", "route", "onset"],
                      "clave": ("route",), "medicion": set()},
        "CSV_ENERGY": {"nombres": ["token", "route", "nx", "ny", "iters", "kahan",
                                   "energy_gpu_j", "energy_cpu_j", "energy_total_j",
                                   "edp_j_s", "joules_per_gflop", "time_total_s",
                                   "flops_total_billions", "energy_gpu_j_per_iter",
                                   "energy_window_reliable", "anchor_every"],
                       "clave": ("route",),
                       # CSV_ENERGY es telemetria de punta a punta: TODO menos
                       # la clave y anchor_every es medicion.
                       "medicion": {"energy_gpu_j", "energy_cpu_j", "energy_total_j",
                                    "edp_j_s", "joules_per_gflop", "time_total_s",
                                    "flops_total_billions", "energy_gpu_j_per_iter",
                                    "energy_window_reliable"}},
    },
}


# ---------------------------------------------------------------------------
# Lectura de logs
# ---------------------------------------------------------------------------

def leer_lineas_csv(path):
    """Agrupa las lineas CSV_* del log por token. CSV_REGION se descarta (son
    marcas de tiempo de pared, no un resultado) -- mismo criterio que
    comparar_gate1.py."""
    por_tipo = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n")
            if not linea.startswith("CSV_"):
                continue
            token = linea.split(",", 1)[0]
            if token == "CSV_REGION":
                continue
            try:
                campos = next(csv.reader([linea]))
            except csv.Error:
                continue
            por_tipo.setdefault(token, []).append(campos)
    return por_tipo


# Linea de configuracion que cada binario imprime UNA VEZ por invocacion.
# Sirve para detectar un log que contenga VARIAS corridas concatenadas (p. ej.
# el log de un barrido completo en vez del de una pasada suelta): en ese caso
# la misma clave de identidad aparece repetida y la comparacion del gate K=0
# emparejaria filas de corridas distintas -- en el mejor caso fallaria por
# conteo, en el peor compararia dos pasadas equivalentes y "pasaria" sin haber
# comparado lo que dice comparar.
_CONFIG_PREFIJO = {
    "gemm": "N=",
    "conv": "HW=",
    "stencil": "Ancla FP64 (anchor-every)",
}


def contar_invocaciones(path, kernel):
    prefijo = _CONFIG_PREFIJO[kernel]
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            if linea.startswith(prefijo):
                n += 1
    return n


def exigir_invocacion_unica(path, kernel, etiqueta, informe):
    """Devuelve True si el log tiene exactamente una corrida."""
    n = contar_invocaciones(path, kernel)
    if n == 1:
        return True
    if n == 0:
        # Los binarios de Fase 3 de Stencil no imprimen la linea del ancla:
        # ahi la ausencia es lo normal y no se puede contar por esta via.
        if kernel == "stencil":
            return True
        informe["no_evaluable"].append(
            "%s (%s): no se encontro la linea de configuracion del binario "
            "('%s...'). No se puede confirmar que el log traiga UNA sola corrida."
            % (etiqueta, path, _CONFIG_PREFIJO[kernel]))
        return False
    informe["no_evaluable"].append(
        "%s (%s): el log trae %d corridas concatenadas, no una. Este gate compara "
        "fila a fila por identidad (route/size/iter), asi que con varias corridas "
        "en el mismo archivo emparejaria filas de pasadas distintas. Use el log de "
        "UNA invocacion -- que es lo que produce gate3_ancla.sbatch."
        % (etiqueta, path, n))
    return False


def indexar(filas, nombres, clave_cols):
    idx = {}
    for campos in filas:
        d = dict(zip(nombres, campos))
        clave = tuple(d.get(c, "") for c in clave_cols)
        idx.setdefault(clave, []).append(d)
    return idx


def num(valor):
    """float(valor) o None si no es un numero finito (NaN/NONFINITE/NA/vacio)."""
    if valor is None:
        return None
    v = valor.strip()
    if v == "" or v.upper() in {"NONFINITE", "NA", "NAN", "NO_EVALUABLE", "INF",
                                "+INF", "-INF"}:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def desviacion_relativa(a, b):
    fa, fb = num(a), num(b)
    if fa is None or fb is None:
        return None
    if fa == 0.0 and fb == 0.0:
        return 0.0
    denom = max(abs(fa), abs(fb))
    return abs(fa - fb) / denom if denom else 0.0


def formato_de_ruta(route):
    r = (route or "").lower()
    if "fp16" in r:
        return "fp16"
    if "bf16" in r:
        return "bf16"
    return None


def es_ruta_anclada(kernel, route):
    """La ruta sobre la que el ancla realmente actua.

    GEMM/Convolucion: solo la ruta CON compensacion ("<FMT>_comp"); el binario
    rechaza --anchor-every>0 sin --comp on.
    Stencil: solo las rutas WMMA bajo compensacion ESPACIAL ("WMMA_*_SP"); el
    binario rechaza --anchor-every>0 sin --spatial-comp on.
    """
    r = (route or "")
    if kernel in ("gemm", "conv"):
        return r.endswith("_comp")
    return r.upper().startswith("WMMA") and r.upper().endswith("_SP")


# ---------------------------------------------------------------------------
# GATE K=0
# ---------------------------------------------------------------------------

def gate_k0(kernel, path_base, path_nuevo, tolerancia, informe):
    ok_base = exigir_invocacion_unica(path_base, kernel, "GATE K=0 / Fase 3", informe)
    ok_nuevo = exigir_invocacion_unica(path_nuevo, kernel, "GATE K=0 / Fase 4", informe)
    if not (ok_base and ok_nuevo):
        return
    base = leer_lineas_csv(path_base)
    nuevo = leer_lineas_csv(path_nuevo)
    if not base or not nuevo:
        informe["no_evaluable"].append(
            "GATE K=0: alguno de los dos logs no trae ninguna linea CSV_* "
            "(base=%d tokens, nuevo=%d tokens)." % (len(base), len(nuevo)))
        return

    esquemas = ESQUEMAS[kernel]
    for token, esquema in esquemas.items():
        if token not in base and token not in nuevo:
            continue
        nombres = esquema["nombres"]
        base_idx = indexar(base.get(token, []), nombres, esquema["clave"])
        nuevo_idx = indexar(nuevo.get(token, []), nombres, esquema["clave"])

        for clave in sorted(set(base_idx) - set(nuevo_idx)):
            informe["fallos"].append(
                "K=0 %s %s: la fila existe en Fase 3 y falta en Fase 4" % (token, clave))
        for clave in sorted(set(nuevo_idx) - set(base_idx)):
            informe["fallos"].append(
                "K=0 %s %s: la fila existe en Fase 4 y falta en Fase 3" % (token, clave))

        comunes = sorted(set(base_idx) & set(nuevo_idx))
        for clave in comunes:
            bs, ns = base_idx[clave], nuevo_idx[clave]
            if len(bs) != len(ns):
                informe["fallos"].append(
                    "K=0 %s %s: %d filas en Fase 3 vs %d en Fase 4"
                    % (token, clave, len(bs), len(ns)))
                continue
            for b, n in zip(bs, ns):
                for col in nombres:
                    if col in esquema["clave"] or col == "token":
                        continue
                    vb, vn = b.get(col, "<falta>"), n.get(col, "<falta>")
                    if col in esquema["medicion"]:
                        d = desviacion_relativa(vb, vn)
                        if d is None:
                            if vb != vn:
                                informe["avisos"].append(
                                    "K=0 %s %s %s: %s -> %s (no numerico)"
                                    % (token, clave, col, vb, vn))
                        else:
                            informe["medidas"].append((d, token, clave, col, vb, vn))
                            if d > tolerancia:
                                informe["avisos"].append(
                                    "K=0 %s %s %s: %s -> %s (%.1f %% de desviacion)"
                                    % (token, clave, col, vb, vn, 100.0 * d))
                    elif vb != vn:
                        informe["fallos"].append(
                            "K=0 %s %s %s: Fase3=%s  Fase4=%s"
                            % (token, clave, col, vb, vn))
        informe["comparadas"]["K=0 " + token] = len(comunes)

    # anchor_every tiene que ser 0 en las DOS corridas: si el log de Fase 4 se
    # genero sin --anchor-every 0 el gate estaria comparando otra cosa.
    for etiqueta, datos in (("Fase 3", base), ("Fase 4", nuevo)):
        for token, esquema in esquemas.items():
            if "anchor_every" not in esquema["nombres"]:
                continue
            idx_col = esquema["nombres"].index("anchor_every")
            for campos in datos.get(token, []):
                if len(campos) > idx_col and campos[idx_col].strip() not in ("0", ""):
                    informe["fallos"].append(
                        "K=0: el log de %s trae anchor_every=%s en una fila %s. "
                        "Este gate exige que las DOS corridas usen K=0; con otro "
                        "valor no esta comparando lo que dice comparar."
                        % (etiqueta, campos[idx_col], token))
                    break


# ---------------------------------------------------------------------------
# GATE K=1
# ---------------------------------------------------------------------------

def _cota_de_fila(route, slack):
    fmt = formato_de_ruta(route)
    if fmt is None:
        return None, None
    return fmt, ULP_POR_FORMATO[fmt] * slack


def gate_k1_encadenado(kernel, path, slack, informe):
    """GEMM/Convolucion: cota de cuantizacion sobre CSV_DRIFT de la ruta _comp."""
    datos = leer_lineas_csv(path)
    filas = datos.get("CSV_DRIFT", [])
    if not filas:
        informe["no_evaluable"].append(
            "GATE K=1: %s no trae ninguna linea CSV_DRIFT." % path)
        return

    nombres = ESQUEMAS[kernel]["CSV_DRIFT"]["nombres"]
    evaluadas = 0
    por_ruta = {}
    for campos in filas:
        d = dict(zip(nombres, campos))
        route = d.get("route", "")
        if not es_ruta_anclada(kernel, route):
            continue
        anchor = d.get("anchor_every", "").strip()
        if anchor != "1":
            informe["fallos"].append(
                "K=1 %s iter=%s: la columna anchor_every vale '%s', no 1. El log "
                "no corresponde a una corrida con --anchor-every 1 (o la columna "
                "no llego al CSV)." % (route, d.get("iter"), anchor))
            continue
        fmt, cota = _cota_de_fila(route, slack)
        if cota is None:
            informe["avisos"].append(
                "K=1 %s: no se pudo deducir el formato de la ruta; fila omitida."
                % route)
            continue
        if d.get("solution_finite", "").strip() not in ("1",):
            informe["fallos"].append(
                "K=1 %s iter=%s: solution_finite=%s. Con el ancla en cada "
                "iteracion la trayectoria no puede dejar de ser finita."
                % (route, d.get("iter"), d.get("solution_finite")))
            continue
        evaluadas += 1
        por_ruta.setdefault(route, []).append(d)
        for col in ("rel_l2", "rel_linf"):
            v = num(d.get(col))
            if v is None:
                informe["fallos"].append(
                    "K=1 %s iter=%s: %s no es un numero finito (%s)."
                    % (route, d.get("iter"), col, d.get(col)))
            elif v > cota:
                informe["fallos"].append(
                    "K=1 %s iter=%s: %s = %.3e supera la cota de cuantizacion de "
                    "%s (2^-%d x %.2f = %.3e). Con K=1 el estado interno debe "
                    "reproducir la trayectoria FP64 exactamente, asi que lo unico "
                    "que puede separar a T de la referencia es el redondeo al "
                    "formato de 16 bits."
                    % (route, d.get("iter"), col, v, fmt.upper(),
                       11 if fmt == "fp16" else 8, slack, cota))

    if evaluadas == 0:
        informe["no_evaluable"].append(
            "GATE K=1: ninguna fila CSV_DRIFT de una ruta anclada (_comp) con "
            "anchor_every=1 en %s." % path)
        return
    informe["comparadas"]["K=1 CSV_DRIFT (rutas ancladas)"] = evaluadas

    # Estabilidad: el error no debe CRECER con las iteraciones. Con K=1 la
    # cota de arriba ya lo garantiza en lo esencial, asi que esto es un aviso,
    # no un fallo: sirve para detectar una deriva lenta que aun cabe bajo la
    # cota pero no deberia existir.
    for route, ds in por_ruta.items():
        ds = sorted(ds, key=lambda d: int(d.get("iter", "0") or 0))
        primero, ultimo = num(ds[0].get("rel_l2")), num(ds[-1].get("rel_l2"))
        if primero and ultimo and ultimo > 3.0 * primero:
            informe["avisos"].append(
                "K=1 %s: rel_l2 crecio de %.3e (iter %s) a %.3e (iter %s) pese a "
                "seguir bajo la cota. Con el ancla en cada iteracion no deberia "
                "haber tendencia; revisar antes de barrer K intermedios."
                % (route, primero, ds[0].get("iter"), ultimo, ds[-1].get("iter")))


def gate_k1_stencil(path, slack, factor_gpu_fp64, informe):
    """Stencil: DOS criterios, uno por columna -- ver el docstring del modulo."""
    datos = leer_lineas_csv(path)
    filas = datos.get("CSV_SUMMARY", [])
    if not filas:
        informe["no_evaluable"].append(
            "GATE K=1: %s no trae ninguna linea CSV_SUMMARY." % path)
        return

    nombres = ESQUEMAS["stencil"]["CSV_SUMMARY"]["nombres"]
    resumen = [dict(zip(nombres, campos)) for campos in filas]

    # (a) Referencia GPU_FP64 de la MISMA corrida.
    ref = next((d for d in resumen if d.get("route", "").upper() == "GPU_FP64"), None)
    if ref is None:
        informe["no_evaluable"].append(
            "GATE K=1 (a): no hay fila CSV_SUMMARY de la ruta GPU_FP64 en %s. "
            "Correr con --fp64-gpu on: el criterio (a) compara la ruta anclada "
            "contra ella." % path)
    ref_l2 = num(ref.get("rel_l2")) if ref else None
    ref_linf = num(ref.get("rel_linf")) if ref else None

    ancladas = [d for d in resumen if es_ruta_anclada("stencil", d.get("route", ""))]
    if not ancladas:
        informe["no_evaluable"].append(
            "GATE K=1: ninguna fila CSV_SUMMARY de una ruta anclada (WMMA_*_SP) "
            "en %s. Correr con --spatial-comp on." % path)
        return

    evaluadas = 0
    for d in ancladas:
        route = d.get("route", "")
        anchor = d.get("anchor_every", "").strip()
        if anchor != "1":
            informe["fallos"].append(
                "K=1 %s: la columna anchor_every vale '%s', no 1. El log no "
                "corresponde a una corrida con --anchor-every 1 (o la columna no "
                "llego al CSV)." % (route, anchor))
            continue
        evaluadas += 1

        # (a) el acumulador FP32 debe alcanzar el nivel de GPU_FP64.
        if ref_l2 is not None:
            for col, base in (("rel_l2", ref_l2), ("rel_linf", ref_linf)):
                v = num(d.get(col))
                if base is None:
                    continue
                if v is None:
                    informe["fallos"].append(
                        "K=1 (a) %s: %s no es finito (%s)." % (route, col, d.get(col)))
                elif v > max(base * factor_gpu_fp64, 0.0):
                    informe["fallos"].append(
                        "K=1 (a) %s: %s = %.3e, mas de %.0fx el %.3e de GPU_FP64 en "
                        "la misma corrida. Con K=1 el paso se sustituye entero por "
                        "stencil2d_fp64_kernel, asi que la trayectoria anclada ES "
                        "la de GPU_FP64 y ambas columnas deberian coincidir."
                        % (route, col, v, factor_gpu_fp64, base))

        # (b) el estado PROPAGADO en 16 bits debe cumplir la cota de cuantizacion.
        fmt, cota = _cota_de_fila(route, slack)
        if cota is None:
            informe["avisos"].append(
                "K=1 (b) %s: no se pudo deducir el formato de la ruta." % route)
            continue
        for col in ("rel_l2_prop", "rel_linf_prop"):
            v = num(d.get(col))
            if v is None:
                informe["avisos"].append(
                    "K=1 (b) %s: %s no evaluable (%s) -- la corrida no lo produjo."
                    % (route, col, d.get(col)))
            elif v > cota:
                informe["fallos"].append(
                    "K=1 (b) %s: %s = %.3e supera la cota de cuantizacion de %s "
                    "(2^-%d x %.2f = %.3e). Esta columna SI mide el estado "
                    "propagado en 16 bits, el analogo de rel_l2 en GEMM/Conv."
                    % (route, col, v, fmt.upper(), 11 if fmt == "fp16" else 8,
                       slack, cota))

    informe["comparadas"]["K=1 CSV_SUMMARY (rutas ancladas)"] = evaluadas
    if ref_l2 is not None:
        informe["notas"].append(
            "GPU_FP64 de referencia en esta corrida: rel_l2=%s rel_linf=%s"
            % (ref.get("rel_l2"), ref.get("rel_linf")))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, choices=["gemm", "conv", "stencil"])
    ap.add_argument("--gate0-base", help="log de Fase 3 (sin --anchor-every)")
    ap.add_argument("--gate0-nuevo", help="log de Fase 4 con --anchor-every 0")
    ap.add_argument("--gate1", help="log de Fase 4 con --anchor-every 1")
    ap.add_argument("--etiqueta", default="", help="nombre de la configuracion")
    ap.add_argument("--tolerancia-medicion", type=float, default=0.15,
                    help="desviacion relativa admitida en columnas de tiempo/"
                         "energia del gate K=0 (default 0.15). NO decide el "
                         "veredicto, solo produce avisos.")
    ap.add_argument("--ulp-slack", type=float, default=1.05,
                    help="factor sobre la cota 2^-p del gate K=1 (default 1.05). "
                         "La cota es exacta para redondeo al mas cercano; la "
                         "holgura cubre el doble redondeo double->float->T que "
                         "hace reseed_double_from_fp64_kernel. Subirlo por "
                         "encima de ~1.1 vacia el gate de contenido.")
    ap.add_argument("--factor-gpu-fp64", type=float, default=10.0,
                    help="Stencil, criterio (a): cuantas veces el rel_l2 de "
                         "GPU_FP64 se admite en la ruta anclada (default 10). "
                         "Lo esperado es que coincidan; el factor absorbe el "
                         "narrowing a float del readout.")
    args = ap.parse_args()

    if not (args.gate0_base or args.gate0_nuevo or args.gate1):
        ap.error("no se pidio ninguna puerta: pase --gate0-base/--gate0-nuevo "
                 "y/o --gate1.")
    if bool(args.gate0_base) != bool(args.gate0_nuevo):
        ap.error("el gate K=0 necesita LOS DOS logs: --gate0-base (Fase 3) y "
                 "--gate0-nuevo (Fase 4 con --anchor-every 0).")

    informe = {"fallos": [], "avisos": [], "medidas": [], "comparadas": {},
               "no_evaluable": [], "notas": []}

    print("=" * 72)
    print("GATE 3 :: ANCLA FP64 :: kernel=%s :: %s"
          % (args.kernel, args.etiqueta or "(sin etiqueta)"))
    print("=" * 72)

    faltantes = [p for p in (args.gate0_base, args.gate0_nuevo, args.gate1)
                 if p and not os.path.isfile(p)]
    if faltantes:
        for p in faltantes:
            print("NO SE PUDO EVALUAR: no existe el log %s" % p, file=sys.stderr)
        return 2

    if args.gate0_base:
        print("\n-- GATE K=0: --anchor-every 0 == Fase 3 --")
        print("   Fase 3 : %s" % args.gate0_base)
        print("   Fase 4 : %s" % args.gate0_nuevo)
        gate_k0(args.kernel, args.gate0_base, args.gate0_nuevo,
                args.tolerancia_medicion, informe)

    if args.gate1:
        print("\n-- GATE K=1: --anchor-every 1 alcanza su piso --")
        print("   log    : %s" % args.gate1)
        if args.kernel == "stencil":
            gate_k1_stencil(args.gate1, args.ulp_slack, args.factor_gpu_fp64, informe)
        else:
            gate_k1_encadenado(args.kernel, args.gate1, args.ulp_slack, informe)
        for fmt, u in sorted(ULP_POR_FORMATO.items()):
            print("   cota de cuantizacion %s: %.3e (x%.2f = %.3e)"
                  % (fmt.upper(), u, args.ulp_slack, u * args.ulp_slack))

    if informe["comparadas"]:
        print("\n-- Filas evaluadas por bloque --")
        for nombre, n in sorted(informe["comparadas"].items()):
            print("   %-34s %d" % (nombre, n))

    for nota in informe["notas"]:
        print("   %s" % nota)

    if informe["medidas"]:
        peores = sorted(informe["medidas"], key=lambda t: -t[0])[:10]
        print("\n-- Mayores desviaciones en columnas de MEDICION (K=0) --")
        print("   Tolerancia %.1f %%. Estas columnas NO deciden el veredicto."
              % (100.0 * args.tolerancia_medicion))
        for d, token, clave, col, vb, vn in peores:
            print("   %6.2f %%  %s %s %s: %s -> %s"
                  % (100.0 * d, token, clave, col, vb, vn))

    if informe["avisos"]:
        print("\n-- AVISOS (revisar; no bloquean) --")
        for a in informe["avisos"]:
            print("   ! " + a)

    if informe["no_evaluable"]:
        print("\n-- NO EVALUABLE --")
        for m in informe["no_evaluable"]:
            print("   ? " + m)
        # Los fallos ya detectados se imprimen igual: suelen ser justo la
        # explicacion de por que el gate se quedo sin filas que evaluar (p. ej.
        # "anchor_every vale 0, no 1" cuando se paso el log equivocado).
        for f in informe["fallos"]:
            print("   X " + f)
        print("\n   El gate no pudo decidir: eso NO es un 'pasa'. Revisar como se")
        print("   generaron los logs antes de lanzar ninguna campana con K>0.")
        return 2

    print("\n-- Veredicto --")
    if informe["fallos"]:
        print("   FALLA: %d problema(s).\n" % len(informe["fallos"]))
        for f in informe["fallos"]:
            print("   X " + f)
        print("\n   Ningun valor de K intermedio (5, 8, 20, 32...) tiene sentido")
        print("   reportar mientras alguna de estas puertas falle.")
        return 1

    total = sum(informe["comparadas"].values())
    print("   PASA: %d fila(s) evaluadas, cero fallos deterministas." % total)
    if args.gate0_base:
        print("   K=0: --anchor-every 0 reproduce Fase 3 columna por columna.")
    if args.gate1:
        print("   K=1: la ruta anclada alcanza el piso que le corresponde.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
