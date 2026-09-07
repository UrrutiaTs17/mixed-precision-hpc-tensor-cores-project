#!/usr/bin/env python3
"""Verifica gpu_fp64_conv_step(): orden de operandos, im2col y padding SAME.

QUE PROBLEMA RESUELVE
---------------------
Fase_3/Convolution/README.md declara "mismo punto de mayor riesgo que GEMM":
cuBLAS es column-major, el resto de conv_chained.cu es row-major, y
gpu_fp64_conv_step() invierte el orden de los operandos de cublasDgemm (col
PRIMERO, W SEGUNDO) para compensarlo. Si estuviera al reves, el binario
compila y corre igual, y la "referencia FP64" contra la que se mide TODO el
drift de la campana seria otra operacion.

Pero en Convolucion el orden de operandos NO es el unico error silencioso
posible, y por eso este gate verifica tres cosas, no una:

  (a) ORDEN DE OPERANDOS de cublasDgemm -- el analogo exacto de GEMM.
  (b) INDEXACION DEL im2col -- im2col_double_kernel mapea (crs, pos) a
      (c, r, s, oh, ow). Una transposicion r<->s, o leer el filtro volteado
      (convolucion en vez de correlacion), produce un resultado perfectamente
      finito y plausible. Con el Laplaciano de 5 puntos del proyecto, que es
      SIMETRICO, ninguno de esos dos errores es visible: por eso este script
      corre un segundo caso con un filtro deliberadamente ASIMETRICO, que es
      la unica forma de distinguirlos.
  (c) PADDING "SAME" (pad=1, stride=1, dilation=1) -- las celdas del borde
      deben leer CEROS fuera del dominio. Si el kernel envolviera (wrap) o
      replicara el borde, el interior saldria identico y solo cambiaria el
      anillo exterior: el error quedaria diluido en una norma global. Este
      script reporta el error del BORDE por separado, para que un fallo de
      padding se nombre como tal en vez de esconderse en un rel_l2 promedio.

  Ademas verifica que build_block_diagonal_filter() produzca de verdad la
  estructura bloque-diagonal documentada (W[k,c,:,:] = 0 salvo c==k, con el
  Laplaciano exacto 0.25/-1.0): un filtro que mezclara canales seguiria dando
  numeros finitos, y la afirmacion "64 simulaciones de Stencil independientes"
  del README dejaria de ser cierta sin que nada lo delatara.

POR QUE EXTRAE EL CODIGO DEL .cu EN VEZ DE REIMPLEMENTARLO
-----------------------------------------------------------
Mismo criterio que verificar_orden_operandos_gemm.py: reimplementar el im2col
aqui probaria que ESTE archivo esta bien, no el del proyecto. El script LEE
del .cu -- sin tocar un caracter -- las constantes (kChannels, kFilterR,
kFilterS, kCRS), grid1d(), build_block_diagonal_filter(),
im2col_double_kernel() y gpu_fp64_conv_step(), y los pega en un binario minimo
que expone el resultado. Si alguien cambia el orden de operandos, la
indexacion del im2col o el padding en conv_chained.cu, este gate falla.

Tambien exige que Fase_3 y Fase_4 tengan las MISMAS funciones (Fase 4
"reutiliza gpu_fp64_conv_step() TAL CUAL" segun su README): si eso dejara de
ser cierto, verificar solo Fase 3 daria una falsa sensacion de seguridad.

POR QUE UN BINARIO PROPIO Y NO ./conv_chained --hw 64
------------------------------------------------------
conv_chained no publica en ningun lado el resultado crudo de
gpu_fp64_conv_step(): solo rel_l2/rel_linf de la ruta WMMA CONTRA esa
referencia. Comparar rel_l2 con NumPy no distingue "la referencia esta mal"
de "la ruta WMMA esta mal" -- justo la ambiguedad que hay que romper.

POR QUE ES UN SCRIPT SEPARADO DE verificar_orden_operandos_gemm.py
------------------------------------------------------------------
Se evaluo un unico script con --kernel gemm|conv y se descarto. Lo unico
comun es la plomeria de detectar nvcc, compilar y correr (~60 lineas
mecanicas). Lo demas es disjunto: alli se extrae UNA funcion, aqui CINCO
simbolos mas cuatro constantes; alli el probe hace un producto NxN, aqui
construye un filtro y corre dos casos; alli la referencia NumPy es `X @ A`,
aqui es una correlacion 2D por canal con padding explicito mas una segunda
referencia im2col independiente. Un script unico seria un
`if kernel == "gemm"` en cada uno de esos pasos, y cada gate tiene que poder
correrse y auditarse SOLO dentro de una sesion de PACCA.

USO
---
    python3 verificar_orden_operandos_conv.py              # hw = 64 (el minimo)
    python3 verificar_orden_operandos_conv.py --hw 128
    python3 verificar_orden_operandos_conv.py --conservar-tmp

Codigos de salida:
    0  todo correcto (orden de operandos, im2col, padding, filtro).
    1  FALLA alguna de las verificaciones.
    2  no se pudo verificar (falta nvcc/GPU, o no se pudo extraer el codigo).
       NO es un "pasa": es un "no se sabe", y se distingue a proposito del 1.
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FUENTE_F3 = os.path.join(REPO_ROOT, "Fase_3", "Convolution", "conv_chained.cu")
FUENTE_F4 = os.path.join(REPO_ROOT, "Fase_4", "Convolution", "conv_chained.cu")

# Misma justificacion que en verificar_orden_operandos_gemm.py: el DGEMM
# subyacente acumula del orden de kCRS * eps_double ~ 1e-13; 1e-11 deja dos
# ordenes de holgura y sigue estando ~11 ordenes por debajo del error de orden
# 1 que produciria cualquiera de los fallos que este gate busca.
TOLERANCIA_REL_LINF = 1e-11

# Simbolos que se extraen del .cu, en el orden en que deben aparecer en el
# probe (build_block_diagonal_filter usa las constantes; gpu_fp64_conv_step
# usa grid1d e im2col_double_kernel).
FIRMAS = [
    ("grid1d", "inline int grid1d("),
    ("build_block_diagonal_filter", "static void build_block_diagonal_filter("),
    ("im2col_double_kernel", "__global__ static void im2col_double_kernel("),
    ("gpu_fp64_conv_step", "static void gpu_fp64_conv_step("),
]

# Constantes que el probe necesita, tomadas literalmente del .cu (no
# reescritas: si alguien cambia kChannels a 32, el probe cambia con el).
CONSTANTES = ["kConversionThreads", "kChannels", "kFilterR", "kFilterS", "kCRS"]


# ---------------------------------------------------------------------------
# Extraccion del codigo real del proyecto
# ---------------------------------------------------------------------------

def extraer_funcion(ruta: str, firma: str) -> str:
    """Texto completo de una funcion C++ de `ruta`, contando llaves.

    Extractor deliberadamente simple: si la funcion tuviera una llave dentro de
    un literal de cadena, el conteo se rompe y el probe no compila -- falla
    ruidosamente, que es el modo de fallo correcto para un gate.
    """
    with open(ruta, "r", encoding="utf-8") as fh:
        lineas = fh.readlines()

    inicio = None
    for i, linea in enumerate(lineas):
        if firma in linea and not linea.lstrip().startswith("//"):
            inicio = i
            break
    if inicio is None:
        raise LookupError("no se encontro '%s' en %s" % (firma, ruta))

    profundidad = 0
    abierta = False
    for j in range(inicio, len(lineas)):
        for ch in lineas[j]:
            if ch == "{":
                profundidad += 1
                abierta = True
            elif ch == "}":
                profundidad -= 1
        if abierta and profundidad == 0:
            return "".join(lineas[inicio:j + 1])
    raise LookupError("cuerpo de '%s' sin cerrar en %s" % (firma, ruta))


def extraer_constante(ruta: str, nombre: str) -> str:
    """Linea `constexpr int <nombre> = ...;` tal cual esta en el .cu."""
    with open(ruta, "r", encoding="utf-8") as fh:
        for linea in fh:
            if re.match(r"\s*constexpr\s+int\s+%s\s*=" % re.escape(nombre), linea):
                return linea.rstrip("\n")
    raise LookupError("no se encontro 'constexpr int %s' en %s" % (nombre, ruta))


def leer_constantes(ruta: str) -> "tuple[list[str], dict[str, int]]":
    """Lineas literales + valores de las constantes de CONSTANTES.

    Se resuelven EN ORDEN y cada una ve a las anteriores como variables, para
    que `kCRS = kChannels * kFilterR * kFilterS` se evalue sola. El eval solo
    ve enteros ya extraidos del .cu y ningun builtin.
    """
    lineas, valores = [], {}
    for nombre in CONSTANTES:
        linea = extraer_constante(ruta, nombre)
        lineas.append(linea)
        rhs = re.sub(r"//.*$", "", linea.split("=", 1)[1].split(";", 1)[0]).strip()
        valores[nombre] = int(eval(rhs, {"__builtins__": {}}, dict(valores)))  # noqa: S307
    return lineas, valores


def normalizar(texto: str) -> str:
    return " ".join(texto.split())


# ---------------------------------------------------------------------------
# Toolchain (identico a verificar_orden_operandos_gemm.py -- ver la nota de
# ese archivo sobre por que los dos gates son autocontenidos)
# ---------------------------------------------------------------------------

def detectar_nvcc() -> "str | None":
    if os.environ.get("NVCC"):
        return os.environ["NVCC"]
    return shutil.which("nvcc")


def detectar_cuda_arch() -> str:
    if os.environ.get("CUDA_ARCH"):
        return os.environ["CUDA_ARCH"]
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        mayor, menor = salida.strip().splitlines()[0].strip().split(".")
        return "%d%d" % (int(mayor), int(menor))
    except Exception:
        return "80"


def detectar_ccbin() -> "str | None":
    """Compilador host para nvcc. Solo relevante en Windows -- ver la nota
    extensa en verificar_orden_operandos_gemm.py."""
    if os.environ.get("CCBIN"):
        return os.environ["CCBIN"]
    if os.name != "nt" or shutil.which("cl"):
        return None
    patrones = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    ]
    candidatos = []
    for patron in patrones:
        candidatos.extend(d for d in _glob.glob(patron)
                          if os.path.isfile(os.path.join(d, "cl.exe")))
    return sorted(candidatos)[-1] if candidatos else None


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

# Se arma con str.replace() sobre marcadores @@...@@, NO con formateo `%` ni
# f-strings: el texto es C++ lleno de `%s`/`%d` de printf y de llaves.
PLANTILLA_PROBE = r"""// GENERADO por Fase_3/tools/verificar_orden_operandos_conv.py -- no editar.
//
// Binario minimo cuyo unico proposito es exponer el resultado de
// gpu_fp64_conv_step() TAL COMO ESTA ESCRITA en @@FUENTE@@ (pegada mas abajo
// junto con el im2col, el constructor del filtro y las constantes que usa,
// sin modificar), para poder compararla contra NumPy.
//
// Uso: probe_conv HW MODO_W X.bin W.bin Y.bin
//   MODO_W = builtin : construye W con build_block_diagonal_filter() del
//                      proyecto y lo VUELCA a W.bin (para que el verificador
//                      pueda auditar tambien el filtro en si).
//   MODO_W = file    : LEE W de W.bin (filtro asimetrico de prueba, la unica
//                      forma de detectar una transposicion r<->s o un volteo
//                      del filtro, invisibles con el Laplaciano simetrico).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s en %s:%d\n",                         \
                   cudaGetErrorString(_err), __FILE__, __LINE__);              \
      std::exit(70);                                                           \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    cublasStatus_t _st = (expr);                                               \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d en %s:%d\n", (int)_st,             \
                   __FILE__, __LINE__);                                        \
      std::exit(71);                                                           \
    }                                                                          \
  } while (0)

// ---- INICIO del codigo extraido de @@FUENTE@@ ----
@@CONSTANTES@@

@@FUNCIONES@@
// ---- FIN del codigo extraido ----

static void leer_bin(const char* ruta, double* dst, size_t n) {
  std::FILE* fh = std::fopen(ruta, "rb");
  if (!fh) { std::fprintf(stderr, "no se pudo abrir %s\n", ruta); std::exit(72); }
  if (std::fread(dst, sizeof(double), n, fh) != n) {
    std::fprintf(stderr, "lectura corta en %s\n", ruta); std::exit(72);
  }
  std::fclose(fh);
}

static void escribir_bin(const char* ruta, const double* src, size_t n) {
  std::FILE* fh = std::fopen(ruta, "wb");
  if (!fh) { std::fprintf(stderr, "no se pudo escribir %s\n", ruta); std::exit(74); }
  std::fwrite(src, sizeof(double), n, fh);
  std::fclose(fh);
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "uso: %s HW builtin|file X.bin W.bin Y.bin\n", argv[0]);
    return 73;
  }
  const int hw = std::atoi(argv[1]);
  const bool w_builtin = (std::strcmp(argv[2], "builtin") == 0);
  const size_t field = (size_t)kChannels * (size_t)hw * (size_t)hw;
  const size_t col_sz = (size_t)kCRS * (size_t)hw * (size_t)hw;
  const size_t w_sz = (size_t)kChannels * (size_t)kCRS;

  std::vector<double> h_x(field), h_w(w_sz), h_y(field);
  leer_bin(argv[3], h_x.data(), field);
  if (w_builtin) {
    // build_block_diagonal_filter() del proyecto, tal cual.
    std::vector<double> W;
    build_block_diagonal_filter(W);
    if (W.size() != w_sz) {
      std::fprintf(stderr, "build_block_diagonal_filter dio %zu elementos, se esperaban %zu\n",
                   W.size(), w_sz);
      return 75;
    }
    h_w = W;
    escribir_bin(argv[4], h_w.data(), w_sz);  // para auditarlo desde Python
  } else {
    leer_bin(argv[4], h_w.data(), w_sz);
  }

  double *d_x = nullptr, *d_w = nullptr, *d_col = nullptr, *d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, field * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_w, w_sz * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_col, col_sz * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_y, field * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), field * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), w_sz * sizeof(double), cudaMemcpyHostToDevice));
  // El scratch se ensucia a proposito con un patron no nulo: si
  // im2col_double_kernel dejara alguna posicion sin escribir, el resultado
  // saldria contaminado de forma visible en vez de "funcionar por casualidad"
  // sobre un buffer que cudaMalloc devolvio en ceros.
  {
    std::vector<double> basura(col_sz, -12345.0);
    CHECK_CUDA(cudaMemcpy(d_col, basura.data(), col_sz * sizeof(double),
                          cudaMemcpyHostToDevice));
  }

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  // LA llamada bajo prueba, con la misma firma que usa run_chained_route():
  //   gpu_fp64_conv_step(handle, d_x_in, d_w_fp64, d_col64, d_x_out, hw)
  gpu_fp64_conv_step(handle, d_x, d_w, d_col, d_y, hw);
  CHECK_CUDA(cudaDeviceSynchronize());
  cublasDestroy(handle);

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, field * sizeof(double), cudaMemcpyDeviceToHost));
  escribir_bin(argv[5], h_y.data(), field);

  cudaFree(d_x); cudaFree(d_w); cudaFree(d_col); cudaFree(d_y);
  return 0;
}
"""


# ---------------------------------------------------------------------------
# Referencias independientes en NumPy
# ---------------------------------------------------------------------------

def referencia_directa(X, W4, hw, canales, R, S):
    """Y[k] = sum_{c,r,s} W[k,c,r,s] * Xpad[c, oh-1+r, ow-1+s], con CEROS fuera.

    Implementacion por desplazamientos sobre un arreglo con padding explicito:
    no usa im2col en ningun paso, asi que no puede "coincidir por compartir el
    bug" con el kernel que se esta verificando. Es la referencia primaria.
    """
    pad = np.zeros((canales, hw + 2, hw + 2), dtype=np.float64)
    pad[:, 1:hw + 1, 1:hw + 1] = X
    Y = np.zeros((canales, hw, hw), dtype=np.float64)
    for k in range(canales):
        for c in range(canales):
            w = W4[k, c]
            if not np.any(w):
                continue
            for r in range(R):
                for s in range(S):
                    coef = w[r, s]
                    if coef == 0.0:
                        continue
                    # ih = oh - 1 + r  ->  en `pad` (desplazado +1): oh + r
                    Y[k] += coef * pad[c, r:r + hw, s:s + hw]
    return Y


def referencia_im2col(X, W2, hw, canales, R, S, crs):
    """Segunda referencia, por el camino im2col + producto matricial de NumPy.

    Replica LITERALMENTE la formula que documenta el README
    (Y[K, outH*outW] = W[K, C*R*S] . col[C*R*S, outH*outW]) pero armando `col`
    en NumPy. Sirve de control cruzado de la referencia directa: si las dos
    difieren, el error esta en ESTE archivo, no en el proyecto, y conviene
    saberlo antes de acusar al .cu.
    """
    pad = np.zeros((canales, hw + 2, hw + 2), dtype=np.float64)
    pad[:, 1:hw + 1, 1:hw + 1] = X
    col = np.empty((crs, hw * hw), dtype=np.float64)
    for c in range(canales):
        for r in range(R):
            for s in range(S):
                fila = (c * R + r) * S + s
                col[fila] = pad[c, r:r + hw, s:s + hw].reshape(-1)
    return (W2 @ col).reshape(canales, hw, hw)


def rel_linf(ref, test):
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float("inf")
    return float(np.max(np.abs(ref - test)) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, default=64,
                    help="alto/ancho del campo espacial (default 64, el minimo "
                         "que acepta conv_chained: hw*hw debe ser multiplo de 64)")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--nvcc", default=None, help="ruta a nvcc (default: $NVCC o PATH)")
    ap.add_argument("--cuda-arch", default=None,
                    help="arquitectura sm_XX sin el prefijo (default: la de la GPU 0)")
    ap.add_argument("--ccbin", default=None,
                    help="compilador host para nvcc (-ccbin). Solo hace falta en "
                         "Windows; en Linux se ignora.")
    ap.add_argument("--conservar-tmp", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("VERIFICACION DE gpu_fp64_conv_step -- CONVOLUCION")
    print("  (orden de operandos + indexacion im2col + padding SAME + filtro)")
    print("=" * 72)

    # --- 1. Extraer el codigo real de los dos .cu y exigir que coincidan ---
    try:
        const_lineas, const_val = leer_constantes(FUENTE_F3)
        funcs_f3 = [(n, extraer_funcion(FUENTE_F3, f)) for n, f in FIRMAS]
        funcs_f4 = [(n, extraer_funcion(FUENTE_F4, f)) for n, f in FIRMAS]
    except (LookupError, OSError, SyntaxError, ValueError) as exc:
        print("NO SE PUDO VERIFICAR: %s" % exc, file=sys.stderr)
        print("  (cambiaron las firmas de conv_chained.cu? actualizar este gate)",
              file=sys.stderr)
        return 2

    distintas = [n for (n, a), (_, b) in zip(funcs_f3, funcs_f4)
                 if normalizar(a) != normalizar(b)]
    if distintas:
        print("FALLA: estas funciones NO son identicas en Fase 3 y Fase 4: %s"
              % ", ".join(distintas))
        print("  Fase_4/Convolution/README.md afirma que Fase 4 reutiliza")
        print("  gpu_fp64_conv_step() 'TAL CUAL'. Verificar una sola de las dos ya")
        print("  no cubre a la otra: revisar ambas antes de confiar en la campana.")
        return 1

    canales = const_val["kChannels"]
    R, S = const_val["kFilterR"], const_val["kFilterS"]
    crs = const_val["kCRS"]
    print("[1/6] Codigo extraido de Fase_3 y Fase_4/Convolution: identico.")
    print("      kChannels=%d kFilterR=%d kFilterS=%d kCRS=%d" % (canales, R, S, crs))

    if crs != canales * R * S:
        print("FALLA: kCRS (%d) != kChannels*kFilterR*kFilterS (%d)."
              % (crs, canales * R * S))
        return 1

    # --- 2. Toolchain ---
    nvcc = args.nvcc or detectar_nvcc()
    if not nvcc:
        print("NO SE PUDO VERIFICAR: no se encontro nvcc (ni $NVCC ni en PATH).",
              file=sys.stderr)
        print("  Este gate necesita compilar y correr en una GPU Ampere+ real.",
              file=sys.stderr)
        return 2
    arch = args.cuda_arch or detectar_cuda_arch()
    ccbin = args.ccbin or detectar_ccbin()
    print("[2/6] nvcc=%s  arch=sm_%s%s"
          % (nvcc, arch, ("  ccbin=%s" % ccbin) if ccbin else ""))

    hw = args.hw
    if hw <= 0 or (hw * hw) % 64 != 0:
        print("--hw=%d invalido: hw*hw debe ser multiplo de 64 (misma validacion "
              "que parse_args de conv_chained.cu)." % hw, file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="verif_conv_")
    try:
        probe_cu = os.path.join(tmp, "probe_conv.cu")
        probe_bin = os.path.join(tmp, "probe_conv")
        with open(probe_cu, "w", encoding="utf-8") as fh:
            fh.write(PLANTILLA_PROBE
                     .replace("@@FUENTE@@",
                              os.path.relpath(FUENTE_F3, REPO_ROOT).replace("\\", "/"))
                     .replace("@@CONSTANTES@@", "\n".join(const_lineas))
                     .replace("@@FUNCIONES@@", "\n".join(t for _, t in funcs_f3)))

        cmd = [nvcc, "-std=c++17", probe_cu, "-o", probe_bin, "-lcublas",
               "-gencode", "arch=compute_%s,code=sm_%s" % (arch, arch),
               "--allow-unsupported-compiler"]
        if ccbin:
            cmd += ["-ccbin", ccbin]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("NO SE PUDO VERIFICAR: el probe no compilo.", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 2
        print("[3/6] probe compilado (codigo del proyecto pegado sin modificar).")

        rng = np.random.default_rng(args.seed)
        X = rng.standard_normal((canales, hw, hw))
        x_bin = os.path.join(tmp, "X.bin")
        w_bin = os.path.join(tmp, "W.bin")
        y_bin = os.path.join(tmp, "Y.bin")
        X.astype(np.float64).tofile(x_bin)

        fallos = []

        def correr(modo_w):
            p = subprocess.run([probe_bin, str(hw), modo_w, x_bin, w_bin, y_bin],
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError("probe fallo (codigo %d): %s" % (p.returncode, p.stderr))
            return np.fromfile(y_bin, dtype=np.float64).reshape(canales, hw, hw)

        # ---- CASO 1: el filtro real del proyecto ----------------------------
        try:
            Y = correr("builtin")
        except RuntimeError as exc:
            print("NO SE PUDO VERIFICAR: %s" % exc, file=sys.stderr)
            return 2
        W2 = np.fromfile(w_bin, dtype=np.float64).reshape(canales, crs)
        W4 = W2.reshape(canales, canales, R, S)

        # 4a. El filtro en si: bloque-diagonal + Laplaciano exacto.
        laplaciano = np.array([[0.0, 0.25, 0.0],
                               [0.25, -1.0, 0.25],
                               [0.0, 0.25, 0.0]])
        fuera_diag = np.abs(W4[~np.eye(canales, dtype=bool)]).max() if canales > 1 else 0.0
        diag_ok = all(np.array_equal(W4[k, k], laplaciano) for k in range(canales))
        print("[4/6] Filtro de build_block_diagonal_filter():")
        print("      bloques fuera de la diagonal, |max| = %.3e (debe ser 0)" % fuera_diag)
        print("      cada bloque diagonal == Laplaciano 5 puntos exacto: %s"
              % ("SI" if diag_ok else "NO"))
        if fuera_diag != 0.0:
            fallos.append("W mezcla canales: los bloques fuera de la diagonal no son cero. "
                          "La afirmacion '64 simulaciones de Stencil independientes' del "
                          "README dejaria de ser cierta.")
        if not diag_ok:
            fallos.append("los bloques diagonales de W no son el Laplaciano exacto "
                          "(0.25 vecino, -1.0 centro, 0 en esquinas).")

        # 4b. Las dos referencias NumPy deben coincidir entre si primero.
        ref_dir = referencia_directa(X, W4, hw, canales, R, S)
        ref_col = referencia_im2col(X, W2, hw, canales, R, S, crs)
        cruce = rel_linf(ref_dir, ref_col)
        if cruce > TOLERANCIA_REL_LINF:
            print("NO SE PUDO VERIFICAR: las dos referencias NumPy no coinciden "
                  "entre si (%.3e). El bug esta en este script, no en el .cu."
                  % cruce, file=sys.stderr)
            return 2

        err_sim = rel_linf(ref_dir, Y)
        print("[5/6] Caso 1 -- filtro real (Laplaciano simetrico):")
        print("      rel_linf(referencia directa, GPU) = %.3e" % err_sim)
        print("      control cruzado directa-vs-im2col  = %.3e" % cruce)
        if err_sim > TOLERANCIA_REL_LINF:
            fallos.append("caso 1: gpu_fp64_conv_step no reproduce conv(X,W) "
                          "(rel_linf = %.3e > %.1e)." % (err_sim, TOLERANCIA_REL_LINF))

        # ---- CASO 2: filtro ASIMETRICO -------------------------------------
        # El Laplaciano es simetrico: con el, una transposicion r<->s o un
        # volteo del filtro (convolucion en vez de correlacion) son INVISIBLES.
        # Este filtro tiene los nueve coeficientes distintos y una escala
        # propia por canal, asi que distingue las tres cosas a la vez:
        # orientacion (r,s), volteo, y mezcla de canales.
        W4_asim = np.zeros((canales, canales, R, S))
        base = np.arange(1.0, R * S + 1.0).reshape(R, S)  # 1..9, todos distintos
        for k in range(canales):
            W4_asim[k, k] = base * (1.0 + k)
        W2_asim = W4_asim.reshape(canales, crs)
        W2_asim.astype(np.float64).tofile(w_bin)
        try:
            Y2 = correr("file")
        except RuntimeError as exc:
            print("NO SE PUDO VERIFICAR: %s" % exc, file=sys.stderr)
            return 2

        ref_asim = referencia_directa(X, W4_asim, hw, canales, R, S)
        # Hipotesis alternativas, para poder NOMBRAR el fallo si lo hay.
        ref_flip = referencia_directa(X, W4_asim[:, :, ::-1, ::-1], hw, canales, R, S)
        ref_transp = referencia_directa(X, np.swapaxes(W4_asim, 2, 3), hw, canales, R, S)
        err_asim = rel_linf(ref_asim, Y2)
        print("[6/6] Caso 2 -- filtro asimetrico (nueve coeficientes distintos):")
        print("      correlacion, W tal cual (correcto)   %.3e" % err_asim)
        print("      convolucion, W volteado              %.3e" % rel_linf(ref_flip, Y2))
        print("      W transpuesto (r<->s)                %.3e" % rel_linf(ref_transp, Y2))
        if err_asim > TOLERANCIA_REL_LINF:
            detalle = ""
            if rel_linf(ref_flip, Y2) <= TOLERANCIA_REL_LINF:
                detalle = " Lo que calcula es la CONVOLUCION (filtro volteado), no la " \
                          "correlacion que asume el im2col."
            elif rel_linf(ref_transp, Y2) <= TOLERANCIA_REL_LINF:
                detalle = " Lo que calcula usa el filtro TRANSPUESTO (r y s cambiados)."
            fallos.append("caso 2: la indexacion del im2col no es la documentada "
                          "(rel_linf = %.3e).%s" % (err_asim, detalle))

        # ---- Padding: el anillo del borde, reportado aparte ----------------
        # Un error de padding (wrap/replicate en vez de ceros) deja el interior
        # EXACTO y solo altera el borde: en una norma global sobre hw=64 eso es
        # ~1/16 de las celdas y podria pasar por ruido. Aqui se mira solo.
        borde = np.zeros((hw, hw), dtype=bool)
        borde[0, :] = borde[-1, :] = borde[:, 0] = borde[:, -1] = True
        interior = ~borde
        err_borde = rel_linf(ref_asim[:, borde], Y2[:, borde])
        err_interior = rel_linf(ref_asim[:, interior], Y2[:, interior])
        print()
        print("      Padding SAME (pad=1, ceros fuera del dominio):")
        print("        rel_linf en el INTERIOR %.3e" % err_interior)
        print("        rel_linf en el BORDE    %.3e" % err_borde)
        if err_borde > TOLERANCIA_REL_LINF >= err_interior:
            fallos.append("padding: el interior es correcto pero el BORDE no. El "
                          "im2col no esta rellenando con ceros fuera del dominio "
                          "(wrap? replicate? off-by-one en pad?).")
        elif err_borde > TOLERANCIA_REL_LINF:
            fallos.append("padding: el borde tampoco coincide (rel_linf = %.3e)."
                          % err_borde)

        # ---- Veredicto ------------------------------------------------------
        print()
        if not fallos:
            print("PASA: gpu_fp64_conv_step calcula conv(X,W) con padding SAME.")
            print("      - El orden invertido de los argumentos de cublasDgemm compensa")
            print("        correctamente la convencion column-major.")
            print("      - im2col_double_kernel indexa (c,r,s,oh,ow) como documenta el")
            print("        README, sin transposicion ni volteo del filtro.")
            print("      - El borde lee ceros: el padding SAME es el declarado.")
            print("      - build_block_diagonal_filter da 64 canales independientes con")
            print("        el Laplaciano exacto.")
            return 0

        print("FALLA: %d problema(s) en gpu_fp64_conv_step / im2col / filtro:" % len(fallos))
        for f in fallos:
            print("   X %s" % f)
        print()
        print("       NINGUNA columna rel_l2 / rel_linf de una campana de Convolucion")
        print("       (Fase 3 o Fase 4) es valida mientras esto falle: la 'referencia")
        print("       FP64' contra la que se mide todo no es la misma operacion que")
        print("       calcula la ruta WMMA.")
        return 1
    finally:
        if args.conservar_tmp:
            print("\n(probe conservado en %s)" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
