#!/usr/bin/env python3
"""Verifica el ORDEN DE OPERANDOS de cublasDgemm en gpu_fp64_step() (GEMM).

QUE PROBLEMA RESUELVE
---------------------
cuBLAS es column-major; el kernel WMMA de common/wmma_gemm.cuh es row-major.
Para que ambas rutas calculen la MISMA operacion matematica X*A sobre el mismo
buffer sin transponer nada, gpu_fp64_step() invierte el orden de los operandos
en la llamada a cublasDgemm (A primero, X segundo). El comentario de cabecera
de Fase_3/GEMM/gemm_chained.cu y Fase_4/GEMM/gemm_chained.cu declara esto "el
punto de mayor riesgo de error silencioso" del archivo: si estuviera al reves,
el binario compila y corre igual, y la trayectoria de "referencia FP64" seria
A*X (o X^T*A) en vez de X*A -- comparando peras con manzanas, sin ningun error
visible, en TODAS las columnas rel_l2/rel_linf de la campana.

Este script es la verificacion que esos comentarios piden y que no existia.

POR QUE EXTRAE LA FUNCION DEL .cu EN VEZ DE REIMPLEMENTARLA
-----------------------------------------------------------
Reimplementar la llamada a cublasDgemm aqui y compararla con NumPy no probaria
NADA sobre el proyecto: probaria que ESTE archivo esta bien. Lo que hay que
verificar es el codigo que corre en la campana.

Por eso el script LEE el texto de gpu_fp64_step() directamente del .cu del
proyecto y lo pega, sin tocar un caracter, dentro de un binario minimo que
expone su resultado. Si alguien invierte el orden de los argumentos en
gemm_chained.cu, este gate falla -- que es exactamente lo que se quiere.

Ademas exige que Fase_3/GEMM y Fase_4/GEMM tengan la MISMA funcion (Fase 4
"reutiliza gpu_fp64_step() TAL CUAL", segun su propio README y su comentario
de cabecera: si eso dejara de ser cierto, verificar una sola de las dos daria
una falsa sensacion de seguridad). Verificar Fase 3 verifica tambien Fase 4
SOLO mientras esa igualdad se sostenga, y el script la comprueba.

POR QUE UN BINARIO PROPIO Y NO ./gemm_chained --n 16
-----------------------------------------------------
Dos razones concretas:
  1. gemm_chained no expone en ningun lado el resultado crudo de
     gpu_fp64_step(): solo publica rel_l2/rel_linf de la ruta WMMA CONTRA esa
     referencia. Comparar rel_l2 contra NumPy no distingue "la referencia esta
     mal" de "la ruta WMMA esta mal" -- justo la ambiguedad que hay que romper.
  2. parse_args() RECHAZA --n 16 y --n 32 (exige N multiplo de kBlockTileM=64,
     requisito de wmma_gemm_kernel, no de cuBLAS). El tamano chico que pide la
     verificacion solo es alcanzable fuera de ese binario.

POR QUE ES UN SCRIPT SEPARADO DE verificar_orden_operandos_conv.py
------------------------------------------------------------------
Se evaluo un unico script con --kernel gemm|conv. Se descarto: lo unico
compartido es la plomeria de detectar nvcc, compilar y correr (~60 lineas
mecanicas); TODO lo demas -- que simbolos extraer del .cu, el main() del
probe, y sobre todo la referencia NumPy (un producto denso NxN contra un
im2col con padding SAME sobre un filtro bloque-diagonal de 64 canales) -- es
disjunto. Un script unico seria un `if kernel == "gemm"` en cada paso. Ademas,
cada uno tiene que poder correrse y auditarse SOLO, dentro de una sesion de
PACCA, sin arrastrar un modulo compartido. Ver la nota gemela al final de
verificar_orden_operandos_conv.py.

USO
---
    python3 verificar_orden_operandos_gemm.py            # n = 32 (default)
    python3 verificar_orden_operandos_gemm.py --n 16
    python3 verificar_orden_operandos_gemm.py --conservar-tmp   # deja el probe

Codigos de salida:
    0  el orden de operandos es X*A (correcto).
    1  FALLA: el resultado no es X*A (o coincide con A*X / X^T*A).
    2  no se pudo verificar (falta nvcc/GPU, o no se pudo extraer la funcion).
       NO es un "pasa": es un "no se sabe", y se distingue a proposito del 1.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FUENTE_F3 = os.path.join(REPO_ROOT, "Fase_3", "GEMM", "gemm_chained.cu")
FUENTE_F4 = os.path.join(REPO_ROOT, "Fase_4", "GEMM", "gemm_chained.cu")

# Tolerancia sobre el error relativo en norma infinito entre el resultado de
# gpu_fp64_step y X@A calculado en NumPy (ambos en FP64). Un DGEMM de tamano
# 16-64 acumula del orden de n * eps_double ~ 1e-14; 1e-12 deja tres ordenes
# de holgura sobre eso y sigue estando ~10 ordenes por debajo de la diferencia
# que produciria un orden de operandos equivocado (donde el "error" relativo es
# de orden 1, no de orden 1e-12).
TOLERANCIA_REL_LINF = 1e-12


# ---------------------------------------------------------------------------
# Extraccion del codigo real del proyecto
# ---------------------------------------------------------------------------

def extraer_funcion(ruta: str, firma: str) -> str:
    """Devuelve el texto completo de una funcion C++ del archivo `ruta`.

    Busca la primera linea que contenga `firma` y devuelve desde ahi hasta la
    llave que cierra el cuerpo, contando llaves. Es un extractor deliberadamente
    simple: si algun dia la funcion tuviera una llave dentro de un literal de
    cadena o de un comentario, este conteo se rompe -- y falla RUIDOSAMENTE (no
    compila el probe), que es el modo de fallo correcto para un gate.
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

    # Avanza hasta la primera '{' del cuerpo y cuenta hasta cerrarlo.
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


def normalizar(texto: str) -> str:
    """Compara dos versiones de la misma funcion ignorando espacios en blanco."""
    return " ".join(texto.split())


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

def detectar_nvcc() -> str | None:
    if os.environ.get("NVCC"):
        return os.environ["NVCC"]
    return shutil.which("nvcc")


def detectar_cuda_arch() -> str:
    """Arquitectura de la GPU 0 via nvidia-smi; 80 (A100) si no se puede."""
    if os.environ.get("CUDA_ARCH"):
        return os.environ["CUDA_ARCH"]
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        cap = salida.strip().splitlines()[0].strip()  # p. ej. "8.0"
        mayor, menor = cap.split(".")
        return "%d%d" % (int(mayor), int(menor))
    except Exception:
        return "80"


def detectar_ccbin() -> str | None:
    """Compilador host para nvcc. Solo relevante en Windows.

    En Linux (PACCA, y cualquier maquina con el entorno conda de
    environment.yml) nvcc encuentra g++ solo y esta funcion devuelve None: no
    se pasa -ccbin y el comando queda identico al que usan los .sbatch.

    En Windows nvcc exige cl.exe en el PATH y falla con "Cannot find compiler
    'cl.exe'" si no lo esta. Localizarlo aqui permite correr ESTE gate (que es
    barato, chico y no necesita el resto del proyecto) en una maquina de
    desarrollo con GPU Ampere+, sin depender de una sesion de PACCA. No
    convierte a Windows en una plataforma soportada para las campanas -- ver
    REQUIREMENTS.md.
    """
    if os.environ.get("CCBIN"):
        return os.environ["CCBIN"]
    if os.name != "nt":
        return None
    if shutil.which("cl"):
        return None  # ya esta en el PATH; nvcc lo encuentra solo
    import glob as _glob
    patrones = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    ]
    candidatos = []
    for patron in patrones:
        candidatos.extend(d for d in _glob.glob(patron)
                          if os.path.isfile(os.path.join(d, "cl.exe")))
    # Version de toolset mas nueva (el orden lexicografico de 14.16/14.29/14.44
    # coincide con el cronologico dentro de una misma generacion).
    return sorted(candidatos)[-1] if candidatos else None


# ---------------------------------------------------------------------------
# Probe: binario minimo que expone el resultado de gpu_fp64_step
# ---------------------------------------------------------------------------

# El probe se arma con str.replace() sobre los marcadores @@...@@, NO con
# formateo `%` ni f-strings: el texto es C++ lleno de `%s`/`%d` de printf y de
# llaves, y cualquiera de los dos mecanismos habituales obligaria a escaparlos
# uno por uno (un `%` sin escapar reventaria el gate al generarlo, no al
# usarlo). Con replace(), el C++ va literal.
PLANTILLA_PROBE = r"""// GENERADO por Fase_3/tools/verificar_orden_operandos_gemm.py -- no editar.
//
// Binario minimo cuyo unico proposito es exponer el resultado de la funcion
// gpu_fp64_step() TAL COMO ESTA ESCRITA en el .cu del proyecto (pegada mas
// abajo sin modificar), para poder compararla contra NumPy.
#include <cstdio>
#include <cstdlib>
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
@@FUNCION@@
// ---- FIN del codigo extraido ----

static void leer_bin(const char* ruta, double* dst, size_t n) {
  std::FILE* fh = std::fopen(ruta, "rb");
  if (!fh) { std::fprintf(stderr, "no se pudo abrir %s\n", ruta); std::exit(72); }
  if (std::fread(dst, sizeof(double), n, fh) != n) {
    std::fprintf(stderr, "lectura corta en %s\n", ruta); std::exit(72);
  }
  std::fclose(fh);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "uso: %s N X.bin A.bin Y.bin\n", argv[0]);
    return 73;
  }
  const int n = std::atoi(argv[1]);
  const size_t count = (size_t)n * (size_t)n;

  std::vector<double> h_x(count), h_a(count), h_y(count);
  leer_bin(argv[2], h_x.data(), count);
  leer_bin(argv[3], h_a.data(), count);

  double *d_x = nullptr, *d_a = nullptr, *d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, count * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_y, count * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), count * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), count * sizeof(double), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  // LA llamada bajo prueba, con la misma firma que usa run_chained_route():
  //   gpu_fp64_step(handle, d_x_in, d_a, d_x_out, n)
  gpu_fp64_step(handle, d_x, d_a, d_y, n);
  CHECK_CUDA(cudaDeviceSynchronize());
  cublasDestroy(handle);

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, count * sizeof(double), cudaMemcpyDeviceToHost));
  std::FILE* out = std::fopen(argv[4], "wb");
  if (!out) { std::fprintf(stderr, "no se pudo escribir %s\n", argv[4]); return 74; }
  std::fwrite(h_y.data(), sizeof(double), count, out);
  std::fclose(out);

  cudaFree(d_x); cudaFree(d_a); cudaFree(d_y);
  return 0;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32,
                    help="tamano de la matriz cuadrada de prueba (default 32). "
                         "Chico a proposito: un orden de operandos equivocado se "
                         "delata igual a n=16 que a n=4096, y esto tiene que "
                         "poder correrse en segundos antes de una campana.")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--nvcc", default=None, help="ruta a nvcc (default: $NVCC o PATH)")
    ap.add_argument("--cuda-arch", default=None,
                    help="arquitectura sm_XX sin el prefijo (default: la de la GPU 0)")
    ap.add_argument("--ccbin", default=None,
                    help="compilador host para nvcc (-ccbin). Solo hace falta en "
                         "Windows, donde nvcc exige cl.exe; en Linux se ignora.")
    ap.add_argument("--conservar-tmp", action="store_true",
                    help="no borrar el directorio temporal con el probe generado")
    args = ap.parse_args()

    print("=" * 72)
    print("VERIFICACION DE ORDEN DE OPERANDOS -- GEMM (gpu_fp64_step)")
    print("=" * 72)

    # --- 1. Extraer la funcion real de los dos .cu y exigir que coincidan ---
    try:
        fn_f3 = extraer_funcion(FUENTE_F3, "static void gpu_fp64_step(")
        fn_f4 = extraer_funcion(FUENTE_F4, "static void gpu_fp64_step(")
    except (LookupError, OSError) as exc:
        print("NO SE PUDO VERIFICAR: %s" % exc, file=sys.stderr)
        print("  (cambio la firma de gpu_fp64_step? este gate hay que actualizarlo)",
              file=sys.stderr)
        return 2

    if normalizar(fn_f3) != normalizar(fn_f4):
        print("FALLA: gpu_fp64_step() NO es identica en Fase 3 y Fase 4.")
        print("  Fase_4/GEMM/README.md afirma que Fase 4 la 'reutiliza TAL CUAL'.")
        print("  Verificar una sola de las dos ya no cubre a la otra: revisar ambas")
        print("  a mano antes de confiar en cualquier campana.")
        return 1
    print("[1/4] gpu_fp64_step() extraida de Fase_3 y Fase_4/GEMM: identicas. OK")

    # --- 2. Toolchain ---
    nvcc = args.nvcc or detectar_nvcc()
    if not nvcc:
        print("NO SE PUDO VERIFICAR: no se encontro nvcc (ni $NVCC ni en PATH).",
              file=sys.stderr)
        print("  Este gate necesita compilar y correr en una GPU Ampere+ real: no hay",
              file=sys.stderr)
        print("  ruta de validacion local sin GPU (ver REQUIREMENTS.md).", file=sys.stderr)
        return 2
    arch = args.cuda_arch or detectar_cuda_arch()
    ccbin = args.ccbin or detectar_ccbin()
    print("[2/4] nvcc=%s  arch=sm_%s%s"
          % (nvcc, arch, ("  ccbin=%s" % ccbin) if ccbin else ""))

    n = args.n
    if n <= 0:
        print("--n debe ser positivo.", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="verif_gemm_")
    try:
        probe_cu = os.path.join(tmp, "probe_gemm.cu")
        probe_bin = os.path.join(tmp, "probe_gemm")
        with open(probe_cu, "w", encoding="utf-8") as fh:
            fh.write(PLANTILLA_PROBE
                     .replace("@@FUENTE@@", os.path.relpath(FUENTE_F3, REPO_ROOT).replace("\\", "/"))
                     .replace("@@FUNCION@@", fn_f3))

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
        print("[3/4] probe compilado (funcion del proyecto pegada sin modificar).")

        # --- 3. Datos de prueba ---
        # X y A aleatorias y ASIMETRICAS: con matrices simetricas, X*A y (A*X)^T
        # coinciden y el gate no distinguiria nada. Con estas, X@A, A@X y X.T@A
        # son tres matrices claramente distintas.
        rng = np.random.default_rng(args.seed)
        X = rng.standard_normal((n, n))
        A = rng.standard_normal((n, n))

        x_bin = os.path.join(tmp, "X.bin")
        a_bin = os.path.join(tmp, "A.bin")
        y_bin = os.path.join(tmp, "Y.bin")
        # C-order = row-major, la misma convencion en la que gemm_chained.cu
        # guarda X y A en memoria.
        X.astype(np.float64).tofile(x_bin)
        A.astype(np.float64).tofile(a_bin)

        proc = subprocess.run([probe_bin, str(n), x_bin, a_bin, y_bin],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("NO SE PUDO VERIFICAR: el probe fallo en ejecucion (codigo %d)."
                  % proc.returncode, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 2

        Y = np.fromfile(y_bin, dtype=np.float64).reshape(n, n)

        # --- 4. Comparacion contra NumPy ---
        # La referencia se calcula EXPLICITAMENTE como X @ A (no A @ X), que es
        # la operacion que el encadenamiento X(n+1) = X(n)*A tiene que hacer.
        esperado = X @ A
        candidatos = {
            "X @ A  (correcto)": esperado,
            "A @ X  (operandos invertidos)": A @ X,
            "X.T @ A (X transpuesta)": X.T @ A,
            "(X @ A).T (resultado transpuesto)": esperado.T,
        }

        def rel_linf(ref, test):
            denom = np.max(np.abs(ref))
            return float(np.max(np.abs(ref - test)) / denom) if denom > 0 else float("inf")

        print("[4/4] Distancia relativa (norma infinito) a cada hipotesis:")
        for nombre, cand in candidatos.items():
            print("        %-36s %.3e" % (nombre, rel_linf(cand, Y)))

        err = rel_linf(esperado, Y)
        print()
        if err <= TOLERANCIA_REL_LINF:
            print("PASA: gpu_fp64_step calcula X*A (rel_linf = %.3e <= %.1e)."
                  % (err, TOLERANCIA_REL_LINF))
            print("      El orden invertido de los argumentos de cublasDgemm compensa")
            print("      correctamente la convencion column-major. La trayectoria de")
            print("      referencia FP64 de Fase 3 y Fase 4 es la que el plan pide.")
            return 0

        print("FALLA: gpu_fp64_step NO calcula X*A (rel_linf = %.3e > %.1e)."
              % (err, TOLERANCIA_REL_LINF))
        for nombre, cand in candidatos.items():
            if nombre.startswith("X @ A"):
                continue
            if rel_linf(cand, Y) <= TOLERANCIA_REL_LINF:
                print("       Lo que SI esta calculando es: %s" % nombre)
        print()
        print("       Este es exactamente el error silencioso que el comentario de")
        print("       cabecera de gemm_chained.cu advierte. NINGUNA columna rel_l2 /")
        print("       rel_linf de una campana de GEMM (Fase 3 o Fase 4) es valida")
        print("       mientras esto falle: la 'referencia FP64' contra la que se")
        print("       mide todo no es la misma operacion que calcula la ruta WMMA.")
        return 1
    finally:
        if args.conservar_tmp:
            print("\n(probe conservado en %s)" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
