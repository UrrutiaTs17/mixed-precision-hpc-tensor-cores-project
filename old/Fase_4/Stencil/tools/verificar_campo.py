#!/usr/bin/env python3
"""Compara campos archivados y reporta la POSICION de la maxima diferencia.

ErrorMetrics (Fase_2/common.cuh) guarda max_abs, rel_l2 y rel_linf, pero no
DONDE ocurre el maximo. Anadir esa posicion obligaria a tocar
compare_fp64_ref_vs_fp32, que comparten todas las rutas del binario. En vez de
eso, este script trabaja sobre los volcados crudos que ya produce
--archive-iters: reconstruye los campos, los resta y localiza el maximo.

Sirve para diagnosticar la formulacion Y = X H + V X: si un error de indice en
las cuatro bandas exteriores se colara, la maxima diferencia caeria
sistematicamente en el borde de los tiles de 16x16 (x-1 o y-1 multiplo de 16),
y no repartida por el interior. El script clasifica cada maximo por esa
propiedad.

Formato de los .bin: row-major little-endian sin cabecera. El dtype se deduce
del tamano del fichero (8 bytes/celda = float64, 4 = float32).

Uso:
    verificar_campo.py --dir <archivo> --nx 63 --ny 65
    verificar_campo.py --dir <archivo> --nx 63 --ny 65 --ref CPU_FP64
"""

import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    print("numpy no disponible; se omite la verificacion de campo.")
    sys.exit(0)


def cargar(ruta, nx, ny):
    n = nx * ny
    tam = os.path.getsize(ruta)
    if tam == n * 8:
        dt = np.float64
    elif tam == n * 4:
        dt = np.float32
    else:
        raise ValueError(
            f"{os.path.basename(ruta)}: {tam} bytes no cuadra con {nx}x{ny} "
            f"en float32 ({n*4}) ni float64 ({n*8})")
    return np.fromfile(ruta, dtype=dt).astype(np.float64).reshape(ny, nx), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directorio de --archive-dir")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--ref", default="CPU_FP64", help="prefijo de la ruta de referencia")
    ap.add_argument("--tile", type=int, default=16, help="lado del tile WMMA")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"  (no existe {args.dir}; se omite)")
        return 0

    bins = sorted(f for f in os.listdir(args.dir) if f.endswith(".bin"))
    if not bins:
        print(f"  (sin .bin en {args.dir}; se omite)")
        return 0

    refs = [f for f in bins if f.startswith(args.ref)]
    if not refs:
        print(f"  (sin campo de referencia {args.ref}_* en {args.dir})")
        print(f"  disponibles: {', '.join(bins)}")
        return 0
    ref_name = refs[0]
    ref, ref_dt = cargar(os.path.join(args.dir, ref_name), args.nx, args.ny)
    print(f"  referencia: {ref_name}  ({np.dtype(ref_dt).name})")

    ref_linf = np.max(np.abs(ref[np.isfinite(ref)])) if np.any(np.isfinite(ref)) else 0.0

    for name in bins:
        if name == ref_name:
            continue
        try:
            campo, dt = cargar(os.path.join(args.dir, name), args.nx, args.ny)
        except ValueError as e:
            print(f"  {name}: {e}")
            continue

        finito = np.isfinite(campo).all()
        d = np.abs(campo - ref)
        d[~np.isfinite(d)] = 0.0
        idx = int(np.argmax(d))
        iy, ix = divmod(idx, args.nx)
        max_abs = float(d[iy, ix])
        rel_linf = max_abs / ref_linf if ref_linf > 0 else float("nan")

        # ¿El maximo cae en el borde de un tile? Los tiles interiores empiezan
        # en x=1,y=1 y miden args.tile, asi que las celdas de borde de tile son
        # aquellas con (ix-1) % tile in {0, tile-1}.
        bx = (ix - 1) % args.tile in (0, args.tile - 1)
        by = (iy - 1) % args.tile in (0, args.tile - 1)
        borde_malla = ix in (0, args.nx - 1) or iy in (0, args.ny - 1)
        if borde_malla:
            donde = "borde fisico de la malla (no se recalcula)"
        elif bx or by:
            donde = "BORDE DE TILE  <- revisar las bandas exteriores"
        else:
            donde = "interior de tile"

        print(f"  {name:<34} max|d|={max_abs:.6e}  rel_linf={rel_linf:.6e}")
        print(f"      posicion (x={ix}, y={iy}) -> {donde}")
        print(f"      finito: {'si' if finito else 'NO'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
