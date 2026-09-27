#!/usr/bin/env python3
"""tools/variabilidad_relabel_job_id.py

Reescribe la columna job_id de los CSV de la campana de variabilidad
lanzada DENTRO de un holder compartido (tools/variabilidad_dentro_holder.sh)
a un valor sintetico unico por replica: "<job_id_original>-r<N>".

POR QUE HACE FALTA: cada replica corre como `srun --jobid=<holder>`, asi
que TODAS heredan el MISMO SLURM_JOB_ID -- pero run_statistics.py cuenta
replicas independientes por job_id distinto (Dataset.cells(),
n_replicas=nunique(job_id)). Sin este paso, la campana entera contaria
como "1 sola replica" pese a ser mediciones genuinamente independientes
(cada srun relanza el binario completo desde cero, misma garantia de
independencia que un sbatch separado -- lo unico compartido es la
ETIQUETA de asignacion SLURM, no la ejecucion).

Localiza las replicas por convencion de directorio: cada replica escribe
en su propio "results/variabilidad/r<N>/" (ver variabilidad_dentro_holder.sh),
asi que el numero de replica sale de esa ruta, no de ningun dato dentro
del CSV.

Uso:
    python3 tools/variabilidad_relabel_job_id.py \
        Fase_4/GEMM/results/variabilidad \
        Fase_4/Convolution/results/variabilidad \
        Fase_4/Stencil/results/variabilidad

Modifica los CSV IN PLACE. Es idempotente: si ya se corrio antes sobre el
mismo archivo, job_id ya trae el sufijo "-rN" y no se toca de nuevo
(evita "-r1-r1" si se corre dos veces por accidente).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPLICA_DIR_RE = re.compile(r"^r(\d+)$")


def find_replica_csvs(vardir: Path) -> list[tuple[Path, str]]:
    """Devuelve [(ruta_csv, sufijo_replica), ...] para todos los CSV bajo
    vardir/r<N>/**/*.csv."""
    out = []
    if not vardir.is_dir():
        return out
    for rdir in sorted(vardir.iterdir()):
        m = REPLICA_DIR_RE.match(rdir.name)
        if not m or not rdir.is_dir():
            continue
        rep = m.group(1)
        for csv_path in rdir.rglob("*.csv"):
            out.append((csv_path, rep))
    return out


def relabel(csv_path: Path, rep: str) -> bool:
    df = pd.read_csv(csv_path, dtype=str)
    if "job_id" not in df.columns or df.empty:
        return False
    suffix = f"-r{rep}"
    already_done = df["job_id"].astype(str).str.endswith(suffix).all()
    if already_done:
        return False
    df["job_id"] = df["job_id"].astype(str).apply(
        lambda v: v if v.endswith(suffix) else f"{v}{suffix}"
    )
    df.to_csv(csv_path, index=False)
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print("Uso: variabilidad_relabel_job_id.py <vardir> [<vardir> ...]", file=sys.stderr)
        return 1
    total = 0
    for arg in argv:
        vardir = Path(arg)
        pairs = find_replica_csvs(vardir)
        for csv_path, rep in pairs:
            if relabel(csv_path, rep):
                total += 1
                print(f"[relabel] {csv_path} -> job_id con sufijo -r{rep}")
    print(f"Total de CSV reetiquetados: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
