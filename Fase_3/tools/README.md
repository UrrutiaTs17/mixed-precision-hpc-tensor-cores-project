# Fase 3 — tools

Post-proceso de CSV para los binarios de Fase 3. Dos scripts, dos esquemas — **no comparten código** porque los binarios que los alimentan emiten columnas distintas:

| Script | Para | Esquema de entrada |
|---|---|---|
| `extract_csv.py` | `Fase_3/Stencil/stencil_tensor_activation.cu` | `CSV_DRIFT`/`CSV_SUMMARY`/`CSV_ONSET`/`CSV_HORIZON`/`CSV_STORE`/`CSV_ENERGY`, columnas `nx`/`ny`/`kahan`/`route` |
| `extract_csv_chained.py` | `gemm_chained.cu` (`Fase_3/GEMM`) y `conv_chained.cu` (`Fase_3/Convolution`) | Solo `CSV_DRIFT`/`CSV_SUMMARY`, columnas `n`/`hw` (unificadas aquí como `size`) |

Ambos se invocan igual, con `--kernel` seleccionando el esquema y el nombre de archivo de salida:

```bash
python3 extract_csv.py --input run_123.log --outdir results --job-id 123 --kernel stencil
python3 extract_csv_chained.py --input run_456.log --outdir results --job-id 456 --kernel gemm
python3 extract_csv_chained.py --input run_789.log --outdir results --job-id 789 --kernel conv
```

Los `.sbatch` de cada carpeta (`Fase_3/Stencil/`, `Fase_3/GEMM/`, `Fase_3/Convolution/`) los invocan automáticamente al terminar la corrida, buscándolos en `../tools/` — ver el bloque final de cualquiera de esos `.sbatch`.

## Por qué `extract_csv_chained.py` reconstruye `anchor_every` desde una línea de cabecera

`gemm_chained.cu` y `conv_chained.cu` no escriben `anchor_every` como columna del `CSV_SUMMARY`/`CSV_DRIFT` en sí (pendiente — ver "Qué falta" de `Fase_4/GEMM/README.md` y `Fase_4/Convolution/README.md`). Por eso el script lee la línea de configuración que ambos binarios imprimen al arrancar cada corrida (`N=... anchor_every=K ...` / `HW=... anchor_every=K ...`) y la propaga a las filas de la ruta `_comp` que le siguen — la ruta `_none` nunca usa el ancla, sin importar con qué `--anchor-every` se haya lanzado la corrida (el binario exige `--comp on` para `--anchor-every>0`).

## Qué falta

- Migrar los scripts de gate (`comparar_gate1.py`, `validar_gate2.py`, `gate1_regresion.sbatch` — hoy solo en `old/Fase_4/Stencil/`) y escribir sus equivalentes para el ancla de GEMM/Conv (los gates K=0/K=1 que las tres READMEs de ancla piden correr antes de confiar en cualquier resultado).
- Agregar `anchor_every` como columna real del CSV en los tres binarios, para no depender de reconstruirla desde una línea de texto.
