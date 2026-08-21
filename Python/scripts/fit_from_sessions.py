"""Consolidação e ajuste de curva multi-sessões (Fase 2 / Metodologia).

Lê todos os CSVs de calibração na pasta sessions/ para um bloco (ou todos os blocos),
agrega os dados de múltiplas repetições e posições (centro, direita, etc.),
ajusta a melhor curva polinomial robusta e salva no calibration.json com metadados.

Uso:
    python Python/scripts/fit_from_sessions.py --block 1
    python Python/scripts/fit_from_sessions.py --all-blocks
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.calibration_store import GRAVITY_M_S2


def _load_points_for_block(sessions_dir: Path, block_id: int) -> tuple[list[float], list[float], float, list[str]]:
    raw_sums: list[float] = []
    reference_kgs: list[float] = []
    tare_sums: list[float] = []
    source_files: list[str] = []

    for csv_path in sorted(sessions_dir.glob(f"calib_*_bloco{block_id}_*.csv")):
        source_files.append(csv_path.name)
        try:
            with csv_path.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg = row.get("reference_kg")
                    raw_sum = row.get("raw_sum_block")
                    tare = row.get("tare_block_sum")
                    if ref_kg is None or raw_sum is None or ref_kg == "" or raw_sum == "":
                        continue
                    r_kg = float(ref_kg)
                    r_sum = float(raw_sum)
                    t_sum = float(tare) if tare else 0.0

                    if r_kg == 0.0:
                        tare_sums.append(r_sum)

                    raw_sums.append(r_sum)
                    reference_kgs.append(r_kg)
        except OSError as err:
            print(f"[AVISO] Falha ao ler {csv_path.name}: {err}")

    mean_tare = float(np.mean(tare_sums)) if tare_sums else 0.0
    return raw_sums, reference_kgs, mean_tare, source_files


def fit_and_update_block(
    sessions_dir: Path,
    block_id: int,
    output_json: Path,
) -> bool:
    raw_sums, weights_kg, tare_sum, sources = _load_points_for_block(sessions_dir, block_id)

    if len(weights_kg) < 3:
        print(f"[AVISO] Bloco {block_id}: Poucos pontos ({len(weights_kg)} encontrados). Mínimo 3.")
        return False

    weights_arr = np.array(weights_kg)
    sums_arr = np.array(raw_sums)

    net_sums = sums_arr - tare_sum
    forces_n = weights_arr * GRAVITY_M_S2

    degree = min(2, len(weights_arr) - 1)
    if degree == 2:
        A = np.column_stack([net_sums ** 2, net_sums])
        fit_res, _, _, _ = np.linalg.lstsq(A, forces_n, rcond=None)
        coeffs_n = [float(fit_res[0]), float(fit_res[1]), 0.0]
    else:
        A = net_sums[:, np.newaxis]
        fit_res, _, _, _ = np.linalg.lstsq(A, forces_n, rcond=None)
        coeffs_n = [float(fit_res[0]), 0.0]

    predicted_n = np.polyval(coeffs_n, net_sums)
    rmse_n = float(np.sqrt(np.mean((predicted_n - forces_n) ** 2)))
    rmse_kg = rmse_n / GRAVITY_M_S2

    # Atualiza JSON com metadados completos
    data = {"version": 3, "blocks": {}}
    if output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    data["version"] = 3
    if "blocks" not in data:
        data["blocks"] = {}

    data["blocks"][str(block_id)] = {
        "unit": "newton",
        "calibrated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_sessions": sources,
        "samples_count": len(weights_kg),
        "tare_block_sum": round(tare_sum, 4),
        "polynomial_degree": degree,
        "coefficients_raw_to_n": coeffs_n,
        "rmse_n": round(rmse_n, 4),
        "rmse_kg": round(rmse_kg, 4),
        "calibration_points": [
            {"kg": float(k), "n": round(float(k) * GRAVITY_M_S2, 4), "raw_sum": float(s)}
            for k, s in zip(weights_arr, sums_arr)
        ],
    }

    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Bloco {block_id} consolidado com sucesso a partir de {len(sources)} sessões:")
    for src in sources:
        print(f"     - {src}")
    print(f"  Total de amostras:  {len(weights_kg)}")
    print(f"  Tara consolidada:   {tare_sum:.2f}")
    print(f"  RMSE multi-sessão:  {rmse_n:.4f} N ({rmse_kg:.4f} kg)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidação Multi-Sessões de Calibração")
    parser.add_argument("--block", type=int, default=1, choices=range(1, 9),
                        help="ID do bloco a consolidar (padrão: 1)")
    parser.add_argument("--all-blocks", action="store_true",
                        help="Consolida todos os blocos de 1 a 8 que tiverem sessões")
    parser.add_argument("--sessions-dir", type=str, default="sessions",
                        help="Diretório de sessões CSV (padrão: sessions)")
    parser.add_argument("--output", type=str, default="calibration.json",
                        help="Arquivo calibration.json de saída (padrão: calibration.json)")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    output_path = Path(args.output)

    print(f"\n{'='*70}")
    print("  CONSOLIDAÇÃO MULTI-SESSÕES (Metodologia Fase 2)")
    print(f"{'='*70}")

    if args.all_blocks:
        for bid in range(1, 9):
            fit_and_update_block(sessions_dir, bid, output_path)
    else:
        fit_and_update_block(sessions_dir, args.block, output_path)

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
