"""Migra calibration.json de v2 (coeficientes em kg) para v3 (coeficientes em Newton).

Uso:
    python Python/scripts/migrate_calib_v2_to_v3.py
    python Python/scripts/migrate_calib_v2_to_v3.py --input calibration.json --output calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GRAVITY_M_S2: float = 9.81


def migrate_v2_to_v3(data: dict) -> dict:
    """Converte estrutura v2 (kg) para v3 (newton) in-place."""
    if data.get("version", 1) >= 3:
        print("[INFO] Arquivo já está em v3 ou superior — nada a fazer.")
        return data

    for bid_str, block in data.get("blocks", {}).items():
        pts = block.get("calibration_points", [])

        # Adiciona campo 'n' em cada ponto
        for pt in pts:
            pt["n"] = round(pt["kg"] * GRAVITY_M_S2, 4)

        # Re-ajusta curva com pontos em Newton
        tare = block.get("tare_block_sum", 0.0)
        raw_sums = np.array([p["raw_sum"] for p in pts])
        n_vals = np.array([p["n"] for p in pts])
        net_sums = raw_sums - tare

        degree = min(2, len(pts) - 1)
        coeffs_n = np.polyfit(net_sums, n_vals, deg=degree).tolist()

        # Calcula RMSE em Newton
        predicted_n = np.polyval(coeffs_n, net_sums)
        rmse_n = float(np.sqrt(np.mean((predicted_n - n_vals) ** 2)))
        rmse_kg = rmse_n / GRAVITY_M_S2

        # Atualiza bloco para v3
        block["unit"] = "newton"
        block["coefficients_raw_to_n"] = coeffs_n
        block["rmse_n"] = round(rmse_n, 4)
        block["rmse_kg"] = round(rmse_kg, 4)

        # Remove campo legado
        block.pop("coefficients", None)

        print(f"  Bloco {bid_str}: {len(pts)} pontos migrados | "
              f"RMSE = {rmse_n:.4f} N ({rmse_kg:.4f} kg)")

    data["version"] = 3
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Migra calibration.json v2 → v3")
    parser.add_argument("--input", type=str, default="calibration.json",
                        help="Arquivo de entrada (padrão: calibration.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Arquivo de saída (padrão: sobrescreve o input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"[INFO] Lido: {input_path} (versão {data.get('version', 1)})")

    migrated = migrate_v2_to_v3(data)

    output_path.write_text(
        json.dumps(migrated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[OK] Salvo em: {output_path} (versão {migrated['version']})")


if __name__ == "__main__":
    main()
