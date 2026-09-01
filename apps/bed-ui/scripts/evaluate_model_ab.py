"""Comparação entre Modelo A (Soma Direta) e Modelo B (Integração 2D Trapezoidal).

Metodologia §15 e §16:
  - Modelo A: F_A = Σ_k F_k(soma_k)  →  m_A = F_A / g
  - Modelo B: F_B ≈ ∬ p(x,y) dA      →  m_B = F_B / g

Uso:
    python Python/scripts/evaluate_model_ab.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import HID_ROWS, HID_COLS
from services.calibration_store import load_calibration, GRAVITY_M_S2, BLOCK_REGIONS
from services.math_pipeline import compute_model_a, compute_model_b


def _generate_synthetic_test_matrices() -> list[tuple[str, float, np.ndarray]]:
    """Gera matrizes com distribuições conhecidas de carga para comparar A vs B."""
    cases = []

    # Caso 1: Carga concentrada centralizada no Bloco 1 (16x16)
    m1 = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    r_sl, c_sl = BLOCK_REGIONS[1]
    m1[r_sl, c_sl] = 50.0  # uniforme
    cases.append(("Bloco 1 Uniforme (Carga Central)", 5.0, m1))

    # Caso 2: Carga concentrada em borda (gradiente)
    m2 = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    for r in range(16, 32):
        for c in range(0, 16):
            m2[r, c] = float((r - 16) * 4 + (c * 2))
    cases.append(("Bloco 1 Gradiente de Borda", 3.0, m2))

    # Caso 3: Carga pontual com pico no centro
    m3 = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    m3[24, 8] = 200.0
    m3[23:26, 7:10] = 80.0
    cases.append(("Carga Pontual Concentrada (Pico)", 2.0, m3))

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação Modelo A vs Modelo B")
    parser.add_argument("--calib", type=str, default="calibration.json",
                        help="Caminho do calibration.json (padrão: calibration.json)")
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    print(f"\n{'='*75}")
    print("  COMPARAÇÃO METROLÓGICA: MODELO A (Soma) vs MODELO B (Integração 2D)")
    print(f"{'='*75}")
    print(f"Arquivo de Calibração: {args.calib}")
    print(f"Status da Calibração: {'VÁLIDA (' + str(len(calib.blocks)) + ' bloco(s))' if calib.is_valid else 'INVÁLIDA'}\n")

    if not calib.is_valid:
        print("[ERRO] Nenhuma calibração válida carregada. Execute calibrate_block.py primeiro.")
        sys.exit(1)

    cases = _generate_synthetic_test_matrices()

    print(f"{'Cenario de Distribuicao':<35} {'Modelo A (kg)':>14} {'Modelo B (kg)':>14} {'Delta (kg)':>10}")
    print("-" * 77)

    for desc, ref_kg, mat in cases:
        ma = compute_model_a(mat, calib)
        mb = compute_model_b(mat, calib)
        delta = mb - ma
        print(f"{desc:<35} {ma:>14.3f} {mb:>14.3f} {delta:>+10.3f}")

    print("\n" + "="*77)
    print("ANALISE DE CONCLUSAO (Metodologia §16):")
    print("  - Em cargas uniformes ou bem distribuidas: Modelo A ~ Modelo B (Delta ~ 0).")
    print("  - Em cargas com forte gradiente espacial: Modelo B aplica a ponderacao 2D.")
    print("="*77 + "\n")


if __name__ == "__main__":
    main()
