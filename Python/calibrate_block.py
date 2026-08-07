"""Script de calibração — bloco 16×16 da manta FSR.

Como usar:
    cd "Projeto Walter"
    .\venv\Scripts\activate   (Windows)
    python Python/calibrate_block.py --block 1

Coloque pesos conhecidos sobre a placa rígida posicionada no bloco escolhido.
O script guia cada etapa interativamente.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Resolve imports do projeto
sys.path.insert(0, str(Path(__file__).parent))
from config.settings import HID_VID, HID_PID, HID_PACKET_SIZE, HID_ROWS, HID_COLS

try:
    import hid
except ImportError:
    print("[ERRO] hidapi nao instalado. Execute: pip install hidapi")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Mapeamento block_id → slice da matriz global 32×64
# Espelha o mapeamento ButtonShowDeal do C# (mesmo do HidFrameReader)
# ---------------------------------------------------------------------------

BLOCK_REGIONS: dict[int, tuple[slice, slice]] = {
    1: (slice(16, 32), slice(0,  16)),
    2: (slice(16, 32), slice(16, 32)),
    3: (slice(16, 32), slice(32, 48)),
    4: (slice(16, 32), slice(48, 64)),
    5: (slice(0,  16), slice(48, 64)),
    6: (slice(0,  16), slice(32, 48)),
    7: (slice(0,  16), slice(16, 32)),
    8: (slice(0,  16), slice(0,  16)),
}

SAMPLE_SECONDS = 5      # segundos de coleta estável por ponto de peso
RENDER_INTERVAL = 0.025  # 25 ms — mesma cadência do sistema principal


# ---------------------------------------------------------------------------
# Parse HID — idêntico ao HidFrameReader._parse_hid_packet
# ---------------------------------------------------------------------------

def parse_packet(data: bytes, matrix: np.ndarray) -> None:
    block_id = data[0]
    if block_id == 0 or block_id > 8:
        return
    for i in range(1, len(data) - 1, 3):
        if i + 2 >= len(data):
            break
        x_local = data[i]
        y_local = data[i + 1]
        pressure = data[i + 2]
        if x_local == 0 or y_local == 0:
            break
        if block_id < 5:
            x = 16 * ((9 - block_id) // 5) + (16 - x_local)
            y = 16 * (block_id - 1) + (16 - y_local)
        else:
            x = 16 * ((9 - block_id) // 5) + (x_local - 1)
            y = 16 * (8 - block_id) + (y_local - 1)
        if 0 <= x < HID_ROWS and 0 <= y < HID_COLS:
            matrix[x, y] = float(pressure)


def read_block_sum_stable(device: hid.device, block_id: int, duration_s: float) -> float:
    """Drena o buffer HID por `duration_s` segundos e retorna a média da soma do bloco."""
    matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    row_sl, col_sl = BLOCK_REGIONS[block_id]
    samples: list[float] = []
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        frame_start = time.monotonic()
        try:
            while True:
                raw = device.read(HID_PACKET_SIZE)
                if not raw:
                    break
                parse_packet(bytes(raw), matrix)
        except OSError:
            break
        samples.append(float(matrix[row_sl, col_sl].sum()))
        remaining = RENDER_INTERVAL - (time.monotonic() - frame_start)
        if remaining > 0:
            time.sleep(remaining)

    return float(np.mean(samples)) if samples else 0.0


# ---------------------------------------------------------------------------
# Calibração principal
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibração de bloco FSR 16x16")
    p.add_argument("--block", type=int, choices=range(1, 9), default=1,
                   help="Block ID a calibrar (1-8, padrão: 1)")
    p.add_argument("--output", type=str, default="calibration.json",
                   help="Arquivo de saída (padrão: calibration.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    block_id = args.block
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print(f"  CALIBRAÇÃO — Bloco {block_id}  (região: {BLOCK_REGIONS[block_id]})")
    print(f"{'='*60}")
    print("""
PREPARAÇÃO FÍSICA:
  1. Posicione a placa RÍGIDA E PLANA exatamente sobre o bloco.
     (placa de MDF, acrílico ou metal com ≥10mm de espessura)
  2. A placa deve cobrir toda a área 16×16 do bloco.
  3. Nenhum peso sobre a placa ainda.
  4. HID conectado ao USB.
""")
    input("  Pressione Enter quando pronto...")

    # Conectar HID
    try:
        device = hid.device()
        device.open(HID_VID, HID_PID)
        device.set_nonblocking(True)
        print(f"\n[OK] HID conectado — VID=0x{HID_VID:04X} PID=0x{HID_PID:04X}\n")
    except OSError as err:
        print(f"\n[ERRO] Falha ao abrir HID: {err}")
        sys.exit(1)

    weight_kg_list: list[float] = []
    raw_sum_list: list[float] = []

    print("Digite os pesos disponíveis UM A UM.")
    print("Sugestão de pontos: 0, 5, 10, 15, 20, 25, 30 kg")
    print("Deixe em branco e Enter para finalizar.\n")

    while True:
        entrada = input("Próximo peso (kg) — ou Enter para finalizar: ").strip()
        if entrada == "":
            if len(weight_kg_list) < 3:
                print("  [AVISO] Adicione ao menos 3 pontos para um ajuste de curva válido.\n")
                continue
            break

        try:
            kg = float(entrada.replace(",", "."))
        except ValueError:
            print("  Valor inválido. Digite um número (ex: 10 ou 10.5)\n")
            continue

        if kg == 0:
            input("\n  [AÇÃO] Retire todos os pesos da placa. Pressione Enter para medir TARA...")
        else:
            input(f"\n  [AÇÃO] Coloque {kg:.1f} kg sobre a placa. Aguarde estabilizar (~5s) e pressione Enter...")

        print(f"  Coletando amostras por {SAMPLE_SECONDS}s...", end="", flush=True)
        media = read_block_sum_stable(device, block_id, SAMPLE_SECONDS)
        print(f"  ✓  sum_bruto = {media:.1f}")

        weight_kg_list.append(kg)
        raw_sum_list.append(media)
        print(f"  Ponto salvo: {kg:.1f} kg → sum={media:.1f}\n")

    device.close()

    # ---------------------------------------------------------------------------
    # Ajuste de curva polinomial
    # ---------------------------------------------------------------------------
    weights = np.array(weight_kg_list)
    sums_raw = np.array(raw_sum_list)

    # Tara: soma bruta do bloco com 0 kg
    zero_idx = np.where(weights == 0.0)[0]
    tare_sum = float(sums_raw[zero_idx[0]]) if len(zero_idx) > 0 else 0.0
    net_sums = sums_raw - tare_sum  # subtrai tara → ponto 0 vira origem

    # Polinômio grau 2 (ou 1 se poucos pontos): net_sum → kg
    degree = min(2, len(weights) - 1)
    coeffs = np.polyfit(net_sums, weights, deg=degree).tolist()
    poly = np.poly1d(coeffs)

    # RMSE de validação
    predicted = poly(net_sums)
    rmse = float(np.sqrt(np.mean((predicted - weights) ** 2)))

    print("\n" + "="*60)
    print("  RESULTADO")
    print("="*60)
    print(f"  Tara (sum com 0 kg):      {tare_sum:.1f}")
    print(f"  Grau do polinômio:        {degree}")
    print(f"  Coeficientes:             {[round(c, 8) for c in coeffs]}")
    print(f"  RMSE:                     {rmse:.3f} kg")
    if rmse > 1.5:
        print("  [AVISO] RMSE > 1.5 kg — adicione mais pontos intermediários.")

    print("\n  Verificação ponto a ponto:")
    print(f"  {'Real (kg)':>10} | {'Net Sum':>10} | {'Predito':>10} | {'Erro':>8}")
    print("  " + "-"*46)
    for kg_r, ns, pred in zip(weights, net_sums, predicted):
        print(f"  {kg_r:>10.1f} | {ns:>10.1f} | {pred:>10.2f} | {kg_r - pred:>+8.3f}")

    # ---------------------------------------------------------------------------
    # Salvar JSON
    # ---------------------------------------------------------------------------
    output = {
        "block_id": block_id,
        "tare_block_sum": tare_sum,
        "polynomial_degree": degree,
        "coefficients": coeffs,
        "rmse_kg": round(rmse, 4),
        "calibration_points": [
            {"kg": float(k), "raw_sum": float(s)}
            for k, s in zip(weights, sums_raw)
        ],
        "usage": (
            "kg_bloco = np.polyval(coefficients, raw_sum_bloco - tare_block_sum). "
            "Para manta inteira: calcular kg_bloco para cada bloco e somar."
        ),
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[OK] Salvo em: {output_path.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
