"""Script de calibração — bloco 16×16 da manta FSR.

Como usar:
    cd "Projeto Walter"
    .\\venv\\Scripts\\activate   (Windows)
    python Python/calibrate_block.py --block 1

Todos os setores podem estar conectados simultaneamente.
O script isola a leitura do bloco alvo por slice da matriz.
Cada execução atualiza APENAS o bloco calibrado no calibration.json,
preservando os dados dos demais blocos já salvos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import HID_VID, HID_PID, HID_PACKET_SIZE, HID_ROWS, HID_COLS

try:
    import hid
except ImportError:
    print("[ERRO] hidapi nao instalado. Execute: pip install hidapi")
    sys.exit(1)


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

SAMPLE_SECONDS = 5
RENDER_INTERVAL = 0.025


# ---------------------------------------------------------------------------
# HID — parse idêntico ao HidFrameReader
# ---------------------------------------------------------------------------

def _parse_packet(data: bytes, matrix: np.ndarray) -> None:
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


def _read_block_sum(device: hid.device, block_id: int, duration_s: float) -> float:
    """Drena o buffer HID por `duration_s` segundos e retorna a média do bloco alvo.

    Outros blocos podem estar ativos — são ignorados via slice.
    """
    matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    row_sl, col_sl = BLOCK_REGIONS[block_id]
    samples: list[float] = []
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        t0 = time.monotonic()
        try:
            while True:
                raw = device.read(HID_PACKET_SIZE)
                if not raw:
                    break
                _parse_packet(bytes(raw), matrix)
        except OSError:
            break
        samples.append(float(matrix[row_sl, col_sl].sum()))
        rem = RENDER_INTERVAL - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

    return float(np.mean(samples)) if samples else 0.0


# ---------------------------------------------------------------------------
# Persistência — formato multi-bloco v2
# ---------------------------------------------------------------------------

def _load_existing(path: Path) -> dict:
    """Carrega calibration.json existente (v1 ou v2) como dict interno."""
    if not path.exists():
        return {"version": 2, "blocks": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [AVISO] {path.name} corrompido — será recriado.")
        return {"version": 2, "blocks": {}}

    # Migração: formato v1 (single-block) → v2 (multi-block)
    if "version" not in data and "block_id" in data:
        bid = str(data["block_id"])
        migrated: dict = {"version": 2, "blocks": {bid: {
            "tare_block_sum": data.get("tare_block_sum", 0.0),
            "polynomial_degree": data.get("polynomial_degree", 2),
            "coefficients": data.get("coefficients", []),
            "rmse_kg": data.get("rmse_kg", 0.0),
            "calibration_points": data.get("calibration_points", []),
        }}}
        print(f"  [INFO] calibration.json v1 migrado para v2 (bloco {bid} preservado).")
        return migrated

    return data


def _save_block(path: Path, block_id: int, block_data: dict) -> None:
    """Atualiza apenas a entrada do bloco `block_id` no JSON."""
    existing = _load_existing(path)
    existing["blocks"][str(block_id)] = block_data
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Ajuste de curva
# ---------------------------------------------------------------------------

def _fit_curve(
    weights: np.ndarray,
    sums_raw: np.ndarray,
    tare_sum: float,
) -> tuple[list[float], float, np.ndarray]:
    """Retorna (coefficients, rmse_kg, net_sums)."""
    net = sums_raw - tare_sum
    degree = min(2, len(weights) - 1)
    coeffs = np.polyfit(net, weights, deg=degree).tolist()
    predicted = np.polyval(coeffs, net)
    rmse = float(np.sqrt(np.mean((predicted - weights) ** 2)))
    return coeffs, rmse, net


def _print_table(weights: np.ndarray, net_sums: np.ndarray, coeffs: list[float]) -> None:
    predicted = np.polyval(coeffs, net_sums)
    print(f"\n  {'#':<4} {'Real (kg)':>10} {'Net Sum':>10} {'Predito':>10} {'Erro':>8}")
    print("  " + "-" * 50)
    for i, (kg_r, ns, pred) in enumerate(zip(weights, net_sums, predicted)):
        print(f"  {i:<4} {kg_r:>10.3f} {ns:>10.1f} {pred:>10.3f} {kg_r - pred:>+8.3f}")


# ---------------------------------------------------------------------------
# CLI principal
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

    existing = _load_existing(output_path)
    blocos_prontos = list(existing.get("blocks", {}).keys())

    print(f"\n{'='*60}")
    print(f"  CALIBRAÇÃO — Bloco {block_id}")
    print(f"{'='*60}")
    if blocos_prontos:
        print(f"  Blocos já calibrados: {', '.join(blocos_prontos)}")
    print("""
PREPARAÇÃO FÍSICA:
  - Placa rígida e plana posicionada sobre o bloco alvo.
  - Os demais blocos podem estar conectados e ativos — o script
    lê APENAS a região do bloco escolhido.
  - Nenhum peso sobre a placa ainda.
""")
    input("  Pressione Enter quando pronto...")

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

    print("Comandos disponíveis no prompt de peso:")
    print("  <número>  — registra um ponto de calibração")
    print("  r         — remove o último ponto registrado")
    print("  l         — lista os pontos atuais")
    print("  Enter     — finaliza (mínimo 3 pontos)\n")

    while True:
        entrada = input("Peso (kg) / r / l / Enter p/ finalizar: ").strip().lower()

        # --- Finalizar ---
        if entrada == "":
            if len(weight_kg_list) < 3:
                print("  [AVISO] Mínimo 3 pontos necessários.\n")
                continue
            break

        # --- Listar ---
        if entrada == "l":
            if not weight_kg_list:
                print("  (nenhum ponto registrado ainda)\n")
            else:
                print(f"  {'#':<4} {'Peso (kg)':>10} {'Sum bruto':>12}")
                for i, (kg, s) in enumerate(zip(weight_kg_list, raw_sum_list)):
                    print(f"  {i:<4} {kg:>10.3f} {s:>12.1f}")
                print()
            continue

        # --- Remover último ---
        if entrada == "r":
            if not weight_kg_list:
                print("  Nenhum ponto para remover.\n")
                continue
            kg_rem = weight_kg_list.pop()
            raw_rem = raw_sum_list.pop()
            print(f"  Ponto removido: {kg_rem:.3f} kg (sum={raw_rem:.1f})\n")
            continue

        # --- Novo ponto ---
        try:
            kg = float(entrada.replace(",", "."))
        except ValueError:
            print("  Valor inválido. Digite um número, 'r', 'l' ou Enter.\n")
            continue

        if kg == 0:
            input("\n  [AÇÃO] Retire todos os pesos. Pressione Enter para medir TARA...")
        else:
            input(f"\n  [AÇÃO] Coloque {kg:.3f} kg. Aguarde estabilizar e pressione Enter...")

        print(f"  Coletando {SAMPLE_SECONDS}s...", end="", flush=True)
        media = _read_block_sum(device, block_id, SAMPLE_SECONDS)
        print(f"  ✓  sum_bruto = {media:.1f}")

        weight_kg_list.append(kg)
        raw_sum_list.append(media)
        print(f"  Ponto [{len(weight_kg_list) - 1}] salvo: {kg:.3f} kg → sum={media:.1f}")
        print("  (use 'r' para desfazer este ponto se necessário)\n")

    device.close()

    # Ajuste de curva
    weights = np.array(weight_kg_list)
    sums_raw = np.array(raw_sum_list)

    zero_idx = np.where(weights == 0.0)[0]
    tare_sum = float(sums_raw[zero_idx[0]]) if len(zero_idx) > 0 else 0.0

    coeffs, rmse, net_sums = _fit_curve(weights, sums_raw, tare_sum)

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  Tara (sum 0 kg):   {tare_sum:.1f}")
    print(f"  Coeficientes:      {[round(c, 8) for c in coeffs]}")
    print(f"  RMSE:              {rmse:.4f} kg")
    if rmse > 1.5:
        print("  [AVISO] RMSE > 1.5 kg — considere mais pontos intermediários.")

    _print_table(weights, net_sums, coeffs)

    block_data = {
        "tare_block_sum": tare_sum,
        "polynomial_degree": min(2, len(weights) - 1),
        "coefficients": coeffs,
        "rmse_kg": round(rmse, 4),
        "calibration_points": [
            {"kg": float(k), "raw_sum": float(s)}
            for k, s in zip(weights, sums_raw)
        ],
    }

    _save_block(output_path, block_id, block_data)

    blocos_apos = list(_load_existing(output_path).get("blocks", {}).keys())
    print(f"\n[OK] Bloco {block_id} salvo em: {output_path.resolve()}")
    print(f"     Blocos no arquivo: {', '.join(blocos_apos)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
