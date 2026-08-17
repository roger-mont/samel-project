"""Script de calibração interativa — blocos da manta FSR.

Como usar:
    cd "Projeto Walter"
    .\\venv\\Scripts\\activate
    python Python/calibrate_block.py

O script permite escolher e alterar interativamente o bloco (1 a 8),
a posição da carga (center, upper, etc.) e o número da repetição,
além de permitir múltiplos ensaios sequenciais sem reiniciar o script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import HID_VID, HID_PID, HID_PACKET_SIZE, HID_ROWS, HID_COLS
from services.session_logger import CalibrationSessionLogger
from services.calibration_store import load_calibration
from services.math_pipeline import compute_model_a
from scripts.fit_model_c import fit_model_c

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

POSITION_OPTIONS: dict[str, str] = {
    "1": "center",
    "2": "upper",
    "3": "lower",
    "4": "left",
    "5": "right",
    "6": "full_body",
}

SAMPLE_SECONDS = 5
RENDER_INTERVAL = 0.025
GRAVITY_M_S2: float = 9.81


# ---------------------------------------------------------------------------
# HID — parse de pacotes e drenagem do buffer
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


def _read_block_sum(device: hid.device, block_id: int, duration_s: float) -> tuple[float, float]:
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

    if not samples:
        return 0.0, 0.0
    return float(np.mean(samples)), float(np.std(samples))


# ---------------------------------------------------------------------------
# Persistência JSON
# ---------------------------------------------------------------------------

def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {"version": 3, "blocks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [AVISO] {path.name} corrompido — será recriado.")
        return {"version": 3, "blocks": {}}
    return data


def _save_block(path: Path, block_id: int, block_data: dict) -> None:
    existing = _load_existing(path)
    existing["version"] = 3
    if "blocks" not in existing:
        existing["blocks"] = {}
    existing["blocks"][str(block_id)] = block_data
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ajuste de curva e visualização
# ---------------------------------------------------------------------------

def _fit_curve(
    weights_kg: np.ndarray,
    sums_raw: np.ndarray,
    tare_sum: float,
) -> tuple[list[float], float, float, np.ndarray]:
    net = sums_raw - tare_sum
    forces_n = weights_kg * GRAVITY_M_S2
    degree = min(2, len(weights_kg) - 1)
    coeffs = np.polyfit(net, forces_n, deg=degree).tolist()
    predicted_n = np.polyval(coeffs, net)
    rmse_n = float(np.sqrt(np.mean((predicted_n - forces_n) ** 2)))
    rmse_kg = rmse_n / GRAVITY_M_S2
    return coeffs, rmse_n, rmse_kg, net


def _print_table(weights_kg: np.ndarray, net_sums: np.ndarray, coeffs_n: list[float]) -> None:
    forces_n = weights_kg * GRAVITY_M_S2
    predicted_n = np.polyval(coeffs_n, net_sums)
    predicted_kg = predicted_n / GRAVITY_M_S2
    print(f"\n  {'#':<4} {'Real(kg)':>9} {'Real(N)':>9} {'NetSum':>10} {'Pred(N)':>9} {'Pred(kg)':>9} {'Err(kg)':>8}")
    print("  " + "-" * 68)
    for i, (kg, fn, ns, pn, pk) in enumerate(zip(weights_kg, forces_n, net_sums, predicted_n, predicted_kg)):
        print(f"  {i:<4} {kg:>9.3f} {fn:>9.2f} {ns:>10.1f} {pn:>9.2f} {pk:>9.3f} {kg - pk:>+8.3f}")


# ---------------------------------------------------------------------------
# Prompts Interativos de Configuração
# ---------------------------------------------------------------------------

def _prompt_block_id(default: int = 1) -> int:
    while True:
        resp = input(f"  Bloco a calibrar (1 a 8) [Enter = {default}]: ").strip()
        if not resp:
            return default
        if resp.isdigit() and 1 <= int(resp) <= 8:
            return int(resp)
        print("  [ERRO] Digite um número de 1 a 8.")


def _prompt_position(default: str = "center") -> str:
    print("  Opções de posição:")
    print("    [1] center (centro)      [2] upper (superior)   [3] lower (inferior)")
    print("    [4] left (esquerda)      [5] right (direita)    [6] full_body (corpo todo)")
    while True:
        resp = input(f"  Posição da carga [1-6 ou nome, Enter = '{default}']: ").strip().lower()
        if not resp:
            return default
        if resp in POSITION_OPTIONS:
            return POSITION_OPTIONS[resp]
        if resp in POSITION_OPTIONS.values():
            return resp
        print("  [ERRO] Posição inválida. Escolha de 1 a 6 ou digite o nome.")


def _prompt_repetition(default: int = 1) -> int:
    while True:
        resp = input(f"  Número da repetição [Enter = {default}]: ").strip()
        if not resp:
            return default
        if resp.isdigit() and int(resp) >= 1:
            return int(resp)
        print("  [ERRO] Digite um número inteiro >= 1.")


# ---------------------------------------------------------------------------
# Coleta de Pontos e Ensaio
# ---------------------------------------------------------------------------

def _log_point_to_csv(
    session_logger: CalibrationSessionLogger,
    session_id: str,
    block_id: int,
    kg: float,
    position_tag: str,
    repetition: int,
    media: float,
    std: float,
    cv: float,
    estavel: bool,
    weight_kg_list: list[float],
    raw_sum_list: list[float],
    existing: dict,
) -> None:
    if 0.0 in weight_kg_list:
        cur_tare = raw_sum_list[weight_kg_list.index(0.0)]
    else:
        cur_tare = float(existing.get("blocks", {}).get(str(block_id), {}).get("tare_block_sum", 0.0))

    block_calib = existing.get("blocks", {}).get(str(block_id))
    est_kg = None
    if block_calib and "coefficients_raw_to_n" in block_calib:
        net_val = max(0.0, media - cur_tare)
        est_n = float(np.polyval(block_calib["coefficients_raw_to_n"], net_val))
        est_kg = round(est_n / GRAVITY_M_S2, 3)

    csv_file = session_logger.log_calibration_point(
        session_id=session_id,
        block_id=block_id,
        reference_kg=kg,
        position_tag=position_tag,
        repetition=repetition,
        raw_sum=media,
        net_sum=max(0.0, media - cur_tare),
        mean=media,
        std=std,
        cv_pct=cv,
        estimated_kg=est_kg,
        stability_state="stable" if estavel else "transient",
        time_since_load_s=float(SAMPLE_SECONDS),
        tare_block_sum=cur_tare,
    )
    if csv_file:
        print(f"  [CSV] Registrado em: {csv_file.name}")


def _collect_points_interactive(
    device: hid.device,
    block_id: int,
    position_tag: str,
    repetition: int,
    session_logger: CalibrationSessionLogger,
    session_id: str,
    existing: dict,
) -> tuple[np.ndarray, np.ndarray]:
    weight_kg_list: list[float] = []
    raw_sum_list: list[float] = []

    print("\nComandos disponíveis no prompt de peso:")
    print("  <número>  — registra um ponto de calibração (ex: 0, 5, 10)")
    print("  r         — remove o último ponto registrado")
    print("  l         — lista os pontos atuais")
    print("  Enter     — finaliza e ajusta a curva (mínimo 3 pontos)\n")

    while True:
        entrada = input("Peso (kg) / r / l / Enter p/ finalizar: ").strip().lower()

        if entrada == "":
            if len(weight_kg_list) < 3:
                print("  [AVISO] Mínimo 3 pontos necessários.\n")
                continue
            break

        if entrada == "l":
            if not weight_kg_list:
                print("  (nenhum ponto registrado ainda)\n")
            else:
                print(f"\n  {'#':<4} {'Peso (kg)':>10} {'Sum bruto':>12}")
                for i, (kg, s) in enumerate(zip(weight_kg_list, raw_sum_list)):
                    print(f"  {i:<4} {kg:>10.3f} {s:>12.1f}")
                print()
            continue

        if entrada == "r":
            if not weight_kg_list:
                print("  Nenhum ponto para remover.\n")
                continue
            kg_rem = weight_kg_list.pop()
            raw_rem = raw_sum_list.pop()
            print(f"  Ponto removido: {kg_rem:.3f} kg (sum={raw_rem:.1f})\n")
            continue

        try:
            kg = float(entrada.replace(",", "."))
        except ValueError:
            print("  Valor inválido. Digite um número, 'r', 'l' ou Enter.\n")
            continue

        if kg == 0:
            input("\n  [AÇÃO] Retire todos os pesos. Pressione Enter para medir TARA...")
        else:
            input(f"\n  [AÇÃO] Coloque {kg:.3f} kg. Aguarde COMPLETAMENTE parado e pressione Enter...")

        print(f"  Coletando {SAMPLE_SECONDS}s...", end="", flush=True)
        media, std = _read_block_sum(device, block_id, SAMPLE_SECONDS)
        cv = (std / media * 100) if media > 0 else 0.0
        estavel = cv < 8.0
        simbolo = "✓" if estavel else "⚠"
        print(f"  {simbolo}  sum={media:.1f}  std={std:.1f}  variação={cv:.1f}%")

        if not estavel:
            print("  [AVISO] Medição instável (variação > 8%) — placa ou peso pode ter oscilado.")
            resp = input("  Refazer esta medição? (s/N): ").strip().lower()
            if resp == "s":
                print()
                continue

        if weight_kg_list and kg > weight_kg_list[-1]:
            ultimo_sum = raw_sum_list[-1]
            if media < ultimo_sum:
                print(f"  [ERRO DE MONOTONIA] sum caiu ({ultimo_sum:.0f}→{media:.0f}) com peso maior. "
                      f"Medição provavelmente incorreta.")
                resp = input("  Descartar e refazer? (s/N): ").strip().lower()
                if resp == "s":
                    print()
                    continue

        weight_kg_list.append(kg)
        raw_sum_list.append(media)
        print(f"  Ponto [{len(weight_kg_list) - 1}] salvo: {kg:.3f} kg → sum={media:.1f}")

        _log_point_to_csv(
            session_logger, session_id, block_id, kg, position_tag, repetition,
            media, std, cv, estavel, weight_kg_list, raw_sum_list, existing
        )
        print("  (use 'r' para desfazer este ponto se necessário)\n")

    return np.array(weight_kg_list), np.array(raw_sum_list)


def _run_single_calibration(
    device: hid.device,
    block_id: int,
    position_tag: str,
    repetition: int,
    session_logger: CalibrationSessionLogger,
    output_path: Path,
) -> None:
    session_id = session_logger.generate_session_id()
    existing = _load_existing(output_path)

    print(f"\n{'='*60}")
    print(f"  INICIANDO ENSAIO — Bloco {block_id} | Posição: '{position_tag}' | Repetição: #{repetition}")
    print(f"  Sessão ID: {session_id[:8]}...")
    print(f"{'='*60}")
    print("""
PREPARAÇÃO FÍSICA:
  - Placa rígida e plana posicionada sobre o bloco alvo.
  - Nenhum peso sobre a placa ainda.
""")
    input("  Pressione Enter quando a placa estiver posicionada...")

    weights, sums_raw = _collect_points_interactive(
        device, block_id, position_tag, repetition,
        session_logger, session_id, existing
    )

    zero_idx = np.where(weights == 0.0)[0]
    tare_sum = float(sums_raw[zero_idx[0]]) if len(zero_idx) > 0 else 0.0

    coeffs_n, rmse_n, rmse_kg, net_sums = _fit_curve(weights, sums_raw, tare_sum)

    print("\n" + "=" * 60)
    print("  RESULTADO DO AJUSTE (unidade física interna: Newton)")
    print("=" * 60)
    print(f"  Tara do bloco (sum 0 kg): {tare_sum:.1f}")
    print(f"  Coeficientes (→N):        {[round(c, 8) for c in coeffs_n]}")
    print(f"  RMSE de calibração:       {rmse_n:.4f} N  ({rmse_kg:.4f} kg)")
    if rmse_kg > 1.5:
        print("  [AVISO] RMSE > 1.5 kg — considere adicionar mais pontos.")

    _print_table(weights, net_sums, coeffs_n)

    block_data = {
        "unit": "newton",
        "tare_block_sum": tare_sum,
        "polynomial_degree": min(2, len(weights) - 1),
        "coefficients_raw_to_n": coeffs_n,
        "rmse_n": round(rmse_n, 4),
        "rmse_kg": round(rmse_kg, 4),
        "calibration_points": [
            {"kg": float(k), "n": round(float(k) * GRAVITY_M_S2, 4), "raw_sum": float(s)}
            for k, s in zip(weights, sums_raw)
        ],
    }

    _save_block(output_path, block_id, block_data)
    blocos_apos = list(_load_existing(output_path).get("blocks", {}).keys())
    print(f"\n[OK] Bloco {block_id} salvo com sucesso em: {output_path.name}")
    print(f"     Blocos no arquivo: {', '.join(blocos_apos)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Calibração Global da Maca Inteira (Modelo C / Adultos)
# ---------------------------------------------------------------------------

def _read_full_mat_mass(
    device: hid.device,
    calib_path: Path,
    duration_s: float,
) -> tuple[float, float, float]:
    """Lê a matriz inteira 32x64 e calcula a massa física (kg) e variação."""
    calib = load_calibration(calib_path)
    matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    mass_samples: list[float] = []
    raw_sum_samples: list[float] = []
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

        mass_kg = compute_model_a(matrix, calib)
        mass_samples.append(mass_kg)
        raw_sum_samples.append(float(np.sum(matrix)))

        rem = RENDER_INTERVAL - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

    if not mass_samples:
        return 0.0, 0.0, 0.0

    mean_mass = float(np.mean(mass_samples))
    std_mass = float(np.std(mass_samples))
    mean_raw_sum = float(np.mean(raw_sum_samples))
    return mean_mass, std_mass, mean_raw_sum


def _run_global_calibration(
    device: hid.device,
    session_logger: CalibrationSessionLogger,
    output_path: Path,
    sessions_dir: Path,
) -> None:
    session_id = session_logger.generate_session_id()
    print(f"\n{'='*60}")
    print("  CALIBRAÇÃO GLOBAL DA MACA INTEIRA — MODELO C (ADULTOS / CORPO INTEIRO)")
    print(f"  Sessão ID: {session_id[:8]}...")
    print(f"{'='*60}")
    print("""
PREPARAÇÃO FÍSICA:
  - Maca com colchão, lençol e travesseiro posicionados normalmente.
  - Certifique-se de que a maca está VAZIA para medir a TARA inicial.
""")
    input("  Pressione Enter quando a maca estiver VAZIA para medir TARA...")

    print(f"  Medindo tara da maca ({SAMPLE_SECONDS}s)...", end="", flush=True)
    tare_mass, tare_std, tare_raw = _read_full_mat_mass(device, output_path, SAMPLE_SECONDS)
    print(f"  ✓  Massa residual inicial = {tare_mass:.2f} kg (sum bruto={tare_raw:.0f})\n")

    weight_real_list: list[float] = []
    weight_measured_list: list[float] = []

    print("Comandos disponíveis no prompt de peso:")
    print("  <número>  — registra um voluntário com peso conhecido na balança (ex: 72.5)")
    print("  r         — remove o último voluntário registrado")
    print("  l         — lista as medições atuais")
    print("  Enter     — finaliza a coleta e ajusta o Modelo C global\n")

    while True:
        entrada = input("Peso real do voluntário (kg) / r / l / Enter p/ finalizar: ").strip().lower()

        if entrada == "":
            if len(weight_real_list) < 3:
                print("  [AVISO] Mínimo 3 pessoas/pesos recomendados para ajustar a curva global.\n")
                resp = input("  Deseja finalizar mesmo assim? (s/N): ").strip().lower()
                if resp != "s":
                    continue
            break

        if entrada == "l":
            if not weight_real_list:
                print("  (nenhum voluntário registrado ainda)\n")
            else:
                print(f"\n  {'#':<4} {'Real (kg)':>12} {'Medido Físico (kg)':>20} {'Diferença (kg)':>16}")
                print("  " + "-" * 56)
                for i, (r, m) in enumerate(zip(weight_real_list, weight_measured_list)):
                    print(f"  {i:<4} {r:>12.2f} {m:>20.2f} {m - r:>+16.2f}")
                print()
            continue

        if entrada == "r":
            if not weight_real_list:
                print("  Nenhum ponto para remover.\n")
                continue
            r_rem = weight_real_list.pop()
            m_rem = weight_measured_list.pop()
            print(f"  Ponto removido: {r_rem:.2f} kg real (medido={m_rem:.2f} kg)\n")
            continue

        try:
            real_kg = float(entrada.replace(",", "."))
            if real_kg <= 0.0:
                print("  [ERRO] O peso deve ser maior que 0 kg.\n")
                continue
        except ValueError:
            print("  Valor inválido. Digite um número (ex: 70.5), 'r', 'l' ou Enter.\n")
            continue

        input(f"\n  [AÇÃO] Voluntário de {real_kg:.2f} kg DEITADO na maca (parado). Pressione Enter para ler...")
        print(f"  Coletando dados da maca inteira ({SAMPLE_SECONDS}s)...", end="", flush=True)
        mean_mass, std_mass, mean_raw = _read_full_mat_mass(device, output_path, SAMPLE_SECONDS)
        net_measured_kg = max(0.0, mean_mass - tare_mass)
        cv = (std_mass / mean_mass * 100) if mean_mass > 0 else 0.0
        estavel = cv < 8.0
        simbolo = "✓" if estavel else "⚠"

        print(f"  {simbolo}  medido={net_measured_kg:.2f} kg  std={std_mass:.2f}  variação={cv:.1f}%")

        if not estavel:
            print("  [AVISO] Medição instável (variação corporal > 8%). Voluntário pode ter se movido.")
            resp = input("  Refazer esta medição? (s/N): ").strip().lower()
            if resp == "s":
                print()
                continue

        weight_real_list.append(real_kg)
        weight_measured_list.append(net_measured_kg)
        print(f"  Voluntário [{len(weight_real_list) - 1}] salvo: Real {real_kg:.2f} kg → Medido {net_measured_kg:.2f} kg")

        # Registra no CSV global
        csv_file = session_logger.log_calibration_point(
            session_id=session_id,
            block_id=0,  # 0 indica Maca Global
            reference_kg=real_kg,
            position_tag="full_body",
            repetition=1,
            raw_sum=mean_raw,
            net_sum=max(0.0, mean_raw - tare_raw),
            mean=mean_raw,
            std=std_mass,
            cv_pct=cv,
            estimated_kg=round(net_measured_kg, 3),
            stability_state="stable" if estavel else "transient",
            time_since_load_s=float(SAMPLE_SECONDS),
            tare_block_sum=tare_raw,
        )
        if csv_file:
            print(f"  [CSV] Registrado em: {csv_file.name}")
        print("  (use 'r' para desfazer se necessário)\n")

    if len(weight_real_list) >= 2:
        print("\n" + "="*60)
        print("  AJUSTANDO MODELO C COM OS NOVOS DADOS DA MACA INTEIRA...")
        print("="*60)
        fit_model_c(sessions_dir=sessions_dir, output_json=output_path)
    else:
        print("\n[INFO] Menos de 2 voluntários coletados. Modelo C não foi reajustado.")


# ---------------------------------------------------------------------------
# CLI e Loop Principal
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibração interativa de bloco FSR 16x16")
    p.add_argument("--block", type=int, choices=range(1, 9), default=1,
                   help="Bloco inicial sugerido (1-8, padrão: 1)")
    p.add_argument("--position", type=str, default="center",
                   help="Posição inicial sugerida (padrão: center)")
    p.add_argument("--repetition", type=int, default=1,
                   help="Repetição inicial sugerida (padrão: 1)")
    p.add_argument("--output", type=str, default="calibration.json",
                   help="Arquivo de saída (padrão: calibration.json)")
    p.add_argument("--sessions-dir", type=str, default="sessions",
                   help="Diretório dos arquivos CSV de sessão (padrão: sessions)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    session_logger = CalibrationSessionLogger(output_dir=args.sessions_dir)

    print(f"\n{'='*60}")
    print("  SISTEMA INTERATIVO DE CALIBRAÇÃO — MANTA FSR 32×64")
    print(f"{'='*60}")

    try:
        device = hid.device()
        device.open(HID_VID, HID_PID)
        device.set_nonblocking(True)
        print(f"[OK] HID conectado — VID=0x{HID_VID:04X} PID=0x{HID_PID:04X}\n")
    except OSError as err:
        print(f"\n[ERRO] Falha ao abrir HID: {err}")
        sys.exit(1)

    cur_block = args.block
    cur_pos = args.position
    cur_rep = args.repetition

    try:
        while True:
            existing = _load_existing(output_path)
            blocos_prontos = list(existing.get("blocks", {}).keys())
            if blocos_prontos:
                print(f"Blocos já calibrados no JSON: {', '.join(blocos_prontos)}\n")

            print("Escolha o modo de calibração:")
            print("  [1] Calibrar Bloco Individual (1 a 8) — placa rígida + pesos menores")
            print("  [2] Calibrar Maca Inteira (Modelo C)  — voluntários / adultos / corpo todo")
            modo = input("Modo [1/2, Enter = 1]: ").strip()

            if modo == "2":
                _run_global_calibration(
                    device=device,
                    session_logger=session_logger,
                    output_path=output_path,
                    sessions_dir=Path(args.sessions_dir),
                )
            else:
                print("\nConfiguração da sessão de bloco:")
                cur_block = _prompt_block_id(cur_block)
                cur_pos = _prompt_position(cur_pos)
                cur_rep = _prompt_repetition(cur_rep)

                _run_single_calibration(
                    device=device,
                    block_id=cur_block,
                    position_tag=cur_pos,
                    repetition=cur_rep,
                    session_logger=session_logger,
                    output_path=output_path,
                )

            print("\nO que deseja fazer a seguir?")
            print(f"  [1] Nova repetição do bloco {cur_block}")
            print("  [2] Configurar novo ensaio / trocar modo")
            print("  [3] Finalizar e sair")
            escolha = input("Escolha [1/2/3, Enter = 3]: ").strip()

            if escolha == "1":
                cur_rep += 1
                print(f"\n>>> Preparando Repetição #{cur_rep} para Bloco {cur_block}...")
                continue
            if escolha == "2":
                cur_rep = 1
                print("\n>>> Configurando novo ensaio...")
                continue

            print("\nEncerrando calibração. Todos os dados foram salvos com sucesso.")
            break
    finally:
        device.close()


if __name__ == "__main__":
    main()
