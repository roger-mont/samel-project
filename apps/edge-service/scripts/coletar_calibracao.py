"""Assistente Interativo de Coleta de Sessões de Calibração para a Maca Samel.

Lê a manta sensora USB HID, amostra durante repouso da carga, registra a TARA
(peso 0) e anexa medições a data/sessions/calib_wide_YYYYMMDD_v3.csv.

Cada frame recebe o próprio timestamp e a flag de estabilidade calculada pela
mesma máquina de estados do edge-service. O RAW completo de cada sessão vai para
data/sessions/raw/<session_id>.u8 (uint8, 2.048 bytes por frame, mesma ordem do CSV).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import uuid

import numpy as np

# Adiciona o diretório do edge-service ao path
edge_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(edge_src))

from core.settings import CalibrationParams
from math_engine.reading_state import ReadingState, ReadingStateMachine
from storage.calibration_store import BLOCK_REGIONS

try:
    import hid
except ImportError:
    print("[ERRO] hidapi não instalado. Execute: pip install hidapi")
    sys.exit(1)

HID_VID = 0x1ACC
HID_PID = 0x1A4D
HID_PACKET_SIZE = 64
HID_ROWS = 32
HID_COLS = 64
SAMPLE_INTERVAL = 0.025  # ~40 Hz
SOFTWARE_VERSION = "3.0.0"


def parse_hid_packet(data: bytes, matrix: np.ndarray) -> None:
    """Decodifica pacote binário HID WangYing na matriz 32x64."""
    block_id = data[0]
    if block_id == 0 or block_id > 8:
        return
    eff_id = ((block_id + 3) % 8) + 1
    for i in range(1, len(data) - 1, 3):
        if i + 2 >= len(data):
            break
        x_local = data[i]
        y_local = data[i + 1]
        pressure = data[i + 2]
        if x_local == 0 or y_local == 0:
            break
        if eff_id < 5:
            x = 16 * ((9 - eff_id) // 5) + (16 - x_local)
            y = 16 * (eff_id - 1) + (16 - y_local)
        else:
            x = 16 * ((9 - eff_id) // 5) + (x_local - 1)
            y = 16 * (8 - eff_id) + (y_local - 1)
        if 0 <= x < HID_ROWS and 0 <= y < HID_COLS:
            matrix[x, y] = float(pressure)


def conectar_hid() -> hid.device:
    """Conecta ao dispositivo USB HID real da manta."""
    dev = hid.device()
    try:
        dev.open(HID_VID, HID_PID)
        dev.set_nonblocking(1)
        return dev
    except Exception as err:
        print(f"[ERRO] Falha ao abrir dispositivo HID (VID=0x{HID_VID:04X}, PID=0x{HID_PID:04X}): {err}")
        print("Certifique-se de que a manta está conectada via USB.")
        sys.exit(1)


def coletar_amostra(
    dev: hid.device,
    duracao_s: float = 15.0,
    is_tare: bool = False,
    baseline: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, float]:
    """Amostra leituras contínuas por 'duracao_s' segundos, sem descartar frames.

    O transitório fica registrado e marcado como instável pela máquina de estados,
    em vez de ser cortado por um tempo fixo.
    """
    matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    base = baseline if baseline is not None else np.zeros((HID_ROWS, HID_COLS))
    machine = ReadingStateMachine(CalibrationParams())
    deadline = time.monotonic() + duracao_s
    t_start = time.monotonic()

    # Drena buffer inicial acumulado
    while dev.read(HID_PACKET_SIZE):
        pass

    frames_registrados: list[dict] = []

    label = "TARA (maca vazia)" if is_tare else "CARGA"
    print(f"\n[+] Coletando {label} por {duracao_s:.0f}s...")

    while time.monotonic() < deadline:
        t0 = time.monotonic()
        while True:
            raw = dev.read(HID_PACKET_SIZE)
            if not raw:
                break
            parse_hid_packet(bytes(raw), matrix)

        elapsed = time.monotonic() - t_start
        corrected = np.maximum(0.0, matrix - base)
        state = machine.update(corrected, elapsed)
        metrics = machine.metrics
        soma_total = float(matrix.sum())

        frames_registrados.append({
            "timestamp_iso": datetime.now(timezone.utc).astimezone().isoformat(),
            "tempo_s": round(elapsed, 3),
            "b_vals": [float(matrix[BLOCK_REGIONS[bid]].sum()) for bid in range(1, 9)],
            "soma_total": soma_total,
            "soma_corrigida": float(corrected.sum()),
            "estado": state.value,
            "estavel": 1 if state in (ReadingState.ESTAVEL, ReadingState.MEDICAO_VALIDA) else 0,
            "stabilization_time_s": metrics.stabilization_time_s,
            "raw": np.clip(np.rint(matrix), 0, 255).astype(np.uint8),
        })

        rem = max(0.0, duracao_s - elapsed)
        print(
            f"\r  [Tempo restante: {rem:4.1f}s | Frames: {len(frames_registrados):4d} | "
            f"Soma: {soma_total:7.0f} | {state.value:<15}] ",
            end="",
            flush=True,
        )

        dt = SAMPLE_INTERVAL - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)

    estaveis = sum(f["estavel"] for f in frames_registrados)
    print(f"\n[✓] Amostragem concluída! ({len(frames_registrados)} frames, {estaveis} estáveis)")

    if not frames_registrados:
        return [], np.zeros(8), 0.0

    b_medias = np.mean([f["b_vals"] for f in frames_registrados], axis=0)
    soma_media = float(np.mean([f["soma_total"] for f in frames_registrados]))

    return frames_registrados, b_medias, soma_media


def executar_tara(dev: hid.device, sessions_dir: Path, duracao_s: float = 5.0) -> np.ndarray | None:
    """Registra a TARA (peso 0 kg) e devolve o baseline por célula (média dos frames)."""
    input("\n-> Certifique-se de que a maca está VAZIA (sem nenhum peso) e pressione ENTER...")
    frames_registrados, _, soma_media = coletar_amostra(dev, duracao_s=duracao_s, is_tare=True)

    if not frames_registrados:
        print("[!] Nenhum frame capturado durante a tara. Tente novamente.")
        return None

    stack = np.stack([f["raw"].astype(np.float64) for f in frames_registrados])
    baseline = stack.mean(axis=0)
    desvio_medio = float(stack.std(axis=0).mean())

    salvar_sessao(sessions_dir, 0.0, "tare", 1, frames_registrados, duracao_s, baseline_raw=float(baseline.sum()))

    print("[✓] TARA registrada.")
    print(f"    Soma média de repouso: {soma_media:.1f} | desvio médio por célula: {desvio_medio:.2f} RAW")
    return baseline


def salvar_sessao(
    sessions_dir: Path,
    peso_real_kg: float,
    posicao: str,
    repeticao: int,
    frames_registrados: list[dict],
    duracao_s: float,
    direction: str = "UP",
    cycle_id: int = 1,
    baseline_raw: float = 0.0,
    block_id: int = 0,
    cell_id: str = "all",
) -> tuple[Path, int]:
    """Salva frames da sessão em calib_wide_YYYYMMDD_v3.csv e o RAW em raw/<session_id>.u8."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    hoje_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = sessions_dir / f"calib_wide_{hoje_str}_v3.csv"
    raw_dir = sessions_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "timestamp_iso",
        "session_id",
        "software_version",
        "peso_real_kg",
        "posicao",
        "repeticao",
        "direction",
        "cycle_id",
        "stable",
        "stabilization_time_s",
        "baseline_raw",
        "raw_mean",
        "raw_std",
        "block_id",
        "cell_id",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "soma_total",
        "soma_corrigida",
        "tempo_s",
        "estado",
        "estavel",
        "raw_file",
        "raw_frame_idx",
        "duracao_s",
    ]

    novo = not csv_path.exists() or csv_path.stat().st_size == 0
    session_uuid = str(uuid.uuid4())
    raw_path = raw_dir / f"{session_uuid}.u8"
    linhas_gravadas = 0

    # Média e desvio só da parte estável; sem trecho estável, fica vazio
    somas_estaveis = [f["soma_total"] for f in frames_registrados if f["estavel"]]
    raw_mean = round(float(np.mean(somas_estaveis)), 4) if somas_estaveis else ""
    raw_std = round(float(np.std(somas_estaveis)), 4) if len(somas_estaveis) > 1 else ""
    sessao_estavel = bool(somas_estaveis)
    ts = next((f["stabilization_time_s"] for f in frames_registrados if f["stabilization_time_s"] is not None), None)

    with raw_path.open("wb") as raw_f:
        for frame in frames_registrados:
            raw_f.write(frame["raw"].tobytes(order="C"))

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if novo:
            writer.writerow(headers)

        for idx, frame in enumerate(frames_registrados):
            row = [
                frame["timestamp_iso"],
                session_uuid,
                SOFTWARE_VERSION,
                round(peso_real_kg, 4),
                posicao,
                repeticao,
                direction.upper(),
                cycle_id,
                1 if sessao_estavel else 0,
                round(ts, 3) if ts is not None else "",
                round(baseline_raw, 4),
                raw_mean,
                raw_std,
                block_id,
                cell_id,
                *(round(float(b), 4) for b in frame["b_vals"]),
                round(frame["soma_total"], 4),
                round(frame["soma_corrigida"], 4),
                frame["tempo_s"],
                frame["estado"],
                frame["estavel"],
                f"raw/{raw_path.name}",
                idx,
                round(duracao_s, 2),
            ]
            writer.writerow(row)
            linhas_gravadas += 1

    return csv_path, linhas_gravadas


def carregar_contadores_repeticao(sessions_dir: Path) -> dict[tuple[float, str], int]:
    """Descobre a maior repetição já gravada para cada par (peso_real_kg, posicao) hoje."""
    counters: dict[tuple[float, str], int] = {}
    hoje_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    for csv_path in sessions_dir.glob(f"calib_wide_{hoje_str}*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        p = round(float(row.get("peso_real_kg", 0.0)), 2)
                        pos = str(row.get("posicao", "center")).strip()
                        rep = int(row.get("repeticao", 1))
                        key = (p, pos)
                        if rep > counters.get(key, 0):
                            counters[key] = rep
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
    return counters


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    sessions_dir = base_dir / "data" / "sessions"

    print("=" * 70)
    print(" SAMEL MACA INTELIGENTE — ASSISTENTE DE CALIBRAÇÃO DE PESO")
    print("=" * 70)
    print("Este utilitário registra ensaios de peso real e a TARA.")
    print("Para registrar a TARA, digite 0 quando solicitado o peso.\n")

    dev = conectar_hid()
    print("[✓] Manta USB HID conectada com sucesso.")

    rep_counters = carregar_contadores_repeticao(sessions_dir)
    baseline: np.ndarray | None = None

    posicoes_map = {
        "1": "center",
        "2": "head",
        "3": "foot",
        "4": "right",
        "5": "left",
        "6": "full_body",
    }

    try:
        while True:
            print("\n" + "-" * 70)
            peso_str = input("-> Digite o peso real em kg [0 para TARA / 's' para sair]: ").strip()
            if peso_str.lower() in ("s", "sair", "exit", "q"):
                break

            try:
                peso_kg = float(peso_str.replace(",", "."))
                if peso_kg < 0:
                    print("[!] O peso não pode ser negativo.")
                    continue
            except ValueError:
                print("[!] Valor de peso inválido. Digite um número (ex: 0, 50, 72.5).")
                continue

            # Se digitou 0 -> TARA
            if peso_kg == 0.0:
                dur_str = input("Duração da tara em segundos [padrão 5s]: ").strip() or "5"
                dur_tara = float(dur_str) if dur_str.replace(".", "").isdigit() else 5.0
                nova = executar_tara(dev, sessions_dir, duracao_s=dur_tara)
                if nova is not None:
                    baseline = nova
                continue

            if baseline is None:
                print("[!] Ainda não há tara nesta execução: a estabilidade será avaliada sobre o RAW sem baseline.")

            print("\nEscolha a posição da carga na maca:")
            print("  1. Centro (center — B2, B3, B6, B7)")
            print("  2. Cabeceira (head — B1, B8)")
            print("  3. Peseira (foot — B4, B5)")
            print("  4. Lado Direito (right — B1, B2, B3, B4)")
            print("  5. Lado Esquerdo (left — B8, B7, B6, B5)")
            print("  6. Corpo Inteiro / Deitado (full_body)")
            pos_opcao = input("Opção [1-6, padrão 1]: ").strip() or "1"
            posicao = posicoes_map.get(pos_opcao, "center")

            # Cálculo automático do número da repetição
            rep_key = (round(peso_kg, 2), posicao)
            repeticao = rep_counters.get(rep_key, 0) + 1
            print(f"  -> Repetição: #{repeticao} (detectada automaticamente)")

            # Sentido do ciclo e ID para histerese
            dir_opcao = input("Direção da carga [1=UP (subida), 2=DOWN (descida), padrão 1]: ").strip() or "1"
            direction = "DOWN" if dir_opcao == "2" else "UP"

            ciclo_str = input("Número do ciclo independente [padrão 1]: ").strip() or "1"
            cycle_id = int(ciclo_str) if ciclo_str.isdigit() else 1

            dur_str = input("Tempo de amostragem em segundos [padrão 60s]: ").strip() or "60"
            duracao = float(dur_str) if dur_str.replace(".", "").isdigit() else 60.0

            input(f"\n-> Pressione ENTER e posicione {peso_kg} kg ({direction}) na maca...")

            frames_registrados, b_medias, soma_media = coletar_amostra(dev, duracao, is_tare=False, baseline=baseline)

            if not frames_registrados:
                print("[!] Nenhum dado detectado. Verifique se o peso foi posicionado.")
                continue

            print(f"\nResultados obtidos para {peso_kg} kg ({posicao}) [{direction}] - Ciclo #{cycle_id} Rep #{repeticao}:")
            for i, b_val in enumerate(b_medias):
                print(f"  B{i+1}: {b_val:7.1f}", end="  " if (i + 1) % 4 != 0 else "\n")
            print(f"  Soma Média: {soma_media:.1f}")
            if not any(f["estavel"] for f in frames_registrados):
                print("[!] O sinal não estabilizou. Considere repetir com um patamar mais longo.")

            conf = input(f"\nDeseja salvar este ensaio ({len(frames_registrados)} frames)? (S/n): ").strip().lower()
            if conf in ("", "s", "sim", "y", "yes"):
                arquivo_salvo, total_f = salvar_sessao(
                    sessions_dir,
                    peso_kg,
                    posicao,
                    repeticao,
                    frames_registrados,
                    duracao,
                    direction=direction,
                    cycle_id=cycle_id,
                    baseline_raw=float(baseline.sum()) if baseline is not None else 0.0,
                )
                rep_counters[rep_key] = repeticao
                print(f"[✓] {total_f} frames gravados com sucesso (Ciclo #{cycle_id}, {direction}) em: {arquivo_salvo.name}")

    finally:
        dev.close()
        print("\nDispositivo USB liberado.")


if __name__ == "__main__":
    main()
