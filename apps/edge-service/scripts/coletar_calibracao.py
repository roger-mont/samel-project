"""Assistente Interativo de Coleta de Sessões de Calibração para a Maca Samel.

Lê a manta sensora USB HID, amostra durante repouso da carga,
permite calibração de TARA (peso 0) e anexa medições a data/sessions/calib_wide_YYYYMMDD.csv.
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

from storage.calibration_store import BLOCK_REGIONS, load_calibration, GRAVITY_M_S2
from storage.tare_store import save_tare, load_tare

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
) -> tuple[list[dict], np.ndarray, float]:
    """Amostra leituras contínuas por 'duracao_s' segundos.
    
    Descarta os primeiros 2.0s (acomodação) e retorna lista de frames capturados.
    """
    matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    deadline = time.monotonic() + duracao_s
    t_start = time.monotonic()

    # Drena buffer inicial acumulado
    while dev.read(HID_PACKET_SIZE):
        pass

    frames_registrados: list[dict] = []
    tempo_descarte_inicial_s = 1.5 if not is_tare else 0.5

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
        soma_total = float(matrix.sum())

        b_instantaneo = [float(matrix[BLOCK_REGIONS[bid]].sum()) for bid in range(1, 9)]

        if elapsed >= tempo_descarte_inicial_s and (is_tare or soma_total > 50.0):
            frames_registrados.append({
                "tempo_s": round(elapsed, 3),
                "b_vals": b_instantaneo,
                "soma_total": soma_total,
            })

        rem = max(0.0, duracao_s - elapsed)
        print(f"\r  [Tempo restante: {rem:4.1f}s | Frames válidos: {len(frames_registrados):3d} | Soma: {soma_total:6.0f}] ", end="", flush=True)

        dt = SAMPLE_INTERVAL - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)

    print(f"\n[✓] Amostragem concluída! ({len(frames_registrados)} frames)")

    if not frames_registrados:
        return [], np.zeros(8), 0.0

    b_medias = np.mean([f["b_vals"] for f in frames_registrados], axis=0)
    soma_media = float(np.mean([f["soma_total"] for f in frames_registrados]))

    return frames_registrados, b_medias, soma_media


def executar_tara(dev: hid.device, sessions_dir: Path, duracao_s: float = 5.0) -> None:
    """Executa a calibração de TARA (peso 0 kg) e atualiza o tare.json."""
    input("\n-> Certifique-se de que a maca está VAZIA (sem nenhum peso) e pressione ENTER...")
    frames_registrados, b_medias, soma_media = coletar_amostra(dev, duracao_s=duracao_s, is_tare=True)

    if not frames_registrados:
        print("[!] Nenhum frame capturado durante a tara. Tente novamente.")
        return

    # Estima massa base da manta vazia para tara
    storage_dir = Path(__file__).resolve().parent.parent / "src" / "storage"
    calib_path = storage_dir / "calibration.json"
    tare_path = storage_dir / "tare.json"

    calib = load_calibration(calib_path)
    mat_vazia = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
    # preenche com médias por bloco
    for bid, (r_sl, c_sl) in BLOCK_REGIONS.items():
        mat_vazia[r_sl, c_sl] = b_medias[bid - 1] / (16 * 16)

    massa_vazia_kg = calib.predict_mass(mat_vazia) if calib.is_valid else (soma_media / (GRAVITY_M_S2 * 100.0))
    save_tare(massa_vazia_kg, tare_path)

    # Registra no dataset como peso 0.0
    salvar_sessao(sessions_dir, 0.0, "tare", 1, frames_registrados, duracao_s)

    print(f"[✓] TARA configurada com sucesso!")
    print(f"    Soma média de repouso: {soma_media:.1f}")
    print(f"    Offset de tara salvo: {massa_vazia_kg:.4f} kg (em {tare_path.name})")


def salvar_sessao(
    sessions_dir: Path,
    peso_real_kg: float,
    posicao: str,
    repeticao: int,
    frames_registrados: list[dict],
    duracao_s: float,
) -> tuple[Path, int]:
    """Salva todos os frames da sessão no formato calib_wide_YYYYMMDD.csv."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    hoje_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = sessions_dir / f"calib_wide_{hoje_str}.csv"

    headers = [
        "timestamp_iso",
        "session_id",
        "software_version",
        "peso_real_kg",
        "posicao",
        "repeticao",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "soma_total",
        "tempo_s",
        "estavel",
        "modelo_estimado",
    ]

    novo = not csv_path.exists() or csv_path.stat().st_size == 0
    session_uuid = str(uuid.uuid4())
    linhas_gravadas = 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if novo:
            writer.writerow(headers)

        for frame in frames_registrados:
            row = [
                datetime.now(timezone.utc).astimezone().isoformat(),
                session_uuid,
                "2.0.0",
                round(peso_real_kg, 4),
                posicao,
                repeticao,
                *(round(float(b), 4) for b in frame["b_vals"]),
                round(frame["soma_total"], 4),
                frame["tempo_s"],
                1,
                0.0,
            ]
            writer.writerow(row)
            linhas_gravadas += 1

    return csv_path, linhas_gravadas


def carregar_contadores_repeticao(sessions_dir: Path) -> dict[tuple[float, str], int]:
    """Descobre a maior repetição já gravada para cada par (peso_real_kg, posicao) hoje."""
    counters: dict[tuple[float, str], int] = {}
    hoje_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = sessions_dir / f"calib_wide_{hoje_str}.csv"
    if not csv_path.exists():
        return counters

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
    print("Este utilitário registra ensaios de peso real e calibra a TARA.")
    print("Para calibrar o ZERO/TARA, basta digitar 0 quando solicitado o peso.\n")

    dev = conectar_hid()
    print("[✓] Manta USB HID conectada com sucesso.")

    rep_counters = carregar_contadores_repeticao(sessions_dir)

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

            # Se digitou 0 -> Calibração de TARA
            if peso_kg == 0.0:
                executar_tara(dev, sessions_dir, duracao_s=5.0)
                continue

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

            dur_str = input("Tempo de amostragem em segundos [padrão 15s]: ").strip() or "15"
            duracao = float(dur_str) if dur_str.replace(".", "").isdigit() else 15.0

            input("\n-> Posicione a carga na maca e pressione ENTER para iniciar a medição...")

            frames_registrados, b_medias, soma_media = coletar_amostra(dev, duracao, is_tare=False)

            if not frames_registrados:
                print("[!] Nenhum dado detectado. Verifique se o peso foi posicionado.")
                continue

            print(f"\nResultados médios obtidos para {peso_kg} kg ({posicao}) - Repetição #{repeticao}:")
            for i, b_val in enumerate(b_medias):
                print(f"  B{i+1}: {b_val:7.1f}", end="  " if (i + 1) % 4 != 0 else "\n")
            print(f"  Soma Média: {soma_media:.1f}")

            conf = input(f"\nDeseja salvar este ensaio ({len(frames_registrados)} frames)? (S/n): ").strip().lower()
            if conf in ("", "s", "sim", "y", "yes"):
                arquivo_salvo, total_f = salvar_sessao(
                    sessions_dir, peso_kg, posicao, repeticao, frames_registrados, duracao
                )
                rep_counters[rep_key] = repeticao
                print(f"[✓] {total_f} frames registrados com sucesso (Repetição #{repeticao}) em: {arquivo_salvo.name}")

    finally:
        dev.close()
        print("\nDispositivo USB liberado.")

    print("\n" + "=" * 70)
    re_train = input("Deseja re-treinar o modelo de Machine Learning com as novas amostras agora? (S/n): ").strip().lower()
    if re_train in ("", "s", "sim", "y", "yes"):
        import subprocess
        python_exe = sys.executable
        train_script = base_dir / "scripts" / "train_weight_model.py"
        print("\n[+] Executando treinamento...")
        subprocess.run([python_exe, str(train_script)])


if __name__ == "__main__":
    main()
