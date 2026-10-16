"""Exporta os ensaios de célula gravados em data/experiments/ (orientações de 15/10, item 5).

Durante o experimento, marque a célula ensaiada junto com ciclo, sentido e massa:
    POST /api/v1/experiments/mark {"target_cell": "12x34", "cycle_id": 1,
                                   "loading_direction": "UP", "known_mass_g": 50}

Uma linha por patamar: trecho contínuo de frames com a mesma célula, ciclo, sentido,
massa e tara. O valor do patamar é a média da cauda estável do RAW corrigido
max(0, RAW − baseline) da célula ensaiada, como em gerar_curvas_bloco.py.
A saída alimenta scripts/comparar_generalizacao_celulas.py.

Uso:
    python scripts/exportar_ensaios_celula.py data/experiments/<sessão> [...] \
        [--saida ensaios_celula.csv] [--fracao-cauda 0.5] [--longo frames_celula.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT / "src"))

from core.settings import HID_COLS  # noqa: E402
from storage.experiment_recorder import load_raw_frames, parse_cell  # noqa: E402

PLATEAU_FIELDS = [
    "session_id", "cell", "cell_id", "row", "column", "block_id", "cycle_id", "direction",
    "known_mass_g", "tare_id", "first_frame", "last_frame", "n_frames", "n_tail",
    "baseline_raw", "raw_mean", "raw_median", "corrected_mean", "corrected_median", "corrected_std",
]
LONG_FIELDS = [
    "session_id", "frame_idx", "timestamp", "cell", "cell_id", "block_id", "cycle_id", "direction",
    "known_mass_g", "tare_id", "raw", "baseline_raw", "corrected_raw",
]


def _baselines(session_dir: Path) -> dict[str, np.ndarray]:
    return {f.stem: np.load(f)["baseline_raw"] for f in (session_dir / "baselines").glob("*.npz")}


def _frames(session_dir: Path) -> Iterator[dict[str, Any]]:
    """Frames com célula ensaiada e massa conhecida, com o RAW e o baseline daquela célula."""
    raw = load_raw_frames(session_dir)
    baselines = _baselines(session_dir)
    with (session_dir / "frames.csv").open(encoding="utf-8") as fh:
        for meta in csv.DictReader(fh):
            cell = (meta.get("target_cell") or "").strip()
            if not cell or meta.get("known_mass_g") in (None, ""):
                continue
            r, c = parse_cell(cell)
            idx = int(meta["frame_idx"])
            base = baselines.get(meta.get("tare_id") or "")
            b = float(base[r, c]) if base is not None else 0.0
            value = float(raw[idx, r, c])
            yield {
                "session_id": meta["session_id"],
                "frame_idx": idx,
                "timestamp": meta.get("timestamp"),
                "cell": f"{r}x{c}",
                "cell_id": r * HID_COLS + c,
                "row": r,
                "column": c,
                "block_id": int(meta["target_block"]) if meta.get("target_block") else None,
                "cycle_id": int(meta["cycle_id"]) if meta.get("cycle_id") not in (None, "") else None,
                "direction": meta.get("loading_direction") or "UNKNOWN",
                "known_mass_g": float(meta["known_mass_g"]),
                "tare_id": meta.get("tare_id") or None,
                "raw": value,
                "baseline_raw": b,
                "corrected_raw": max(0.0, value - b),
            }


def _label(f: dict[str, Any]) -> tuple:
    return f["session_id"], f["cell"], f["cycle_id"], f["direction"], f["known_mass_g"], f["tare_id"]


def _segments(frames: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Trechos contínuos de frames com os mesmos rótulos."""
    group: list[dict[str, Any]] = []
    for f in frames:
        if group and (_label(f) != _label(group[-1]) or f["frame_idx"] != group[-1]["frame_idx"] + 1):
            yield group
            group = []
        group.append(f)
    if group:
        yield group


def _summarize(group: list[dict[str, Any]], tail_fraction: float) -> dict[str, Any]:
    start = int(len(group) * (1.0 - tail_fraction))
    tail = np.array([g["corrected_raw"] for g in group[start:]])
    raw_tail = np.array([g["raw"] for g in group[start:]])
    first = group[0]
    return {
        **{k: first[k] for k in ("session_id", "cell", "cell_id", "row", "column", "block_id",
                                 "cycle_id", "direction", "known_mass_g", "tare_id")},
        "first_frame": first["frame_idx"],
        "last_frame": group[-1]["frame_idx"],
        "n_frames": len(group),
        "n_tail": len(tail),
        "baseline_raw": round(first["baseline_raw"], 4),
        "raw_mean": round(float(np.mean(raw_tail)), 4),
        "raw_median": round(float(np.median(raw_tail)), 4),
        "corrected_mean": round(float(np.mean(tail)), 4),
        "corrected_median": round(float(np.median(tail)), 4),
        "corrected_std": round(float(np.std(tail)), 4),
    }


def plateaus(frames: list[dict[str, Any]], tail_fraction: float) -> list[dict[str, Any]]:
    """Um resumo por patamar, com a cauda estável de cada trecho."""
    return [_summarize(g, tail_fraction) for g in _segments(frames)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessoes", nargs="+", type=Path, help="Pastas de data/experiments/<sessão>")
    parser.add_argument("--saida", type=Path, default=Path("ensaios_celula.csv"), help="CSV de patamares")
    parser.add_argument("--fracao-cauda", type=float, default=0.5,
                        help="Fração final de cada patamar usada como regime estável. Padrão: 0.5")
    parser.add_argument("--longo", type=Path, default=None, help="Também grava um CSV com cada frame da célula ensaiada")
    args = parser.parse_args()
    if not 0.0 < args.fracao_cauda <= 1.0:
        raise SystemExit("--fracao-cauda deve estar em (0, 1]")

    all_frames: list[dict[str, Any]] = []
    for session in args.sessoes:
        if not (session / "frames.csv").exists():
            raise SystemExit(f"{session}: não é uma pasta de experimento (falta frames.csv)")
        all_frames.extend(_frames(session))
    if not all_frames:
        raise SystemExit("Nenhum frame com célula ensaiada (target_cell) e massa conhecida (known_mass_g).")

    rows = plateaus(all_frames, args.fracao_cauda)
    with args.saida.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLATEAU_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if args.longo is not None:
        with args.longo.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LONG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_frames)

    cells = sorted({r["cell"] for r in rows})
    print(f"[ok] {args.saida}: {len(rows)} patamares, {len(cells)} célula(s): {', '.join(cells)}")
    for cell in cells:
        mine = [r for r in rows if r["cell"] == cell]
        cycles = sorted({r["cycle_id"] for r in mine if r["cycle_id"] is not None})
        print(f"  {cell} (setor {mine[0]['block_id']}): {len(mine)} patamares, ciclos {cycles or '—'}, "
              f"sentidos {sorted({r['direction'] for r in mine})}")


if __name__ == "__main__":
    main()
