"""Ajusta os ganhos α_s da etapa massa local → massa total (block_curves.json).

M = Σ_s α_s · L_s, com L_s = Σ_{ij∈s} f_s(x_ij) a massa local do setor s pelas
curvas de célula. α_s absorve o que a calibração de uma célula isolada não vê:
a variação entre células do mesmo setor, a carga repartida entre células vizinhas
pelo colchão e a massa perdida nas células abaixo de raw_min_valid.

Entrada: sessões de experimento (data/experiments/<sessão>/) com carga distribuída
de massa conhecida marcada em /experiments/mark (known_mass_g). De cada sessão saem
os trechos contínuos estáveis (estado ESTAVEL ou MEDICAO_VALIDA, sem movimento) com
a mesma massa, ciclo e sentido; cada trecho vira um ponto (L_1..L_8, massa conhecida),
recalculado do RAW gravado e da tara da sessão com as curvas em vigor.

Modos:
- global: um α para a manta inteira (precisa de 2 ou mais trechos);
- setor:  um α por setor, puxado para o α global (ridge, --lambda), porque cada
          setor só é identificável com cargas que o atinjam de forma diferente.

O padrão é só mostrar o resultado (validação deixando um trecho de fora). Com
--gravar, os ganhos vão para block_curves.json; não há reajuste automático.

Uso:
    python scripts/ajustar_ganho_massa_total.py data/experiments/2026* [--modo setor] [--gravar]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT / "src"))

from math_engine.gain_fit import fit_gains as fit, fit_report as report, write_gains  # noqa: E402
from math_engine.mass_models import BlockCurveModel  # noqa: E402
from storage.calibration_store import BLOCK_REGIONS  # noqa: E402
from storage.experiment_recorder import load_raw_frames  # noqa: E402

BLOCKS = sorted(BLOCK_REGIONS)
STABLE_STATES = {"ESTAVEL", "MEDICAO_VALIDA"}


def _true(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "sim", "yes"}


def segments(session_dir: Path, direction: str, min_frames: int) -> list[dict]:
    """Trechos contínuos estáveis com a mesma massa conhecida, ciclo, sentido e tara."""
    with (session_dir / "frames.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict] = []
    current: dict | None = None
    for row in rows:
        ok = (
            row.get("known_mass_g") not in (None, "")
            and row.get("state") in STABLE_STATES
            and not _true(row.get("movement_flag"))
            and (direction == "ALL" or row.get("loading_direction") == direction)
        )
        key = (row.get("known_mass_g"), row.get("cycle_id"), row.get("loading_direction"), row.get("tare_id"))
        if ok and current is not None and current["key"] == key and int(row["frame_idx"]) == current["last"] + 1:
            current["frames"].append(int(row["frame_idx"]))
            current["last"] = int(row["frame_idx"])
            continue
        if current is not None and len(current["frames"]) >= min_frames:
            out.append(current)
        current = {"key": key, "frames": [int(row["frame_idx"])], "last": int(row["frame_idx"])} if ok else None
    if current is not None and len(current["frames"]) >= min_frames:
        out.append(current)
    return out


def load_points(session_dirs: list[Path], model: BlockCurveModel, direction: str, min_frames: int) -> list[dict]:
    points: list[dict] = []
    for d in session_dirs:
        if not (d / "frames.csv").exists() or not (d / "raw.u8").exists():
            print(f"[!] {d}: sem frames.csv ou raw.u8, ignorada")
            continue
        raw = load_raw_frames(d).astype(np.float64)
        baselines = {f.stem: np.load(f)["baseline_raw"] for f in (d / "baselines").glob("*.npz")}
        for seg in segments(d, direction, min_frames):
            mass_g, cycle, loading, tare_id = seg["key"]
            if tare_id not in baselines:
                print(f"[!] {d.name}: trecho sem a tara {tare_id}, ignorado")
                continue
            # Mesma cadeia do worker: média da janela do x_ij = max(0, s_ij − B_ij)
            corrected = np.maximum(raw[seg["frames"]] - baselines[tare_id], 0.0).mean(axis=0)
            local = model.sector_local_sums(corrected)
            points.append({
                "session": d.name,
                "cycle_id": cycle,
                "loading_direction": loading,
                "frames": len(seg["frames"]),
                "known_mass_kg": float(mass_g) / 1000.0,
                "local_kg": [local[b] for b in BLOCKS],
            })
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessoes", nargs="+", type=Path, help="Pastas de data/experiments/")
    parser.add_argument("--curvas", type=Path, default=EDGE_ROOT / "src" / "storage" / "block_curves.json")
    parser.add_argument("--modo", choices=("global", "setor"), default="global")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0,
                        help="Peso que puxa α_s para o α global no modo setor. Padrão: 1.0")
    parser.add_argument("--direcao", choices=("UP", "DOWN", "UNKNOWN", "ALL"), default="UP",
                        help="Sentido de carga usado no ajuste (não misturar ramos). Padrão: UP")
    parser.add_argument("--min-frames", type=int, default=10, help="Frames estáveis mínimos por trecho")
    parser.add_argument("--gravar", action="store_true", help="Grava os ganhos em block_curves.json")
    args = parser.parse_args()

    model = BlockCurveModel(args.curvas)
    if not model.is_ready:
        raise SystemExit(f"{args.curvas}: curvas ausentes ou incompletas")
    points = load_points([p for p in args.sessoes if p.is_dir()], model, args.direcao, args.min_frames)
    if len(points) < 2:
        raise SystemExit(f"Só {len(points)} trecho(s) estável(is) com massa conhecida; são precisos ao menos 2")
    if args.modo == "setor" and len(points) < len(BLOCKS):
        print(f"[!] {len(points)} trechos para {len(BLOCKS)} ganhos: os α_s ficarão perto do α global")

    L = np.array([p["local_kg"] for p in points])
    M = np.array([p["known_mass_kg"] for p in points])
    if np.any(L.sum(axis=1) <= 0):
        raise SystemExit("Há trecho com massa local zero: confira a tara e as massas marcadas")
    alpha = fit(L, M, args.modo, args.lam)
    summary = report(L, M, alpha, args.modo, args.lam)

    print(f"{'sessão':<26} {'ciclo':>5} {'frames':>6} {'massa real':>10} {'local Σ':>9} {'com α':>9}")
    for p, l, pred in zip(points, L.sum(axis=1), L @ alpha):
        print(f"{p['session'][:26]:<26} {str(p['cycle_id']):>5} {p['frames']:>6} "
              f"{p['known_mass_kg']:>8.2f} kg {l:>7.2f} kg {pred:>7.2f} kg")
    print("α_s:", ", ".join(f"{b}={a:.3f}" for b, a in zip(BLOCKS, alpha)))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.gravar:
        print("Ensaio: nada gravado. Use --gravar para atualizar block_curves.json.")
        return
    write_gains(args.curvas, {b: float(a) for b, a in zip(BLOCKS, alpha)}, {
        "source": "experiments",
        "mode": args.modo,
        "lambda": args.lam if args.modo == "setor" else None,
        "loading_direction": args.direcao,
        "sessions": sorted({p["session"] for p in points}),
        **summary,
    })
    print(f"[ok] ganhos gravados em {args.curvas}. Recarregue com POST /api/v1/calibration/reload-model")


if __name__ == "__main__":
    main()
