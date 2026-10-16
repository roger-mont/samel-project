"""Testes das curvas por setor (RAW → massa local) e dos ganhos α_s (massa local → massa total)."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_EDGE = Path(__file__).resolve().parents[2] / "apps" / "edge-service"
if str(_EDGE / "src") not in sys.path:
    sys.path.insert(0, str(_EDGE / "src"))

from math_engine.mass_models import BlockCurveModel, WindowFrame
from storage.calibration_store import BLOCK_REGIONS

SHAPE = (32, 64)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _EDGE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # dataclasses do script precisam do módulo registrado
    spec.loader.exec_module(module)
    return module


gerar = _load_script("gerar_curvas_bloco")
ganho = _load_script("ajustar_ganho_massa_total")


def _write_curves(path: Path, gains: dict[int, float] | None = None, calibrated: bool = False) -> None:
    """1 kg a cada 100 RAW em todos os setores."""
    blocks = {
        str(bid): {
            "kind": "linear_interp",
            "params": {"raw": [0.0, 100.0], "mass_kg": [0.0, 1.0]},
            "raw_min_valid": 1.0,
            "raw_max_valid": 100.0,
            "mass_min_valid": 0.0,
            "mass_max_valid": 1.0,
            "low_sensitivity_raw": None,
        }
        for bid in BLOCK_REGIONS
    }
    data = {"calibration_version": "MC-TEST-003", "blocks": blocks}
    if gains is not None:
        data["aggregation"] = {"gains": {str(b): g for b, g in gains.items()}, "calibrated": calibrated}
    path.write_text(json.dumps(data), encoding="utf-8")


# ----------------------------------------------------------------------------
# Massa local → massa total no modelo
# ----------------------------------------------------------------------------

def test_without_aggregation_gains_default_to_one(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_curves(path)
    model = BlockCurveModel(path)
    assert model.gains == {b: 1.0 for b in BLOCK_REGIONS}
    assert not model.gains_calibrated


def test_sector_gain_scales_only_its_sector(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_curves(path, gains={1: 2.0, 5: 0.5}, calibrated=True)
    model = BlockCurveModel(path)
    assert model.gains_calibrated and model.describe()["gains"][1] == 2.0

    corrected = np.zeros(SHAPE)
    corrected[BLOCK_REGIONS[1]] = 10.0   # 256 células × 0,1 kg = 25,6 kg locais
    corrected[BLOCK_REGIONS[5]] = 10.0   # idem no setor 5
    corrected[BLOCK_REGIONS[3]] = 10.0   # idem no setor 3, α = 1

    local = model.sector_local_sums(corrected)
    assert local[1] == pytest.approx(25.6) and local[2] == 0.0

    est = model.estimate(WindowFrame(corrected, corrected))
    assert est.mass_kg == pytest.approx(25.6 * 2.0 + 25.6 * 0.5 + 25.6)

    data = model.breakdown(corrected, active_blocks=list(BLOCK_REGIONS))
    by = {s["sector"]: s for s in data["sectors"]}
    assert by[1]["local_mass_kg"] == pytest.approx(25.6)
    assert by[1]["mass_kg"] == pytest.approx(51.2)
    assert by[5]["mass_kg"] == pytest.approx(12.8)
    assert data["total_mass_kg"] == pytest.approx(est.mass_kg, abs=1e-3)
    # o mapa por célula continua em massa local, sem o ganho
    assert data["cell_mass_g"][0][48] == pytest.approx(100.0)


def test_non_positive_gain_invalidates_curves(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_curves(path, gains={2: 0.0})
    assert not BlockCurveModel(path).is_ready


# ----------------------------------------------------------------------------
# Gerador das curvas por setor
# ----------------------------------------------------------------------------

def _write_cell_csv(folder: Path, up: list[tuple[float, float]], down: list[tuple[float, float]]) -> None:
    """Patamares de 10 amostras: 4 de transitório (RAW − 20) e 6 estáveis."""
    folder.mkdir(parents=True)
    fields = ["Ponto_Idx", "Massa_kg", "RAW_Amostra", "RAW_Medio", "Baseline_Bij",
              "RAW_Corrigido", "Direcao", "Estavel", "Tempo_Estabilizacao_s"]
    with (folder / "calibracao_memoria - média 1.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        idx = 0
        for direction, branch in (("UP", up), ("DOWN", down)):
            for mass, raw in branch:
                idx += 1
                for k in range(10):
                    sample = raw + 2.0 - 20.0 if k < 4 else raw + 2.0   # baseline 2
                    w.writerow({"Ponto_Idx": idx, "Massa_kg": mass, "RAW_Amostra": sample, "RAW_Medio": raw,
                                "Baseline_Bij": 2.0, "RAW_Corrigido": raw - 8.0, "Direcao": direction,
                                "Estavel": "SIM", "Tempo_Estabilizacao_s": 0.5})


def test_sector_of_follows_block_regions() -> None:
    assert gerar.sector_of(9, 53) == 1
    assert gerar.sector_of(12, 34) == 2
    assert gerar.sector_of(13, 14) == 4
    assert gerar.sector_of(26, 11) == 5
    assert gerar.sector_of(19, 50) == 8
    with pytest.raises(ValueError):
        gerar.sector_of(40, 10)


def test_generator_builds_one_curve_per_sector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    up_a = [(0.05, 20.0), (0.10, 50.0), (0.15, 45.0), (0.20, 70.0)]   # 0,15 fora de ordem
    _write_cell_csv(docs / "1. Célula 5x50", up_a, [(0.15, 60.0), (0.10, 55.0), (0.05, 25.0)])
    _write_cell_csv(docs / "2. Célula 20X5", [(0.05, 10.0), (0.10, 30.0), (0.20, 90.0)],
                    [(0.10, 35.0), (0.05, 12.0)])
    (docs / "Células-Setores.txt").write_text("Setor 1: 5X50\nSetor 5: 20X5\n", encoding="utf-8")
    out = tmp_path / "block_curves.json"
    monkeypatch.setattr(sys, "argv", ["gerar", "--docs", str(docs), "--saida", str(out), "--limiar-g-por-raw", "3"])
    gerar.main()

    data = json.loads(out.read_text(encoding="utf-8"))
    b1, b5, b2 = data["blocks"]["1"], data["blocks"]["5"], data["blocks"]["2"]
    assert b1["cells"] == ["5x50"] and b1["source"] == "cells"
    assert b5["cells"] == ["20x5"]
    assert b2["source"] == "fallback_global" and b2["cells"] == []
    # cauda estável sem baseline: o transitório não entra
    src = {s["cell"]: s for s in data["sources"]}
    assert src["5x50"]["points_up"][0]["raw_corrected"] == pytest.approx(20.0)
    # isotônica: 0,10 e 0,15 empatam em 47,5 e viram um nó de 0,125 kg
    assert b1["params"]["raw"] == [0.0, 20.0, 47.5, 70.0]
    assert b1["params"]["mass_kg"] == pytest.approx([0.0, 0.05, 0.125, 0.2])
    assert b1["raw_max_valid"] == 70.0 and b1["mass_max_valid"] == 0.2
    # 20 → 47,5 RAW vale 75 g / 27,5 RAW ≈ 2,7 g/RAW; 47,5 → 70 vale 75 g / 22,5 ≈ 3,3 g/RAW
    assert b1["low_sensitivity_raw"] == pytest.approx(47.5)
    assert b1["down_curve_not_used"] is not None
    assert data["aggregation"]["calibrated"] is False
    assert set(data["aggregation"]["gains"].values()) == {1.0}

    model = BlockCurveModel(out)
    assert model.is_ready
    corrected = np.zeros(SHAPE)
    corrected[5, 50] = 70.0    # setor 1, topo da curva → 0,2 kg
    corrected[20, 5] = 30.0    # setor 5 → 0,1 kg
    assert model.estimate(WindowFrame(corrected, corrected)).mass_kg == pytest.approx(0.3)


def test_generator_without_threshold_leaves_low_sensitivity_undefined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    _write_cell_csv(docs / "Célula 5x50", [(0.05, 20.0), (0.10, 50.0)], [])
    out = tmp_path / "block_curves.json"
    monkeypatch.setattr(sys, "argv", ["gerar", "--docs", str(docs), "--saida", str(out)])
    gerar.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert all(b["low_sensitivity_raw"] is None for b in data["blocks"].values())


# ----------------------------------------------------------------------------
# Ajuste dos ganhos α_s
# ----------------------------------------------------------------------------

def _write_session(directory: Path, loads: list[tuple[float, dict[int, float]]]) -> None:
    """Uma sessão com um trecho estável por carga; cada carga põe RAW uniforme em alguns setores."""
    (directory / "baselines").mkdir(parents=True)
    baseline = np.full(SHAPE, 3.0)
    np.savez_compressed(directory / "baselines" / "T1.npz", baseline_raw=baseline, baseline_std=np.zeros(SHAPE))
    frames = []
    rows = []
    idx = 0
    for mass_g, raw_by_sector in loads:
        m = baseline.copy()
        for bid, raw in raw_by_sector.items():
            m[BLOCK_REGIONS[bid]] += raw
        for k in range(15):
            state = "ESTABILIZANDO" if k < 3 else "MEDICAO_VALIDA"
            frames.append(m)
            rows.append({"frame_idx": idx, "known_mass_g": mass_g, "cycle_id": 1, "loading_direction": "UP",
                         "tare_id": "T1", "state": state, "movement_flag": "False"})
            idx += 1
    np.clip(np.array(frames), 0, 255).astype(np.uint8).tofile(directory / "raw.u8")
    with (directory / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_segments_keep_only_stable_frames_with_known_mass(tmp_path: Path) -> None:
    session = tmp_path / "s1"
    _write_session(session, [(25600.0, {1: 10.0}), (51200.0, {1: 20.0})])
    segs = ganho.segments(session, "UP", min_frames=10)
    assert [len(s["frames"]) for s in segs] == [12, 12]
    assert ganho.segments(session, "DOWN", min_frames=10) == []


def test_gain_fit_recovers_global_and_sector_gains(tmp_path: Path) -> None:
    curves = tmp_path / "block_curves.json"
    _write_curves(curves)
    model = BlockCurveModel(curves)
    # local: 10 RAW × 256 células × 0,01 kg = 25,6 kg por setor carregado
    session = tmp_path / "s1"
    _write_session(session, [(25600.0 * 1.25, {1: 10.0}), (2 * 25600.0 * 1.25, {1: 10.0, 8: 10.0})])
    points = ganho.load_points([session], model, "UP", min_frames=10)
    assert len(points) == 2
    L = np.array([p["local_kg"] for p in points])
    M = np.array([p["known_mass_kg"] for p in points])
    assert L[0][0] == pytest.approx(25.6)
    assert ganho.fit(L, M, "global", 1.0) == pytest.approx(np.full(8, 1.25))

    # setor 1 com α = 1,2 e setor 8 com α = 0,9, cargas diferentes em cada um
    loads = [(25600.0 * 1.2, {1: 10.0}), (25600.0 * 0.9, {8: 10.0}),
             (25600.0 * (1.2 + 0.9), {1: 10.0, 8: 10.0}), (25600.0 * (2 * 1.2 + 0.9), {1: 20.0, 8: 10.0})]
    session2 = tmp_path / "s2"
    _write_session(session2, loads)
    pts = ganho.load_points([session2], model, "UP", min_frames=10)
    L = np.array([p["local_kg"] for p in pts])
    M = np.array([p["known_mass_kg"] for p in pts])
    alpha = ganho.fit(L, M, "setor", 1e-6)
    assert alpha[0] == pytest.approx(1.2, rel=1e-3) and alpha[7] == pytest.approx(0.9, rel=1e-3)
    summary = ganho.report(L, M, alpha, "setor", 1e-6)
    assert summary["fitted"]["rmse_kg"] < 0.01


# ----------------------------------------------------------------------------
# Comparação de modelos (diagnóstico ao vivo)
# ----------------------------------------------------------------------------

def _definition_with_points() -> dict:
    """Curvas de setor com pontos reais de calibração: setor s satura em 100 + 10·s RAW."""
    blocks = {}
    sources = []
    for bid in BLOCK_REGIONS:
        top = 100.0 + 10 * bid
        masses = [0.05, 0.10, 0.15, 0.20]
        raws = [top * (1 - np.exp(-m / 0.08)) / (1 - np.exp(-0.2 / 0.08)) for m in masses]
        blocks[str(bid)] = {
            "kind": "pchip", "params": {"raw": [0.0, *raws], "mass_kg": [0.0, *masses]},
            "raw_min_valid": 1.0, "raw_max_valid": raws[-1], "mass_min_valid": 0.0, "mass_max_valid": 0.2,
            "low_sensitivity_raw": None,
            "fit": [{"mass_kg": m, "raw_mean": r, "raw_isotonic": r} for m, r in zip(masses, raws)],
        }
        sources.append({"sha1": f"s{bid}", "block": bid,
                        "points_up": [{"mass_kg": m, "raw_corrected": r} for m, r in zip(masses, raws)]})
    return {"calibration_version": "MC-TEST-CMP", "generated_at": "t0", "blocks": blocks, "sources": sources}


def test_comparison_pchip_matches_active_curve(tmp_path: Path) -> None:
    from math_engine.model_comparison import ModelComparison

    definition = _definition_with_points()
    path = tmp_path / "block_curves.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    model = BlockCurveModel(path)
    models = [("pchip_isotonica", "PCHIP"), ("polinomio_3", "Polinômio 3")]
    cmp = ModelComparison(definition, model.gains, models=models)

    corrected = np.zeros(SHAPE)
    corrected[2:6, 50:54] = 60.0     # setor 1
    corrected[20:23, 3:7] = 90.0     # setor 5
    corrected[30, 60] = 500.0        # setor 8, acima da faixa: fica no topo da curva
    active = list(BLOCK_REGIONS)
    out = cmp.evaluate(corrected, active)
    pchip = next(r for r in out["sector"] if r["model"] == "pchip_isotonica")
    est = model.estimate(WindowFrame(corrected, corrected)).mass_kg
    assert pchip["total_mass_kg"] == pytest.approx(est, rel=1e-3)
    assert pchip["sectors_kg"]["8"] == pytest.approx(0.2, abs=1e-3)
    assert {r["model"] for r in out["global"]} == {"pchip_isotonica", "polinomio_3"}
    # setores fora da medição não entram no total
    only_1 = cmp.evaluate(corrected, [1])
    p1 = next(r for r in only_1["sector"] if r["model"] == "pchip_isotonica")
    assert p1["total_mass_kg"] == pytest.approx(p1["sectors_kg"]["1"])


def test_comparison_models_are_anchored_at_zero_and_cached() -> None:
    from math_engine.model_comparison import ComparisonCache, ModelComparison

    definition = _definition_with_points()
    cmp = ModelComparison(definition, {}, models=[("ml_knn", "kNN"), ("polinomio_3", "Polinômio 3")])
    for mode in ("sector", "global"):
        for key in ("ml_knn", "polinomio_3"):
            assert cmp.tables[mode][key][3](np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-3)
    curves = cmp.curves("sector", 3, n=10)
    assert len(curves[0]["samples_raw_mass_g"]) == 10 and curves[0]["raw_max"] == pytest.approx(130.0)

    cache = ComparisonCache()
    a = cache.get(definition, {})
    assert cache.get(definition, {}) is a
    assert cache.get({**definition, "generated_at": "t1"}, {}) is not a


def test_unavailable_model_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    from math_engine import model_comparison as mc

    def broken(x, m):
        raise ImportError("biblioteca ausente")
    monkeypatch.setitem(mc.ALL_FITTERS, "quebrado", broken)
    cmp = mc.ModelComparison(_definition_with_points(), {}, models=[("quebrado", "Quebrado"), ("pchip_isotonica", "PCHIP")])
    rows = cmp.evaluate(np.zeros(SHAPE), list(BLOCK_REGIONS))["sector"]
    assert rows[0]["available"] is False and "ausente" in rows[0]["error"]
    assert rows[1]["available"] is True
