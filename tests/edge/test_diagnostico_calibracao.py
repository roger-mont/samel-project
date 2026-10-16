"""Testes do diagnóstico da calibração pedido nas orientações de 15/10 (MC-2026-002-PROV)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_EDGE = Path(__file__).resolve().parents[2] / "apps" / "edge-service"
for sub in ("src", "scripts"):
    if str(_EDGE / sub) not in sys.path:
        sys.path.insert(0, str(_EDGE / sub))

from fastapi.testclient import TestClient

from api.server import app
from core.settings import CalibrationParams
from database.db_local import LocalDatabase
from hardware.serial_reader import BaseFrameReader
from math_engine.calibration_audit import audit_calibration, isotonic_correction
from math_engine.gain_fit import position_report
from math_engine.mass_models import (
    CELL_SITUATIONS, BlockCurveModel, ModelRegistry, build_curve, build_curve_derivative,
)
from math_engine.residual_mass import ResidualMassTracker
from math_engine.spatial_baseline import SpatialBaseline
from storage.calibration_store import BLOCK_REGIONS
from storage.experiment_recorder import ExperimentRecorder
from storage.gain_points import GainPointStore
from workers.acquisition_worker import AcquisitionWorker

SHAPE = (32, 64)
DT = 0.05


def _origin(block: int) -> tuple[int, int]:
    rs, cs = BLOCK_REGIONS[block]
    return rs.start, cs.start


def _write_curves(path: Path) -> None:
    """Setores lineares (1 kg a cada 100 RAW); setor 2 com topo íngreme e curva DOWN;
    setor 5 com 25 g sem resposta (RAW 0)."""
    linear = {"kind": "linear_interp", "params": {"raw": [0.0, 100.0], "mass_kg": [0.0, 1.0]},
              "raw_min_valid": 1.0, "raw_max_valid": 100.0, "mass_min_valid": 0.0, "mass_max_valid": 1.0,
              "low_sensitivity_raw": None}
    blocks = {str(b): dict(linear) for b in BLOCK_REGIONS}
    blocks["2"] = {
        "kind": "pchip", "params": {"raw": [0.0, 10.0, 40.0, 60.0, 62.0], "mass_kg": [0.0, 0.05, 0.1, 0.15, 0.2]},
        "raw_min_valid": 1.0, "raw_max_valid": 62.0, "mass_min_valid": 0.0, "mass_max_valid": 0.2,
        "low_sensitivity_raw": None,
        "down_curve_not_used": {"kind": "linear_interp", "params": {"raw": [0.0, 62.0], "mass_kg": [0.0, 0.2]}},
    }
    blocks["5"] = {
        "kind": "pchip", "params": {"raw": [0.0, 20.0, 50.0], "mass_kg": [0.0, 0.05, 0.1]},
        "raw_min_valid": 1.0, "raw_max_valid": 50.0, "mass_min_valid": 0.0, "mass_max_valid": 0.1,
        "low_sensitivity_raw": None,
        "fit": [{"mass_kg": 0.025, "raw_mean": 0.0, "raw_isotonic": 0.0},
                {"mass_kg": 0.05, "raw_mean": 20.0, "raw_isotonic": 20.0},
                {"mass_kg": 0.1, "raw_mean": 50.0, "raw_isotonic": 50.0}],
    }
    path.write_text(json.dumps({"calibration_version": "MC-TEST-DIAG", "blocks": blocks}), encoding="utf-8")


@pytest.fixture()
def model(tmp_path: Path) -> BlockCurveModel:
    path = tmp_path / "block_curves.json"
    _write_curves(path)
    m = BlockCurveModel(path)
    assert m.is_ready
    return m


# ----------------------------------------------------------------------------
# dm/dRAW e situação de cada célula
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind,params", [
    ("pchip", {"raw": [0.0, 13.6, 44.3, 56.7, 68.9, 75.0], "mass_kg": [0.0, 0.025, 0.05, 0.1, 0.2, 0.275]}),
    ("linear_interp", {"raw": [0.0, 10.0, 30.0], "mass_kg": [0.0, 0.1, 0.15]}),
    ("power", {"a": 0.001, "b": 1.5}),
])
def test_derivative_matches_finite_difference(kind: str, params: dict) -> None:
    f = build_curve(kind, params)
    df = build_curve_derivative(kind, params)
    top = params["raw"][-1] - 1.0 if "raw" in params else 60.0   # dentro da faixa dos nós
    xs = np.linspace(2.0, top, 37) + 0.123                       # longe dos nós (a linear tem quinas)
    numeric = (f(xs + 1e-5) - f(xs - 1e-5)) / 2e-5
    np.testing.assert_allclose(df(xs), numeric, rtol=1e-3, atol=1e-6)


def test_cell_situations_separate_saturation_from_metrological_limits(model: BlockCurveModel) -> None:
    corrected = np.zeros(SHAPE)
    raw = np.zeros(SHAPE)
    r2, c2 = _origin(2)
    r5, c5 = _origin(5)
    corrected[r2, c2] = 20.0          # dentro da curva
    corrected[r2, c2 + 1] = 61.0      # topo íngreme: 25 g por RAW
    corrected[r2, c2 + 2] = 70.0      # acima do último nó
    corrected[r2, c2 + 3] = 0.5       # abaixo de raw_min_valid
    corrected[r5, c5] = 10.0          # setor 5, antes do primeiro nó (25 g deu RAW 0)
    corrected[r5, c5 + 1] = 30.0
    raw[r2, c2 + 4] = 255.0           # saturação digital, sem carga corrigida

    cd = model.cell_diagnostics(corrected, raw, saturation_raw=255.0, low_sensitivity_g_per_raw=10.0)
    sit = lambda r, c: CELL_SITUATIONS[cd.situation[r, c]]  # noqa: E731
    assert sit(r2, c2) == "IN_RANGE"
    assert sit(r2, c2 + 1) == "LOW_SENSITIVITY" and cd.dm_d_raw_g[r2, c2 + 1] > 10.0
    assert sit(r2, c2 + 2) == "OUT_OF_CALIBRATION"
    assert sit(r2, c2 + 3) == "INACTIVE"
    assert sit(r5, c5) == "LOW_SUPPORT"
    assert sit(r5, c5 + 1) == "IN_RANGE"
    assert sit(r2, c2 + 4) == "DIGITAL_SATURATION"
    assert cd.local_mass_kg[r2, c2 + 2] == pytest.approx(0.2)   # cortada no último nó

    # Limite 0 desliga o LOW_SENSITIVITY por dm/dRAW
    off = model.cell_diagnostics(corrected, raw, 255.0, 0.0)
    assert CELL_SITUATIONS[off.situation[r2, c2 + 1]] == "IN_RANGE"


def test_sector_summary_counts_and_identifies_cells(model: BlockCurveModel) -> None:
    corrected = np.zeros(SHAPE)
    r2, c2 = _origin(2)
    corrected[r2, c2] = 20.0
    corrected[r2 + 1, c2 + 3] = 70.0
    corrected[_origin(1)] = 50.0     # setor 1 linear: 0,5 kg

    summary = model.sector_summary(model.cell_diagnostics(corrected), active_blocks=list(BLOCK_REGIONS))
    s2 = next(s for s in summary["sectors"] if s["sector"] == 2)
    assert s2["cells_out_of_calibration"] == 1
    assert s2["out_of_calibration_cells"] == [f"{r2 + 1}x{c2 + 3}"]
    assert s2["out_of_calibration_mass_kg"] == pytest.approx(0.2)
    # DOWN linear até 62 RAW: 20 RAW → 0,0645 kg; 70 RAW (cortado) → 0,2 kg
    up = float(model.local_mass(2, np.array([20.0]))[0]) + 0.2
    assert s2["local_mass_down_kg"] == pytest.approx(20 / 62 * 0.2 + 0.2, abs=1e-4)
    assert s2["hysteresis_up_minus_down_kg"] == pytest.approx(up - s2["local_mass_down_kg"], abs=1e-4)

    t = summary["totals"]
    assert t["n_active_cells"] == 3
    assert t["n_out_of_calibration"] == 1
    assert t["mass_without_gain_kg"] == pytest.approx(0.5 + up, abs=1e-4)
    assert t["sectors_without_down_curve"] == [1, 3, 4, 5, 6, 7, 8]

    only_1 = model.sector_summary(model.cell_diagnostics(corrected), active_blocks=[1])["totals"]
    assert only_1["mass_kg"] == pytest.approx(0.5) and only_1["n_out_of_calibration"] == 0


def test_cell_records_list_contribution_of_each_cell(model: BlockCurveModel) -> None:
    corrected = np.zeros(SHAPE)
    raw = np.full(SHAPE, 3.0)
    r2, c2 = _origin(2)
    corrected[r2, c2] = 20.0
    corrected[r2, c2 + 1] = 70.0
    raw[r2, c2] += 20.0
    raw[r2, c2 + 1] += 70.0
    cd = model.cell_diagnostics(corrected, raw, 255.0, 10.0)
    recs = model.cell_records(cd, raw, np.full(SHAPE, 3.0))
    assert [r["cell"] for r in recs] == [f"{r2}x{c2 + 1}", f"{r2}x{c2}"]    # maior massa primeiro
    top = recs[0]
    assert top["sector"] == 2 and top["raw"] == 73.0 and top["baseline"] == 3.0 and top["corrected_raw"] == 70.0
    assert top["local_mass_g"] == pytest.approx(200.0) and top["flags"] == ["OUT_OF_CALIBRATION"]
    assert top["cell_id"] == r2 * 64 + c2 + 1


# ----------------------------------------------------------------------------
# Massa residual com a maca vazia
# ----------------------------------------------------------------------------

def test_residual_tracker_statistics_and_reset() -> None:
    tracker = ResidualMassTracker()
    for i, v in enumerate([0.1, -0.1, 0.3]):
        tracker.update("tara-1", float(i), v, v / 2, {1: v})
    snap = tracker.snapshot()
    assert snap["n_frames"] == 3 and snap["duration_s"] == 2.0
    assert snap["mean_kg"] == pytest.approx(0.1)
    assert snap["std_kg"] == pytest.approx(0.2)
    assert snap["max_abs_kg"] == pytest.approx(0.3)
    assert snap["sectors"]["1"]["mean_kg"] == pytest.approx(0.1)

    tracker.update("tara-2", 10.0, 1.0, 1.0, {})       # tara nova recomeça
    assert tracker.snapshot()["n_frames"] == 1 and tracker.snapshot()["tare_id"] == "tara-2"
    tracker.reset()
    assert tracker.snapshot()["n_frames"] == 0 and tracker.snapshot()["tare_id"] == "tara-2"


class _NoReader(BaseFrameReader):
    def is_connected(self) -> bool:
        return True

    def read_frame(self) -> np.ndarray:
        return np.zeros(SHAPE)

    def close(self) -> None:
        pass


class _Bench:
    def __init__(self, tmp: Path) -> None:
        self.curves = tmp / "block_curves.json"
        _write_curves(self.curves)
        self.worker = AcquisitionWorker(
            db=LocalDatabase(db_path=str(tmp / "edge.db")),
            reader=_NoReader(),
            baseline=SpatialBaseline(tmp / "baseline.json"),
            models=ModelRegistry({"block_curves": BlockCurveModel(self.curves)}, selection="block_curves"),
            recorder=ExperimentRecorder(tmp / "exp"),
            params=CalibrationParams(t_tare_s=0.5, min_stable_time_s=1.0),
            active_blocks=list(BLOCK_REGIONS),
            gain_points=GainPointStore(tmp / "gain_points.json"),
        )
        self.now = 1000.0

    def feed(self, frame: np.ndarray, seconds: float) -> None:
        for _ in range(int(round(seconds / DT))):
            self.now += DT
            self.worker.process_frame(frame, self.now)

    def tare(self) -> None:
        self.worker.trigger_tare()
        self.feed(np.full(SHAPE, 2.0), 1.0)


@pytest.fixture()
def bench(tmp_path: Path) -> _Bench:
    return _Bench(tmp_path)


def _residual_frame() -> np.ndarray:
    """Maca vazia com deriva: 10 células do setor 1 com +5 RAW (soma 50 < limiar de carga)."""
    frame = np.full(SHAPE, 2.0)
    r, c = _origin(1)
    frame[r, c:c + 10] += 5.0
    return frame


def test_worker_measures_residual_only_with_empty_bed(bench: _Bench) -> None:
    assert bench.worker.residual.snapshot()["n_frames"] == 0
    bench.tare()
    bench.feed(_residual_frame(), 2.0)
    snap = bench.worker.residual.snapshot()
    assert snap["tare_id"] == bench.worker.baseline.record.tare_id
    assert snap["n_frames"] > 20
    assert snap["last_kg"] == pytest.approx(10 * 0.05, abs=1e-6)   # 10 células × 5 RAW × 0,01 kg
    assert snap["sectors"]["1"]["max_abs_kg"] == pytest.approx(0.5, abs=1e-6)

    load = np.full(SHAPE, 2.0)
    load[BLOCK_REGIONS[3]] += 30.0                                    # carga: não conta como residual
    n = snap["n_frames"]
    bench.feed(load, 1.0)
    assert bench.worker.residual.snapshot()["n_frames"] == n


def test_live_snapshot_exposes_item_8_fields(bench: _Bench) -> None:
    bench.tare()
    load = np.full(SHAPE, 2.0)
    r2, c2 = _origin(2)
    load[BLOCK_REGIONS[1]] += 30.0
    load[r2, c2:c2 + 3] += 70.0                                       # 3 células acima da curva do setor 2
    bench.feed(load, 3.0)
    live = bench.worker.get_live_snapshot()
    for key in ("weight_kg", "weight_status", "sector_local_mass_kg", "n_active_cells", "n_out_of_calibration",
                "n_low_sensitivity", "n_digital_saturation", "residual_mass_empty", "stabilization_time_s",
                "movement_index", "calibration_version", "tare_id", "mass_without_gain_kg", "calibration_alerts"):
        assert key in live, key
    assert set(live["sector_local_mass_kg"]) == {str(b) for b in BLOCK_REGIONS}
    assert live["sector_local_mass_kg"]["1"] == pytest.approx(256 * 0.3, rel=1e-3)
    assert live["n_out_of_calibration"] == 3 and live["out_of_calibration_by_sector"]["2"] == 3
    assert live["measurement_quality"]["OUT_OF_CALIBRATION"] is True
    assert "LOW_DATA_SUPPORT" in live["measurement_quality"]


# ----------------------------------------------------------------------------
# Auditoria das curvas
# ----------------------------------------------------------------------------

def test_isotonic_correction_keeps_measured_and_flags_large_changes() -> None:
    small = isotonic_correction([0.05, 0.1, 0.15], [10.0, 30.0, 29.5])
    assert small["adjusted_masses_kg"] == [0.1, 0.15] and not small["alert"]
    big = isotonic_correction([0.05, 0.1, 0.15, 0.2], [10.0, 40.0, 20.0, 50.0])
    assert big["max_abs_raw"] == pytest.approx(10.0) and big["alert"]


def test_audit_reports_problem_sectors() -> None:
    definition = {
        "calibration_version": "X",
        "blocks": {
            "2": {"cells": ["12x34", "12x37"], "source": "cells", "params": {"raw": [0, 10, 30]},
                  "fit": [{"mass_kg": 0.05, "raw_mean": 10, "raw_isotonic": 10},
                          {"mass_kg": 0.1, "raw_mean": 30, "raw_isotonic": 30}],
                  "sensitivity_g_per_raw": [{"g_per_raw": 2.0}, {"g_per_raw": 20.0}]},
            "5": {"cells": ["26x11"], "source": "cells", "params": {"raw": [0, 13.0, 29.0]},
                  "fit": [{"mass_kg": 0.025, "raw_mean": 0.0, "raw_isotonic": 0.0},
                          {"mass_kg": 0.05, "raw_mean": 13.0, "raw_isotonic": 13.0}],
                  "validation": {"up_curve_on_down_bias_g": 32.2, "up_curve_on_down_rmse_g": 34.6}},
        },
        "sources": [
            {"cell": "12x37", "block": 2,
             "points_up": [{"mass_kg": m, "raw_corrected": r} for m, r in [(0.05, 9), (0.1, 62)]],
             "points_down_not_used": [{"mass_kg": m, "raw_corrected": r} for m, r in [(0.05, 40), (0.1, 30)]]},
        ],
    }
    audit = audit_calibration(definition, low_sensitivity_g_per_raw=10.0)
    text = "\n".join(audit["alerts"])
    assert "Setor 2: patamares fora de ordem na célula 12x37 (DOWN)" in text
    assert "Setor 2: até 20.0 g por RAW" in text
    assert "Setor 5: curva de uma célula só (26x11)" in text
    assert "Setor 5: 25 g sem resposta (RAW 0)" in text
    assert "Setor 5: viés UP→DOWN de +32 g" in text
    assert audit["sectors"]["5"]["low_support_until_raw"] == 13.0


def test_real_curves_audit_matches_the_15_10_notes() -> None:
    definition = json.loads((_EDGE / "src" / "storage" / "block_curves.json").read_text(encoding="utf-8"))
    audit = audit_calibration(definition, 10.0)
    sectors = audit["sectors"]
    assert sectors["4"]["zero_response_masses_kg"] == [0.025] and sectors["5"]["zero_response_masses_kg"] == [0.025]
    assert sectors["5"]["hysteresis"]["up_curve_on_down_bias_g"] > 30
    regularized = {b for b, s in sectors.items() if s["isotonic_used_curve"]["adjusted_masses_kg"]}
    assert {"2", "3", "8"} <= regularized


# ----------------------------------------------------------------------------
# Ensaios por célula: marcação, exportação e generalização
# ----------------------------------------------------------------------------

def test_experiment_marks_target_cell_and_exports_plateaus(bench: _Bench) -> None:
    import exportar_ensaios_celula as exp

    bench.tare()
    rec = bench.worker.recorder
    rec.start({"description": "célula"})
    with pytest.raises(ValueError):
        rec.mark(target_cell="40x10")
    for cycle, mass, direction, value in [(1, 50.0, "UP", 20.0), (1, 100.0, "UP", 40.0), (2, 50.0, "UP", 22.0)]:
        status = rec.mark(target_cell="12x34", cycle_id=cycle, known_mass_g=mass, loading_direction=direction)
        assert status["target_cell"] == "12x34" and status["target_block"] == 2
        frame = np.full(SHAPE, 2.0)
        frame[12, 34] += value
        bench.feed(frame, 0.5)
    session = Path(rec.stop()["directory"])

    with (session / "frames.csv").open(encoding="utf-8") as fh:
        first = next(csv.DictReader(fh))
    assert first["target_cell"] == "12x34" and first["target_block"] == "2"

    rows = exp.plateaus(list(exp._frames(session)), tail_fraction=0.5)
    assert [(r["cycle_id"], r["known_mass_g"]) for r in rows] == [(1, 50.0), (1, 100.0), (2, 50.0)]
    assert rows[1]["corrected_mean"] == pytest.approx(40.0)
    assert rows[1]["baseline_raw"] == pytest.approx(2.0) and rows[1]["block_id"] == 2


def test_generalization_separates_gain_from_shape() -> None:
    import comparar_generalizacao_celulas as gen

    masses = [0.025, 0.05, 0.1, 0.15, 0.2]
    plateaus = ([gen.Plateau("12x34", 2, None, "UP", m, 400 * m) for m in masses]
                + [gen.Plateau("12x37", 2, None, "UP", m, 300 * m) for m in masses]   # mesma forma, outro ganho
                + [gen.Plateau("9x53", 1, None, "UP", m, 350 * m) for m in masses])
    points = gen.cell_points(plateaus)
    block_of = {p.cell: p.block for p in plateaus}
    res = gen.evaluate_cell("12x37", points, block_of)
    # própria: só erra o último patamar, que fica fora da curva ao ser deixado de fora (50 g)
    assert res["own"]["rmse_g"] == pytest.approx(50.0 / np.sqrt(5), abs=0.5)
    assert res["B_sector"]["rmse_g"] > 20.0 and res["B_sector"]["bias_g"] < 0
    assert res["C_sector_gain"]["rmse_g"] == pytest.approx(0.0, abs=0.5)
    assert res["gain_g_ij"] == pytest.approx(4 / 3, rel=1e-3)
    lonely = gen.evaluate_cell("9x53", points, block_of)
    assert lonely["B_sector"] is None and lonely["A_global"] is not None


# ----------------------------------------------------------------------------
# Repetição e posição
# ----------------------------------------------------------------------------

def _pt(mass: float, pos: str | None, est: float) -> dict:
    return {"known_mass_kg": mass, "position": pos, "current_estimate_kg": est}


def test_position_report_tells_spatial_from_gain_error() -> None:
    spatial = position_report([_pt(20, "centro", 23), _pt(20, "cabeceira", 17), _pt(20, "peseira", 20)])
    assert spatial[0]["diagnosis"].startswith("espacial")
    gain = position_report([_pt(20, "centro", 22.0), _pt(20, "centro", 22.2), _pt(20, "cabeceira", 21.9),
                            _pt(20, "peseira", 22.1)])
    assert gain[0]["diagnosis"].startswith("ganho global")
    assert gain[0]["repeatability_std_kg"] == pytest.approx(0.141, abs=1e-3)
    assert position_report([_pt(20, "centro", 21), _pt(30, "centro", 31)]) == []   # sem repetição


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

def test_diagnostics_api(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app.state, "acquisition_worker", bench.worker, raising=False)
    client = TestClient(app)
    bench.tare()
    bench.feed(_residual_frame(), 1.0)

    residual = client.get("/api/v1/diagnostics/residual").json()
    assert residual["n_frames"] > 0
    assert client.post("/api/v1/diagnostics/residual/reset").json()["n_frames"] == 0

    load = np.full(SHAPE, 2.0)
    load[BLOCK_REGIONS[1]] += 30.0
    bench.feed(load, 1.0)
    cells = client.get("/api/v1/diagnostics/cells", params={"limit": 5}).json()
    assert len(cells["cells"]) == 5 and cells["totals"]["n_active_cells"] >= 256
    assert {"raw", "baseline", "corrected_raw", "local_mass_g", "dm_d_raw_g", "situation", "flags"} <= set(cells["cells"][0])

    audit = client.get("/api/v1/diagnostics/calibration").json()
    assert any("Setor 5: 25 g sem resposta" in a for a in audit["alerts"])

    extraction = client.get("/api/v1/calibration/extraction").json()
    assert "totals" in extraction and len(extraction["dm_d_raw_g"]) == 32

    bench.feed(load, 4.0)
    r = client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 70, "position": "lado"})
    assert r.status_code == 422
    r = client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 70, "position": "Centro"})
    assert r.status_code == 200
    state = r.json()
    assert state["points"][0]["position"] == "centro" and state["provisional"] is True
    assert state["position_report"] == []
