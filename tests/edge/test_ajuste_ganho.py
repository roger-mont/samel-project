"""Testes do ajuste de ganho α_s pela página /calibracao: captura estável, pontos, cálculo e gravação."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_EDGE = Path(__file__).resolve().parents[2] / "apps" / "edge-service"
if str(_EDGE / "src") not in sys.path:
    sys.path.insert(0, str(_EDGE / "src"))

from fastapi.testclient import TestClient

from api.server import app
from core.settings import CalibrationParams
from database.db_local import LocalDatabase
from hardware.serial_reader import BaseFrameReader
from math_engine.gain_fit import curves_sha1, fit_gains, write_gains
from math_engine.mass_models import BlockCurveModel, ModelRegistry
from math_engine.spatial_baseline import SpatialBaseline
from storage.calibration_store import BLOCK_REGIONS
from storage.experiment_recorder import ExperimentRecorder
from storage.gain_points import GainPointStore
from workers.acquisition_worker import AcquisitionWorker

SHAPE = (32, 64)
DT = 0.05


def _write_curves(path: Path) -> None:
    """1 kg a cada 100 RAW em todos os setores."""
    blocks = {
        str(bid): {"kind": "linear_interp", "params": {"raw": [0.0, 100.0], "mass_kg": [0.0, 1.0]},
                   "raw_min_valid": 1.0, "raw_max_valid": 100.0, "mass_min_valid": 0.0,
                   "mass_max_valid": 1.0, "low_sensitivity_raw": None}
        for bid in BLOCK_REGIONS
    }
    path.write_text(json.dumps({"calibration_version": "MC-TEST-GANHO", "blocks": blocks}), encoding="utf-8")


class _NoReader(BaseFrameReader):
    def is_connected(self) -> bool:
        return True

    def read_frame(self) -> np.ndarray:
        return np.zeros(SHAPE)

    def close(self) -> None:
        pass


class _Bench:
    """Worker sem thread: os frames entram por process_frame com relógio simulado."""

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


def _load(cells: int, raw: float = 30.0) -> np.ndarray:
    """Manta vazia (RAW 2) com `cells` células do setor 2 em 2 + raw."""
    frame = np.full(SHAPE, 2.0)
    rs, cs = BLOCK_REGIONS[2]
    block = frame[rs, cs].reshape(-1)
    block[:cells] += raw
    frame[rs, cs] = block.reshape(16, 16)
    return frame


@pytest.fixture()
def bench(tmp_path: Path) -> _Bench:
    return _Bench(tmp_path)


# ----------------------------------------------------------------------------
# Cálculo e gravação
# ----------------------------------------------------------------------------

def test_write_gains_records_fit_and_reset_returns_to_one(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_curves(path)
    write_gains(path, {b: 1.2 for b in BLOCK_REGIONS}, {"mode": "global"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["aggregation"]["calibrated"] is True
    assert data["aggregation"]["fit"]["curves_sha1"] == curves_sha1(data)
    model = BlockCurveModel(path)
    assert model.gains_calibrated and model.gains == {b: 1.2 for b in BLOCK_REGIONS}

    write_gains(path, None, None)
    model = BlockCurveModel(path)
    assert not model.gains_calibrated and set(model.gains.values()) == {1.0}
    with pytest.raises(ValueError):
        write_gains(path, {1: 0.0}, None)


def test_global_fit_is_least_squares_on_the_local_sum() -> None:
    L = np.array([[6.0, 6.0, 0, 0, 0, 0, 0, 0], [0, 0, 12.0, 12.0, 0, 0, 0, 0]])
    M = np.array([10.0, 20.0])
    assert fit_gains(L, M, "global", 1.0) == pytest.approx(np.full(8, 10.0 / 12.0))


def test_gain_points_persist_between_instances(tmp_path: Path) -> None:
    store = GainPointStore(tmp_path / "pts.json")
    a = store.add(10.0, np.ones(SHAPE), {"note": "centro"})
    store.add(20.0, np.ones(SHAPE) * 2, {})
    again = GainPointStore(tmp_path / "pts.json")
    assert [p["known_mass_kg"] for p in again.list()] == [10.0, 20.0]
    assert again.corrected(again.list()[1])[0, 0] == 2.0
    assert again.delete(a["id"]) and not again.delete(a["id"])
    assert again.clear() == 1 and GainPointStore(tmp_path / "pts.json").list() == []


# ----------------------------------------------------------------------------
# Captura estável no worker
# ----------------------------------------------------------------------------

def test_stable_capture_waits_for_stable_state(bench: _Bench) -> None:
    bench.tare()
    bench.feed(_load(40), 2.5)          # carga presente, ainda estabilizando
    assert bench.worker.stable_capture() is None
    bench.feed(_load(40), 2.5)          # 1 s de critério atendido → ESTAVEL
    capture = bench.worker.stable_capture()
    assert capture is not None
    mean, frames, seconds = capture
    assert frames >= 3 and seconds <= 5.0
    assert float(np.sum(mean)) == pytest.approx(40 * 30.0, rel=1e-3)
    bench.feed(np.full(SHAPE, 2.0), 1.0)  # carga retirada
    assert bench.worker.stable_capture() is None


# ----------------------------------------------------------------------------
# API do painel
# ----------------------------------------------------------------------------

def test_gain_panel_flow(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app.state, "acquisition_worker", bench.worker, raising=False)
    client = TestClient(app)  # sem "with": não sobe o lifespan (worker real e sync)

    r = client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 10})
    assert r.status_code == 409 and "tara" in r.json()["detail"]

    bench.tare()
    bench.feed(_load(40), 1.0)
    r = client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 10})
    assert r.status_code == 409 and "estável" in r.json()["detail"]

    bench.feed(_load(40), 4.0)                    # 40 células × 0,3 kg = 12 kg pelas curvas
    r = client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 10, "note": "setor 2"})
    assert r.status_code == 200
    state = r.json()
    assert state["points"][0]["local_total_kg"] == pytest.approx(12.0, rel=1e-3)
    assert state["points"][0]["note"] == "setor 2"

    r = client.post("/api/v1/calibration/gain/fit", json={"mode": "global"})
    assert r.status_code == 400                   # um ponto só

    bench.feed(np.full(SHAPE, 2.0), 1.0)
    bench.feed(_load(80), 5.0)                    # 24 kg pelas curvas
    client.post("/api/v1/calibration/gain/points", json={"known_mass_kg": 20})

    r = client.post("/api/v1/calibration/gain/fit", json={"mode": "global"})
    assert r.status_code == 200
    preview = r.json()
    assert preview["alpha"]["2"] == pytest.approx(10 / 12, rel=1e-3)
    assert preview["report"]["fitted"]["rmse_kg"] == pytest.approx(0.0, abs=1e-3)
    assert not BlockCurveModel(bench.curves).gains_calibrated   # calcular não grava

    r = client.post("/api/v1/calibration/gain/apply", json={"mode": "global"})
    assert r.status_code == 200
    state = r.json()
    assert state["calibrated"] and state["fit_matches_curves"]
    assert state["points"][0]["current_estimate_kg"] == pytest.approx(10.0, rel=1e-3)
    assert bench.worker.models.active().gains[2] == pytest.approx(10 / 12, rel=1e-3)

    r = client.post("/api/v1/calibration/gain/reset")
    assert not r.json()["calibrated"] and set(r.json()["gains"].values()) == {1.0}
    assert len(r.json()["points"]) == 2               # voltar para 1 não apaga os pontos

    first = r.json()["points"][0]["id"]
    assert client.delete(f"/api/v1/calibration/gain/points/{first}").status_code == 200
    assert client.delete(f"/api/v1/calibration/gain/points/{first}").status_code == 404
    assert client.delete("/api/v1/calibration/gain/points").json()["points"] == []
