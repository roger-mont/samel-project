"""Testes da cadeia baseline → estados → modelo → validação → registro experimental."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from core.settings import CalibrationParams
from database.db_local import LocalDatabase
from hardware.serial_reader import FakeSerialReader
from math_engine.mass_models import (
    BlockCurveModel, MassEstimate, MassModel, ModelRegistry, WindowFrame, build_curve,
)
from math_engine.quality import QualityFlags
from math_engine.reading_state import ReadingState, ReadingStateMachine
from math_engine.spatial_baseline import SpatialBaseline
from storage.calibration_store import BLOCK_REGIONS
from storage.experiment_recorder import ExperimentRecorder, iter_cell_records, load_raw_frames
from workers.acquisition_worker import AcquisitionWorker

DT = 0.025  # 40 Hz
SHAPE = (32, 64)


def _params(**overrides: float) -> CalibrationParams:
    p = CalibrationParams()
    p.apply_overrides({"min_stable_time_s": 2.0, "t_tare_s": 1.0, "auto_rezero_empty_s": 1.0, **overrides})
    return p


def _load(total_per_cell: float, rows: slice = slice(8, 24), cols: slice = slice(16, 48)) -> np.ndarray:
    m = np.zeros(SHAPE)
    m[rows, cols] = total_per_cell
    return m


# ----------------------------------------------------------------------------
# Ordem física dos setores
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("block_id", range(1, 9))
def test_hid_block_id_lands_in_same_numbered_sector(block_id: int) -> None:
    """[1][8] / [2][7] / [3][6] / [4][5], cabo no 1: o block_id do pacote é o setor."""
    from hardware.serial_reader import HidFrameReader

    reader = object.__new__(HidFrameReader)
    reader._max_seen = 0
    matrix = np.zeros(SHAPE)
    packet = bytes([block_id]) + bytes([x for xy in [(1, 1), (16, 16), (8, 5)] for x in (*xy, 77)])
    reader._parse_hid_packet(packet, matrix)

    inside = np.zeros(SHAPE, dtype=bool)
    inside[BLOCK_REGIONS[block_id]] = True
    assert np.count_nonzero(matrix[inside]) == 3
    assert np.count_nonzero(matrix[~inside]) == 0


def test_sector_layout_matches_matrix_geometry() -> None:
    from storage.calibration_store import SECTOR_LAYOUT

    for depth, (right, left) in enumerate(SECTOR_LAYOUT):
        r_rows, r_cols = BLOCK_REGIONS[right]
        l_rows, l_cols = BLOCK_REGIONS[left]
        assert (r_rows.start, l_rows.start) == (0, 16)          # direito em cima, esquerdo embaixo
        assert r_cols == l_cols                                   # mesma faixa longitudinal
        assert r_cols.start == 48 - 16 * depth                    # cabeceira nas colunas altas


# ----------------------------------------------------------------------------
# Baseline espacial
# ----------------------------------------------------------------------------

def test_tare_averages_frames_over_t_tare(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = SpatialBaseline(tmp_path / "baseline.json")
    base.start_tare(now=0.0, duration_s=1.0)
    t = 0.0
    record = None
    while record is None:
        frame = 6.0 + rng.normal(0, 1.0, SHAPE)
        record = base.feed(frame, t, configuration_id="CFG-x", model_version="m1")
        t += DT

    assert record.n_frames >= 40
    assert record.baseline_mean == pytest.approx(6.0, abs=0.1)
    assert record.baseline_std == pytest.approx(1.0, abs=0.15)
    assert record.configuration_id == "CFG-x" and record.model_version == "m1"
    assert base.is_baseline_ok(max_mean_std_raw=3.0)

    corrected = base.correct(np.full(SHAPE, 4.0))
    assert float(corrected.min()) == 0.0  # x < 0 vira 0

    reloaded = SpatialBaseline(tmp_path / "baseline.json")
    assert reloaded.record.tare_id == record.tare_id
    np.testing.assert_allclose(reloaded.mean_matrix, base.mean_matrix, atol=1e-3)


# ----------------------------------------------------------------------------
# Máquina de estados
# ----------------------------------------------------------------------------

def _run(machine: ReadingStateMachine, frames, t0: float = 0.0) -> tuple[list[ReadingState], float]:
    states = []
    t = t0
    for f in frames:
        states.append(machine.update(f, t))
        t += DT
    return states, t


def test_empty_bed_with_typical_noise_is_no_load() -> None:
    """Manta vazia soma ~125 RAW (taras de 03 e 04/09): abaixo do limiar padrão de 150."""
    machine = ReadingStateMachine(_params())
    rng = np.random.default_rng(1)
    flat = rng.choice(SHAPE[0] * SHAPE[1], size=60, replace=False)

    def empty_frame() -> np.ndarray:
        m = np.zeros(SHAPE[0] * SHAPE[1])
        m[flat] = rng.integers(0, 5, size=flat.size)  # ~60 células acesas, ~2 RAW cada
        return m.reshape(SHAPE)

    frames = [empty_frame() for _ in range(80)]
    assert 90 < np.mean([f.sum() for f in frames]) < 150
    states, _ = _run(machine, frames)
    assert set(states) == {ReadingState.SEM_CARGA}


def test_constant_load_becomes_stable_and_records_stabilization_time() -> None:
    machine = ReadingStateMachine(_params())
    load = _load(40.0)
    states, t = _run(machine, [load] * int(6.0 / DT))
    assert states[0] == ReadingState.CARGA_DETECTADA
    assert ReadingState.ESTABILIZANDO in states
    assert states[-1] == ReadingState.ESTAVEL
    ts = machine.metrics.stabilization_time_s
    # janela (1 s) + Δt (1 s) de aquecimento + 2 s de critério contínuo
    assert ts == pytest.approx(4.0, abs=0.1)


def test_creep_ramp_never_becomes_stable() -> None:
    """Soma subindo 2%/s (creep) não atende D/μ < 1%."""
    machine = ReadingStateMachine(_params())
    frames = [_load(40.0 * (1 + 0.02 * i * DT)) for i in range(int(8.0 / DT))]
    states, _ = _run(machine, frames)
    assert ReadingState.ESTAVEL not in states


def test_redistribution_enters_movement_then_restabilizes() -> None:
    machine = ReadingStateMachine(_params())
    left = _load(40.0, cols=slice(0, 32))
    right = _load(40.0, cols=slice(32, 64))   # mesma soma, outro lugar
    states, t = _run(machine, [left] * int(5.0 / DT))
    assert states[-1] == ReadingState.ESTAVEL

    states, t = _run(machine, [right] * int(6.0 / DT), t0=t)
    assert ReadingState.MOVIMENTO in states
    first_move = states.index(ReadingState.MOVIMENTO)
    assert ReadingState.ESTABILIZANDO in states[first_move:]
    assert states[-1] == ReadingState.ESTAVEL


def test_set_validated_only_promotes_stable_reading() -> None:
    machine = ReadingStateMachine(_params())
    _run(machine, [_load(40.0)] * 20)
    assert machine.set_validated(True) == ReadingState.CARGA_DETECTADA
    _run(machine, [_load(40.0)] * int(6.0 / DT), t0=1.0)
    assert machine.set_validated(True) == ReadingState.MEDICAO_VALIDA
    assert machine.set_validated(False) == ReadingState.ESTAVEL


# ----------------------------------------------------------------------------
# Qualidade e modelos
# ----------------------------------------------------------------------------

def test_weight_valid_requires_every_condition() -> None:
    ok = dict(
        tare_ok=True, baseline_ok=True, signal_stable=True, movement=False, saturation=False,
        missing_cells=False, out_of_calibration=False, low_sensitivity=True, model_valid=True,
    )
    assert QualityFlags(**ok).weight_valid  # LOW_SENSITIVITY é só informativo
    for key, bad in [("tare_ok", False), ("signal_stable", False), ("movement", True),
                     ("saturation", True), ("out_of_calibration", True), ("model_valid", False)]:
        assert not QualityFlags(**{**ok, key: bad}).weight_valid, key


def test_pchip_is_monotone_and_hits_knots() -> None:
    raw = [0.0, 13.6, 44.3, 56.7, 68.9, 75.0]
    mass = [0.0, 0.025, 0.05, 0.1, 0.2, 0.275]
    f = build_curve("pchip", {"raw": raw, "mass_kg": mass})
    np.testing.assert_allclose(f(np.array(raw)), mass, atol=1e-12)
    xs = np.linspace(0, 75, 500)
    assert np.all(np.diff(f(xs)) >= -1e-12)
    with pytest.raises(ValueError):
        build_curve("pchip", {"raw": [0, 10, 5], "mass_kg": [0, 1, 2]})


def _write_block_curves(path: Path, raw_max: float = 100.0) -> None:
    blocks = {
        str(bid): {
            "kind": "linear_interp",
            "params": {"raw": [0.0, 100.0], "mass_kg": [0.0, 1.0]},
            "raw_min_valid": 1.0,
            "raw_max_valid": raw_max,
            "mass_min_valid": 0.0,
            "mass_max_valid": 1.0,
            "low_sensitivity_raw": None,
        }
        for bid in BLOCK_REGIONS
    }
    path.write_text(json.dumps({"calibration_version": "MC-TEST-001", "blocks": blocks}), encoding="utf-8")


def test_block_curve_model_sums_cell_masses(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_block_curves(path)
    model = BlockCurveModel(path)
    assert model.is_ready and model.calibration_version == "MC-TEST-001"

    corrected = np.zeros(SHAPE)
    corrected[0:2, 0:5] = 50.0   # 10 células × 0,5 kg
    corrected[20, 60] = 0.5      # abaixo de raw_min_valid: não conta
    est = model.estimate(WindowFrame(raw_mean=corrected, corrected_mean=corrected))
    assert est.mass_kg == pytest.approx(5.0)
    assert not est.out_of_calibration

    corrected[5, 5] = 150.0
    assert model.estimate(WindowFrame(corrected, corrected)).out_of_calibration


def test_block_curve_breakdown_matches_estimate(tmp_path: Path) -> None:
    path = tmp_path / "block_curves.json"
    _write_block_curves(path)
    model = BlockCurveModel(path)
    corrected = np.zeros(SHAPE)
    corrected[BLOCK_REGIONS[1]] = 20.0          # 256 células × 0,2 kg
    corrected[20, 5] = 150.0                     # setor 5, acima de raw_max_valid=100

    data = model.breakdown(corrected, active_blocks=[1, 2, 3, 4, 5, 6, 7, 8])
    by = {s["sector"]: s for s in data["sectors"]}
    assert by[1]["mass_kg"] == pytest.approx(51.2)
    assert by[5]["cells_out_of_calibration"] == 1
    assert data["out_of_calibration"][20][5] == 1
    est = model.estimate(WindowFrame(corrected, corrected))
    assert data["total_mass_kg"] == pytest.approx(est.mass_kg, abs=1e-3)

    only_1 = model.breakdown(corrected, active_blocks=[1])
    assert only_1["total_mass_kg"] == pytest.approx(51.2)

    samples = model.curve_samples(2, n=11)
    assert samples[0] == [0.0, 0.0] and len(samples) == 11


# ----------------------------------------------------------------------------
# Worker de ponta a ponta (sem thread)
# ----------------------------------------------------------------------------

class SumModel(MassModel):
    """Modelo de teste: 1 kg a cada 100 RAW corrigidos."""

    kind = "test_sum"
    is_ready = True
    model_version = "TEST-1"
    calibration_version = "MC-TEST-002"

    def estimate(self, frame: WindowFrame) -> MassEstimate:
        return self._result(float(frame.corrected_mean.sum()) / 100.0, True)

    def load(self) -> bool:
        return True


def _worker(tmp_path: Path, **params: float) -> AcquisitionWorker:
    return AcquisitionWorker(
        db=LocalDatabase(db_path=str(tmp_path / "edge.db")),
        reader=FakeSerialReader(),
        baseline=SpatialBaseline(tmp_path / "baseline.json"),
        models=ModelRegistry({"test": SumModel()}, selection="test"),
        recorder=ExperimentRecorder(tmp_path / "experiments"),
        params=_params(**params),
    )


def test_worker_publishes_weight_only_when_valid(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    empty = np.full(SHAPE, 6.0)   # 12.288 RAW de baseline
    t = 0.0

    # Sem tara: a soma crua da manta vazia parece carga e nada é validado
    live = worker.process_frame(empty, t)
    assert live["weight_kg"] is None
    assert live["measurement_quality"]["TARE_OK"] is False

    worker.trigger_tare()
    for _ in range(int(1.5 / DT)):
        t += DT
        live = worker.process_frame(empty, t)
    assert live["measurement_quality"]["TARE_OK"] is True
    assert live["weight_status"] == "NO_LOAD"

    loaded = empty + _load(40.0)      # 512 células × 40 RAW = 20.480 → 204,8 kg no modelo de teste
    statuses = []
    for _ in range(int(6.0 / DT)):
        t += DT
        live = worker.process_frame(loaded, t)
        statuses.append(live["weight_status"])
        if live["weight_status"] != "VALID":
            assert live["weight_kg"] is None
    assert "STABILIZING" in statuses
    assert live["weight_status"] == "VALID"
    assert live["weight_kg"] == pytest.approx(204.8)
    assert live["stable"] is True and live["movement"] is False
    assert live["calibration_version"] == "MC-TEST-002"
    assert live["stabilization_time_s"] == pytest.approx(4.0, abs=0.1)

    # Movimento: peso some, último válido continua disponível
    moved = empty + _load(40.0, cols=slice(32, 64))
    t += DT
    live = worker.process_frame(moved, t)
    t += DT
    live = worker.process_frame(moved, t)
    assert live["weight_status"] == "MOVEMENT"
    assert live["weight_kg"] is None
    assert live["last_valid_weight_kg"] == pytest.approx(204.8)

    # Maca vazia (a média da janela precisa cair): último válido é descartado
    for _ in range(int(1.5 / DT)):
        t += DT
        live = worker.process_frame(empty, t)
    assert live["weight_status"] == "NO_LOAD"
    assert live["last_valid_weight_kg"] is None


def test_active_blocks_restrict_measurement(tmp_path: Path) -> None:
    """Com só o bloco 2 ativo, carga em outros blocos não entra na medição."""
    worker = AcquisitionWorker(
        db=LocalDatabase(db_path=str(tmp_path / "edge.db")),
        reader=FakeSerialReader(),
        baseline=SpatialBaseline(tmp_path / "baseline.json"),
        models=ModelRegistry({"test": SumModel()}, selection="test"),
        recorder=ExperimentRecorder(tmp_path / "experiments"),
        params=_params(),
        active_blocks=[2],
    )
    worker.trigger_tare()
    t = 0.0
    for _ in range(int(1.5 / DT)):
        t += DT
        worker.process_frame(np.zeros(SHAPE), t)

    frame = np.zeros(SHAPE)
    frame[BLOCK_REGIONS[5]] = 50.0      # fora do bloco 2: ignorado
    frame[2:4, 40:45] = 30.0            # bloco 2 (linhas 0-15, colunas 32-47): 10 células × 30
    for _ in range(int(6.0 / DT)):
        t += DT
        live = worker.process_frame(frame, t)
    assert live["active_blocks"] == [2]
    assert live["total_raw"] == pytest.approx(300.0)
    assert live["weight_kg"] == pytest.approx(3.0)
    assert len(live["heatmap"]) == 32


def test_saturated_cell_blocks_validation(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker.trigger_tare()
    t = 0.0
    for _ in range(int(1.5 / DT)):
        t += DT
        worker.process_frame(np.zeros(SHAPE), t)
    loaded = _load(40.0)
    loaded[10, 20] = 255.0
    for _ in range(int(6.0 / DT)):
        t += DT
        live = worker.process_frame(loaded, t)
    assert live["state_id"] == "ESTAVEL"
    assert live["measurement_quality"]["SATURATION"] is True
    assert live["weight_kg"] is None
    assert live["saturated_cells"] == 1


def test_auto_tare_runs_once_per_empty_period(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    t = 0.0
    for _ in range(int(3.0 / DT)):
        t += DT
        worker.process_frame(np.full(SHAPE, 0.05), t)
    first = worker.baseline.record
    assert first is not None and first.automatic
    for _ in range(int(3.0 / DT)):
        t += DT
        worker.process_frame(np.full(SHAPE, 0.05), t)
    assert worker.baseline.record.tare_id == first.tare_id


def test_experiment_recorder_keeps_raw_and_rebuilds_cell_records(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker.trigger_tare()
    t = 0.0
    for _ in range(int(1.5 / DT)):
        t += DT
        worker.process_frame(np.full(SHAPE, 6.0), t)

    worker.recorder.start({"description": "teste"})
    worker.recorder.mark(cycle_id=1, known_mass_g=200.0, loading_direction="up")
    frame = np.full(SHAPE, 6.0)
    frame[3, 7] = 90.0
    for _ in range(10):
        t += DT
        worker.process_frame(frame, t)
    result = worker.recorder.stop()

    session_dir = Path(result["directory"])
    raw = load_raw_frames(session_dir)
    assert raw.shape == (10, 32, 64)
    assert int(raw[0, 3, 7]) == 90
    assert len(list((session_dir / "baselines").glob("*.npz"))) == 1

    records = list(iter_cell_records(session_dir, cells=[(3, 7)]))
    assert len(records) == 10
    rec = records[0]
    assert rec["loading_direction"] == "UP"
    assert float(rec["known_mass_g"]) == 200.0
    assert rec["raw_value"] == 90.0
    assert rec["baseline_raw"] == pytest.approx(6.0)
    assert rec["corrected_raw"] == pytest.approx(84.0)

    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["frames"] == 10
