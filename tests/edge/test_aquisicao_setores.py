"""Parecer de 20/10/2026: telemetria por setor, STALE_SECTOR, tara bloqueada, retorno ao
baseline, log de diagnóstico e CSVs de calibração duplicados."""
from __future__ import annotations

import csv
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
from hardware.acquisition_stats import SECTOR_CELLS, AcquisitionSample, AcquisitionStats
from hardware.serial_reader import BaseFrameReader, HidFrameReader
from math_engine.calibration_audit import audit_calibration
from math_engine.mass_models import MassEstimate, MassModel, ModelRegistry, WindowFrame
from math_engine.spatial_baseline import SpatialBaseline
from storage.calibration_store import BLOCK_REGIONS
from storage.diagnostic_log import FIELDS
from storage.experiment_recorder import ExperimentRecorder
from workers.acquisition_worker import AcquisitionWorker

DT = 0.025
SHAPE = (32, 64)


class _NoReader(BaseFrameReader):
    def read_frame(self) -> np.ndarray:
        return np.zeros(SHAPE)

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


class _SumModel(MassModel):
    """1 kg a cada 100 RAW corrigidos."""

    kind = "test_sum"
    is_ready = True
    model_version = "TEST-1"
    calibration_version = "MC-TEST-AQ"

    def estimate(self, frame: WindowFrame) -> MassEstimate:
        return self._result(float(frame.corrected_mean.sum()) / 100.0, True)

    def load(self) -> bool:
        return True


def _sample(now: float, ages: dict[int, float | None] | None = None) -> AcquisitionSample:
    """Telemetria sintética: todos os setores atualizados há 10 ms, salvo os de `ages`."""
    ages = ages or {}
    sectors = {}
    cell_age = np.full(SHAPE, 0.01)
    for s in range(1, 9):
        age = ages.get(s, 0.01)
        sectors[str(s)] = {"age_s": age, "packets": 100, "packets_per_s": 500.0, "incomplete_sweeps": 0}
        cell_age[BLOCK_REGIONS[s]] = np.inf if age is None else age
    return AcquisitionSample(summary={"source": "teste", "packets_per_s": 4000.0, "read_errors": 0,
                                      "sectors": sectors}, cell_age_s=cell_age)


class _Bench:
    def __init__(self, tmp: Path, diagnostics: bool = False, **params: float) -> None:
        p = CalibrationParams()
        p.apply_overrides({"min_stable_time_s": 1.0, "t_tare_s": 0.5, "tare_empty_min_s": 1.0,
                           "auto_rezero_empty_s": 1000.0, **params})
        self.worker = AcquisitionWorker(
            db=LocalDatabase(db_path=str(tmp / "edge.db")),
            reader=_NoReader(),
            baseline=SpatialBaseline(tmp / "baseline.json"),
            models=ModelRegistry({"test": _SumModel()}, selection="test"),
            recorder=ExperimentRecorder(tmp / "exp"),
            params=p,
            diagnostics_dir=(tmp / "diagnostics") if diagnostics else None,
        )
        self.now = 100.0
        self.ages: dict[int, float | None] = {}

    def feed(self, frame: np.ndarray, seconds: float) -> dict:
        live = {}
        for _ in range(int(round(seconds / DT))):
            self.now += DT
            live = self.worker.process_frame(frame, self.now, True, _sample(self.now, self.ages))
        return live

    def first_tare(self) -> None:
        assert self.worker.trigger_tare()["accepted"]
        self.feed(np.full(SHAPE, 2.0), 1.0)
        assert self.worker.baseline.has_baseline


def _load(raw: float = 30.0) -> np.ndarray:
    frame = np.full(SHAPE, 2.0)
    frame[BLOCK_REGIONS[2]] += raw
    frame[BLOCK_REGIONS[7]] += raw
    return frame


@pytest.fixture()
def bench(tmp_path: Path) -> _Bench:
    return _Bench(tmp_path)


# ----------------------------------------------------------------------------
# Telemetria do leitor
# ----------------------------------------------------------------------------

def test_stats_counts_complete_update_and_age() -> None:
    stats = AcquisitionStats("teste")
    cells = list(range(SECTOR_CELLS))
    for i in range(0, SECTOR_CELLS, 21):
        stats.on_packet(3, cells[i:i + 21], [], [], 10.0)
    sample = stats.sample(10.5)
    sec = sample.summary["sectors"]["3"]
    assert sec["complete_updates"] == 1
    assert sec["age_s"] == pytest.approx(0.5)
    assert sec["incomplete_sweeps"] == 0
    assert sample.summary["sectors"]["1"]["age_s"] is None       # setor que nunca chegou


def test_stats_flags_incomplete_sweep_when_cell_repeats() -> None:
    """Sem número de sequência: célula repetida antes de cobrir o setor indica pacote perdido."""
    stats = AcquisitionStats("teste")
    stats.on_packet(2, list(range(0, 100)), [], [], 1.0)
    stats.on_packet(2, list(range(0, 21)), [], [], 1.1)         # recomeçou sem as células 100-255
    sec = stats.sample(1.2).summary["sectors"]["2"]
    assert sec["complete_updates"] == 0
    assert sec["incomplete_sweeps"] == 1
    assert sec["cells_missed"] == SECTOR_CELLS - 100


def test_hid_packet_reports_sector_and_local_cells() -> None:
    reader = object.__new__(HidFrameReader)
    reader._max_seen = 0
    matrix = np.zeros(SHAPE)
    packet = bytes([7]) + bytes([1, 1, 50, 16, 16, 60, 0, 0, 0])
    sector, local, rows, cols = reader._parse_hid_packet(packet, matrix)
    assert sector == 7
    assert local == [0, 255]
    assert all(BLOCK_REGIONS[7][0].start <= r < BLOCK_REGIONS[7][0].stop for r in rows)
    assert np.count_nonzero(matrix) == 2


# ----------------------------------------------------------------------------
# STALE_SECTOR
# ----------------------------------------------------------------------------

def test_stale_sector_blocks_validated_weight(bench: _Bench) -> None:
    bench.first_tare()
    live = bench.feed(_load(), 6.0)
    assert live["weight_status"] == "VALID"
    assert live["measurement_quality"]["STALE_SECTOR"] is False

    bench.ages = {2: 5.0}                                         # setor 2 parado há 5 s
    live = bench.feed(_load(), 0.1)
    assert live["measurement_quality"]["STALE_SECTOR"] is True
    assert live["stale_sectors"] == [2]
    assert live["weight_kg"] is None
    assert live["weight_status"] != "VALID"
    assert live["estimated_mass_kg"] is not None                  # peso atual continua visível
    assert live["last_valid_weight_kg"] is not None               # e o último válido, separado


def test_stale_sector_can_be_disabled(tmp_path: Path) -> None:
    b = _Bench(tmp_path, stale_sector_s=0.0)
    b.ages = {4: None}
    b.first_tare()
    live = b.feed(_load(), 6.0)
    assert live["measurement_quality"]["STALE_SECTOR"] is False
    assert live["weight_status"] == "VALID"


# ----------------------------------------------------------------------------
# Tara bloqueada
# ----------------------------------------------------------------------------

def test_tare_rejected_with_load_and_records_reason(bench: _Bench) -> None:
    bench.first_tare()
    bench.feed(_load(), 1.0)
    result = bench.worker.trigger_tare()
    assert result["accepted"] is False
    assert "carga" in result["reason"]
    assert bench.worker.tare_events[-1]["event"] == "rejeitada"
    assert bench.worker.tare_status()["collecting"] is False


def test_tare_needs_continuous_empty_time(tmp_path: Path) -> None:
    bench = _Bench(tmp_path, tare_empty_min_s=3.0)
    bench.first_tare()
    bench.feed(_load(), 1.0)
    bench.feed(np.full(SHAPE, 2.0), 1.5)                          # vazia há pouco
    assert bench.worker.get_live_snapshot()["state_id"] == "SEM_CARGA"
    result = bench.worker.trigger_tare()
    assert result["accepted"] is False and "maca vazia há só" in result["reason"]
    bench.feed(np.full(SHAPE, 2.0), 3.0)
    old = bench.worker.baseline.record.tare_id
    assert bench.worker.trigger_tare()["accepted"] is True
    bench.feed(np.full(SHAPE, 2.0), 1.0)
    assert bench.worker.baseline.record.tare_id != old
    assert bench.worker.tare_events[-1]["event"] == "concluida"
    assert bench.worker.tare_events[-1]["baseline_sum"] == pytest.approx(2.0 * 2048)


def test_tare_rejected_with_stale_sector(bench: _Bench) -> None:
    bench.first_tare()
    bench.feed(np.full(SHAPE, 2.0), 2.0)
    bench.ages = {6: 3.0}
    bench.feed(np.full(SHAPE, 2.0), 0.1)
    result = bench.worker.trigger_tare()
    assert result["accepted"] is False and "6" in result["reason"]


def test_tare_cancelled_when_load_appears_during_collection(bench: _Bench) -> None:
    bench.first_tare()
    old = bench.worker.baseline.record.tare_id
    bench.feed(np.full(SHAPE, 2.0), 2.0)
    assert bench.worker.trigger_tare()["accepted"] is True
    bench.feed(np.full(SHAPE, 2.0), 0.1)
    bench.feed(_load(), 0.3)
    assert bench.worker.baseline.record.tare_id == old
    assert bench.worker.tare_events[-1]["event"] == "cancelada"
    assert not bench.worker.baseline.is_collecting


# ----------------------------------------------------------------------------
# Retorno ao baseline
# ----------------------------------------------------------------------------

def _residual_after_unload(b: _Bench) -> dict:
    b.first_tare()
    b.feed(_load(), 2.0)
    leftover = np.full(SHAPE, 2.0)
    r, c = BLOCK_REGIONS[2][0].start, BLOCK_REGIONS[2][1].start
    leftover[r, c:c + 5] += 20.0                                  # 100 RAW: abaixo do limiar de carga
    b.feed(leftover, 2.0)
    return b.worker.baseline_return.snapshot()


def test_baseline_return_separates_creep_from_missing_update(tmp_path: Path) -> None:
    fresh = _residual_after_unload(_Bench(tmp_path / "a"))
    assert fresh["sectors"]["2"]["verdict"] == "MEMORIA_OU_CREEP"
    assert fresh["sectors"]["2"]["residual_cells"] == 5
    assert fresh["sectors"]["7"]["verdict"] == "VOLTOU_A_TARA"

    b = _Bench(tmp_path / "b")
    b.ages = {}
    b.first_tare()
    b.feed(_load(), 2.0)
    b.ages = {2: 10.0}                                            # setor 2 sem pacote desde a carga
    leftover = np.full(SHAPE, 2.0)
    leftover[0, 32] = 9.0                                         # célula do setor 2 presa 7 RAW acima da tara
    b.feed(leftover, 2.0)
    frozen = b.worker.baseline_return.snapshot()
    assert frozen["sectors"]["2"]["verdict"] == "SEM_ATUALIZACAO"


# ----------------------------------------------------------------------------
# Log de diagnóstico
# ----------------------------------------------------------------------------

def test_diagnostic_log_has_sectors_tare_and_reference(tmp_path: Path) -> None:
    b = _Bench(tmp_path, diagnostics=True)
    b.worker.diag_log.set_reference(77.0, "centro")
    b.first_tare()
    b.feed(_load(), 2.0)
    b.worker.diag_log.close()
    files = list((tmp_path / "diagnostics").glob("diag_*.csv"))
    assert len(files) == 1
    rows = list(csv.DictReader(files[0].open(encoding="utf-8")))
    assert rows and list(rows[0].keys()) == FIELDS
    last = rows[-1]
    assert last["reference_mass_kg"] == "77.0" and last["reference_note"] == "centro"
    assert float(last["raw_corr_2"]) > 0 and float(last["raw_corr_1"]) == 0
    assert last["age_ms_2"] == "10"
    assert last["tare_id"] == b.worker.baseline.record.tare_id
    assert float(last["baseline_sum"]) == pytest.approx(4096.0)
    assert (tmp_path / "diagnostics" / "tare_events.jsonl").exists()


# ----------------------------------------------------------------------------
# CSVs duplicados (12X34 x 13X26)
# ----------------------------------------------------------------------------

def test_audit_flags_identical_calibration_csvs() -> None:
    definition = {
        "blocks": {},
        "sources": [
            {"cell": "12x34", "block": 2, "sha1": "abc123", "file": "a.csv"},
            {"cell": "13x26", "block": 3, "sha1": "abc123", "file": "b.csv"},
            {"cell": "9x53", "block": 1, "sha1": "fff", "file": "c.csv"},
        ],
    }
    audit = audit_calibration(definition)
    assert audit["duplicate_sources"] == [
        {"sha1": "abc123", "cells": ["12x34", "13x26"], "sectors": [2, 3], "files": ["a.csv", "b.csv"]}]
    assert any("idênticos" in a for a in audit["alerts"])
