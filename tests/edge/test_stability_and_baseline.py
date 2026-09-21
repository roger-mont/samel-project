"""Testes automatizados para o detector de estabilidade FSM e compensador de baseline."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from math_engine.mat_baseline_compensator import MatBaselineCompensator
from math_engine.reading_stability_detector import ReadingStabilityDetector, ReadingState
from workers.acquisition_worker import _resolve_display_weight


def test_reading_stability_detector_empty_bed() -> None:
    """Verifica que leito com zeros ou ruido leve permanece SEM CARGA."""
    detector = ReadingStabilityDetector(empty_sum_threshold=50.0)
    matrix = np.zeros((32, 64), dtype=np.float64)
    state = detector.update(matrix, current_time=0.0)
    assert state == ReadingState.NO_LOAD


def test_reading_stability_detector_transitions_to_moving() -> None:
    """Verifica transição imediata para MOVIMENTO em delta matricial alto."""
    detector = ReadingStabilityDetector(d_max_matrix=100.0)
    matrix_1 = np.zeros((32, 64), dtype=np.float64)
    detector.update(matrix_1, current_time=0.0)

    matrix_2 = np.full((32, 64), 5.0, dtype=np.float64)  # soma = 10240, delta = 10240
    state = detector.update(matrix_2, current_time=0.025)
    assert state == ReadingState.MOVING


def test_reading_stability_detector_converges_to_stable() -> None:
    """Verifica transição ESTABILIZANDO -> CARGA ESTAVEL após Tmin."""
    detector = ReadingStabilityDetector(
        t_min_s=2.0,
        epsilon_s=10.0,
        sigma_max=5.0,
        d_max_matrix=50.0,
        empty_sum_threshold=50.0,
    )
    matrix = np.full((32, 64), 2.0, dtype=np.float64)  # soma = 4096

    # Primeiro frame estabelece referência (d_matriz alto -> MOVING)
    detector.update(matrix, current_time=0.0)

    # Preenche histórico com frames constantes
    for i in range(1, 30):
        t = i * 0.1
        state = detector.update(matrix, current_time=t)
        if t < 2.0:
            assert state in (ReadingState.MOVING, ReadingState.STABILIZING)

    # Aos 3.0s, deve atingir CARGA ESTAVEL
    final_state = detector.update(matrix, current_time=3.0)
    assert final_state == ReadingState.STABLE
    assert detector.get_progress_pct(3.0) == 100.0


def test_resolve_display_weight_freezes_during_motion() -> None:
    """Valida que o peso é retido no último válido durante MOVIMENTO."""
    disp_w, last_w, label, locked = _resolve_display_weight(
        state=ReadingState.MOVING,
        current_net_mass=72.5,
        last_valid_weight=68.0,
    )
    assert disp_w == 68.0
    assert last_w == 68.0
    assert "MOVIMENTO" in label
    assert locked is False


def test_resolve_display_weight_updates_on_stable() -> None:
    """Valida liberação de peso e travamento apenas em CARGA ESTAVEL."""
    disp_w, last_w, label, locked = _resolve_display_weight(
        state=ReadingState.STABLE,
        current_net_mass=70.2,
        last_valid_weight=68.0,
    )
    assert disp_w == 70.2
    assert last_w == 70.2
    assert label == "MEDIÇÃO ESTÁVEL"
    assert locked is True


def test_mat_baseline_compensator_subtraction() -> None:
    """Valida correção sc(t) = max(0, s(t) - B(t))."""
    compensator = MatBaselineCompensator(baseline_path=None)
    raw = np.full((32, 64), 20.0, dtype=np.float64)
    baseline = np.full((32, 64), 5.0, dtype=np.float64)
    compensator.update_baseline(baseline)

    corrected = compensator.correct_frame(raw)
    assert float(np.min(corrected)) == 15.0
    assert float(np.max(corrected)) == 15.0


def test_mat_baseline_compensator_auto_rezero_guard() -> None:
    """Valida que auto-rezero só dispara após tempo mínimo com leito vazio."""
    compensator = MatBaselineCompensator(baseline_path=None)
    empty = np.zeros((32, 64), dtype=np.float64)

    # Primeira verificação não dispara
    assert compensator.evaluate_auto_rezero(empty, current_time=0.0, min_empty_seconds=10.0) is False

    # 5 segundos depois ainda não dispara
    assert compensator.evaluate_auto_rezero(empty, current_time=5.0, min_empty_seconds=10.0) is False

    # 11 segundos depois dispara
    assert compensator.evaluate_auto_rezero(empty, current_time=11.0, min_empty_seconds=10.0) is True

    # Se entrar carga, reseta o tempo
    loaded = np.full((32, 64), 10.0, dtype=np.float64)
    assert compensator.evaluate_auto_rezero(loaded, current_time=12.0, min_empty_seconds=10.0) is False
