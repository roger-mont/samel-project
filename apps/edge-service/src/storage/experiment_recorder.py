"""Registro experimental com o RAW completo (itens 14 e 15 das orientações).

O RAW só é gravado durante uma sessão de experimento (decisão de 09/10).
Cada sessão vira uma pasta em data/experiments/<session_id>/:

- meta.json      parâmetros em vigor, modelo, calibração e horário de início/fim;
- raw.u8         matriz 32×64 de cada frame, uint8 em ordem de linha (2.048 bytes por frame);
- frames.csv     uma linha por frame, na mesma ordem do raw.u8, com metadados,
                 estado, flags, features e estimativa;
- baselines/     cada tara usada na sessão (<tare_id>.npz com média e desvio por célula).

Com o raw.u8 e o baseline dá para refazer a cadeia RAW → filtrado → corrigido →
features → estimativa com outro modelo. iter_cell_records() devolve o formato
longo do item 15, uma linha por célula.

Nos ensaios de célula (orientações de 15/10, item 5), o marcador também guarda a
célula ensaiada (target_cell, "linha x coluna" como nas pastas de docs/) e o setor
dela (target_block), para exportar por cell_id, block_id, cycle_id e direction com
scripts/exportar_ensaios_celula.py.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from core.settings import HID_COLS, HID_ROWS
from storage.calibration_store import BLOCK_REGIONS

logger = logging.getLogger(__name__)

FRAME_FIELDS = [
    "frame_idx",
    "timestamp",
    "session_id",
    "cycle_id",
    "configuration_id",
    "tare_id",
    "known_mass_g",
    "loading_direction",
    "state",
    "stability_flag",
    "movement_flag",
    "saturation_flag",
    "saturated_cells",
    "raw_sum",
    "total_raw",
    "raw_mean",
    "raw_std",
    "delta",
    "movement_index",
    "active_cells",
    "center_x",
    "center_y",
    "spatial_dispersion",
    "stabilization_time_s",
    "calibration_version",
    "model_version",
    "estimated_mass",
    "prediction_valid",
    "weight_status",
    "target_cell",
    "target_block",
    "stale_sectors",
    "max_sector_age_s",
]

LOADING_DIRECTIONS = ("UP", "DOWN", "UNKNOWN")
_CELL_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_cell(text: str) -> tuple[int, int]:
    """'12x34' → (12, 34): linha (0-31) e coluna (0-63) da matriz."""
    m = _CELL_RE.match(text or "")
    if not m:
        raise ValueError("célula deve ser 'linha x coluna', ex.: 12x34")
    row, col = int(m.group(1)), int(m.group(2))
    if not (0 <= row < HID_ROWS and 0 <= col < HID_COLS):
        raise ValueError(f"célula {row}x{col} fora da matriz {HID_ROWS}x{HID_COLS}")
    return row, col


def block_of(row: int, col: int) -> int:
    for bid, (rs, cs) in BLOCK_REGIONS.items():
        if rs.start <= row < rs.stop and cs.start <= col < cs.stop:
            return bid
    raise ValueError(f"célula {row}x{col} fora dos setores")


@dataclass
class ExperimentMarker:
    """Rótulos que o operador muda durante a sessão (massa conhecida, ciclo, sentido)."""

    cycle_id: int | None = None
    known_mass_g: float | None = None
    loading_direction: str = "UNKNOWN"
    target_cell: str | None = None     # célula ensaiada, "linha x coluna"
    target_block: int | None = None


@dataclass
class _Session:
    session_id: str
    directory: Path
    raw_file: Any
    csv_file: Any
    writer: csv.DictWriter
    marker: ExperimentMarker = field(default_factory=ExperimentMarker)
    frames: int = 0
    tare_ids: set[str] = field(default_factory=set)


class ExperimentRecorder:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._lock = threading.Lock()
        self._session: _Session | None = None

    @property
    def active(self) -> bool:
        return self._session is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            s = self._session
            if s is None:
                return {"active": False}
            return {
                "active": True,
                "session_id": s.session_id,
                "directory": str(s.directory),
                "frames": s.frames,
                "cycle_id": s.marker.cycle_id,
                "known_mass_g": s.marker.known_mass_g,
                "loading_direction": s.marker.loading_direction,
                "target_cell": s.marker.target_cell,
                "target_block": s.marker.target_block,
            }

    def start(self, meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._session is not None:
                raise RuntimeError(f"Sessão {self._session.session_id} já está ativa")
            session_id = str(uuid.uuid4())
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            directory = self._base_dir / f"{stamp}_{session_id[:8]}"
            (directory / "baselines").mkdir(parents=True, exist_ok=True)
            payload = {
                "session_id": session_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "frame_shape": [HID_ROWS, HID_COLS],
                "raw_dtype": "uint8",
                **meta,
            }
            (directory / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            raw_file = (directory / "raw.u8").open("ab")
            csv_file = (directory / "frames.csv").open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            self._session = _Session(session_id, directory, raw_file, csv_file, writer)
            logger.info("Experimento %s iniciado em %s", session_id, directory)
        return self.status()

    def mark(
        self,
        cycle_id: int | None = None,
        known_mass_g: float | None = None,
        loading_direction: str | None = None,
        target_cell: str | None = None,
    ) -> dict[str, Any]:
        """Muda os rótulos a partir do próximo frame. target_cell="" limpa a célula ensaiada."""
        cell: tuple[int, int] | None = None
        if target_cell:
            cell = parse_cell(target_cell)
        with self._lock:
            if self._session is None:
                raise RuntimeError("Nenhuma sessão de experimento ativa")
            m = self._session.marker
            if cycle_id is not None:
                m.cycle_id = int(cycle_id)
            if known_mass_g is not None:
                m.known_mass_g = float(known_mass_g)
            if loading_direction is not None:
                direction = loading_direction.upper()
                if direction not in LOADING_DIRECTIONS:
                    raise ValueError(f"loading_direction deve ser um de {LOADING_DIRECTIONS}")
                m.loading_direction = direction
            if target_cell is not None:
                m.target_cell = f"{cell[0]}x{cell[1]}" if cell else None
                m.target_block = block_of(*cell) if cell else None
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            s = self._session
            if s is None:
                return {"active": False}
            s.raw_file.close()
            s.csv_file.close()
            meta_path = s.directory / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
            meta["frames"] = s.frames
            meta["tare_ids"] = sorted(s.tare_ids)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._session = None
            logger.info("Experimento %s encerrado (%d frames)", s.session_id, s.frames)
            return {"active": False, "session_id": s.session_id, "frames": s.frames, "directory": str(s.directory)}

    def record(
        self,
        raw_matrix: np.ndarray,
        fields_: dict[str, Any],
        baseline_mean: np.ndarray | None,
        baseline_std: np.ndarray | None,
    ) -> None:
        """Grava um frame. Chamado pela thread de aquisição a cada leitura."""
        with self._lock:
            s = self._session
            if s is None:
                return
            tare_id = fields_.get("tare_id")
            if tare_id and tare_id not in s.tare_ids and baseline_mean is not None:
                np.savez_compressed(
                    s.directory / "baselines" / f"{tare_id}.npz",
                    baseline_raw=baseline_mean,
                    baseline_std=baseline_std if baseline_std is not None else np.zeros_like(baseline_mean),
                )
                s.tare_ids.add(tare_id)
            s.raw_file.write(np.clip(np.rint(raw_matrix), 0, 255).astype(np.uint8).tobytes(order="C"))
            row = {k: fields_.get(k) for k in FRAME_FIELDS}
            row.update(
                frame_idx=s.frames,
                session_id=s.session_id,
                cycle_id=s.marker.cycle_id,
                known_mass_g=s.marker.known_mass_g,
                loading_direction=s.marker.loading_direction,
                target_cell=s.marker.target_cell,
                target_block=s.marker.target_block,
            )
            s.writer.writerow(row)
            s.frames += 1
            if s.frames % 40 == 0:
                s.raw_file.flush()
                s.csv_file.flush()


# ----------------------------------------------------------------------------
# Leitura para reprocessamento
# ----------------------------------------------------------------------------

def load_raw_frames(session_dir: Path) -> np.ndarray:
    """Devolve as matrizes RAW da sessão, formato (n_frames, 32, 64), uint8."""
    data = np.fromfile(session_dir / "raw.u8", dtype=np.uint8)
    return data.reshape(-1, HID_ROWS, HID_COLS)


def iter_cell_records(session_dir: Path, cells: list[tuple[int, int]] | None = None) -> Iterator[dict[str, Any]]:
    """Formato longo do item 15: uma linha por célula e por frame.

    Junta o RAW de cada frame com o baseline da tara em vigor e os metadados do frame.
    'cells' limita a saída a algumas células (linha, coluna); None = todas as 2.048.
    """
    raw = load_raw_frames(session_dir)
    baselines: dict[str, np.ndarray] = {}
    for f in (session_dir / "baselines").glob("*.npz"):
        baselines[f.stem] = np.load(f)["baseline_raw"]
    zero = np.zeros((HID_ROWS, HID_COLS))
    targets = cells or [(r, c) for r in range(HID_ROWS) for c in range(HID_COLS)]

    with (session_dir / "frames.csv").open(encoding="utf-8") as fh:
        for frame_meta in csv.DictReader(fh):
            idx = int(frame_meta["frame_idx"])
            base = baselines.get(frame_meta.get("tare_id") or "", zero)
            for r, c in targets:
                raw_value = float(raw[idx, r, c])
                yield {
                    **frame_meta,
                    "cell_id": r * HID_COLS + c,
                    "row": r,
                    "column": c,
                    "raw_value": raw_value,
                    "baseline_raw": float(base[r, c]),
                    "corrected_raw": max(0.0, raw_value - float(base[r, c])),
                }
