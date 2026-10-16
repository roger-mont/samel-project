"""Características espaciais da matriz corrigida: centro de pressão, dispersão e células ativas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.settings import HID_ROWS, HID_COLS


# Grids pré-computados (evita recriação a cada frame)
_ROWS_GRID, _COLS_GRID = np.meshgrid(
    np.arange(HID_ROWS, dtype=np.float64),
    np.arange(HID_COLS, dtype=np.float64),
    indexing="ij",
)


@dataclass(frozen=True)
class SpatialFeatures:
    active_cells: int
    total_raw: float
    center_x: float           # coluna do centro de pressão (eixo longitudinal, 0..63)
    center_y: float           # linha do centro de pressão (eixo transversal, 0..31)
    spatial_dispersion: float  # raio de giração em células


def compute_spatial_features(matrix: np.ndarray) -> SpatialFeatures:
    total = float(np.sum(matrix))
    active = int(np.count_nonzero(matrix > 0))
    if total < 1e-6:
        return SpatialFeatures(active, total, 0.0, 0.0, 0.0)
    cop_row = float(np.sum(_ROWS_GRID * matrix)) / total
    cop_col = float(np.sum(_COLS_GRID * matrix)) / total
    dist_sq = (_ROWS_GRID - cop_row) ** 2 + (_COLS_GRID - cop_col) ** 2
    dispersion = float(np.sqrt(np.sum(dist_sq * matrix) / total))
    return SpatialFeatures(active, total, cop_col, cop_row, dispersion)


def compute_cop_2d(matrix: np.ndarray) -> tuple[float, float]:
    """Centro de Pressão (CoP) 2D: (linha, coluna). (0, 0) se a matriz está vazia."""
    total = float(np.sum(matrix))
    if total < 1e-6:
        return 0.0, 0.0

    cop_row = float(np.sum(_ROWS_GRID * matrix)) / total
    cop_col = float(np.sum(_COLS_GRID * matrix)) / total
    return cop_row, cop_col


def compute_cop_normalized(matrix: np.ndarray) -> tuple[float, float]:
    """CoP normalizado entre 0.0 e 1.0 para uso como feature de ML."""
    cop_row, cop_col = compute_cop_2d(matrix)
    if cop_row == 0.0 and cop_col == 0.0:
        return 0.0, 0.0

    return cop_row / (HID_ROWS - 1), cop_col / (HID_COLS - 1)


def compute_block_stats(
    matrix: np.ndarray,
    block_regions: dict[int, tuple[slice, slice]],
) -> dict[str, float]:
    """Extrai estatísticas intra-bloco: percentil 95, variância, sensores ativos."""
    stats: dict[str, float] = {}

    for bid, (row_sl, col_sl) in sorted(block_regions.items()):
        block = matrix[row_sl, col_sl]
        flat = block.ravel()
        active_mask = flat > 0

        active_count = int(np.sum(active_mask))
        stats[f"p95_B{bid}"] = float(np.percentile(flat, 95)) if active_count > 0 else 0.0
        stats[f"var_B{bid}"] = float(np.var(flat)) if active_count > 1 else 0.0
        stats[f"active_B{bid}"] = float(active_count)

    return stats
