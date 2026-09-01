"""Cálculo do Centro de Pressão (CoP) 2D com resolução de pixel.

Calcula a posição exata (x, y) do centroide da distribuição de pressão
na matriz 32×64, fornecendo informação espacial precisa para modelos
de estimação de peso e classificação postural.
"""
from __future__ import annotations

import numpy as np

from core.settings import HID_ROWS, HID_COLS


# Grids pré-computados (evita recriação a cada frame)
_ROWS_GRID, _COLS_GRID = np.meshgrid(
    np.arange(HID_ROWS, dtype=np.float64),
    np.arange(HID_COLS, dtype=np.float64),
    indexing="ij",
)


def compute_cop_2d(matrix: np.ndarray) -> tuple[float, float]:
    """Calcula Centro de Pressão (CoP) 2D real da matriz de pressão.

    Returns:
        (cop_row, cop_col) — coordenadas do centroide.
        Retorna (0.0, 0.0) se a matriz está vazia.
    """
    total = float(np.sum(matrix))
    if total < 1e-6:
        return 0.0, 0.0

    cop_row = float(np.sum(_ROWS_GRID * matrix)) / total
    cop_col = float(np.sum(_COLS_GRID * matrix)) / total
    return cop_row, cop_col


def compute_cop_normalized(matrix: np.ndarray) -> tuple[float, float]:
    """CoP normalizado entre 0.0 e 1.0 para uso como feature de ML.

    Returns:
        (cop_row_norm, cop_col_norm) — coordenadas normalizadas.
    """
    cop_row, cop_col = compute_cop_2d(matrix)
    if cop_row == 0.0 and cop_col == 0.0:
        return 0.0, 0.0

    return cop_row / (HID_ROWS - 1), cop_col / (HID_COLS - 1)


def compute_block_stats(
    matrix: np.ndarray,
    block_regions: dict[int, tuple[slice, slice]],
) -> dict[str, float]:
    """Extrai estatísticas intra-bloco: percentil 95, variância, sensores ativos.

    Complementa a soma por bloco já existente com métricas que capturam
    a distribuição de pressão dentro de cada bloco 16×16.
    """
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
