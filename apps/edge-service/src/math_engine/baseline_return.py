"""Retorno ao baseline depois de retirar a carga (parecer de 20/10/2026, item 4).

Quando a maca volta a SEM_CARGA, cada setor deveria voltar à tara. As células que
continuam acima dela (RAW corrigido > max(k·σ_tara, mínimo)) são separadas em:

- atualizadas há pouco: o firmware mandou valor novo e ele continua alto, ou seja,
  memória física / creep do sensor (o resíduo tende a cair com o tempo);
- sem atualização: a célula não recebe pacote há mais que o limite de STALE_SECTOR,
  ou seja, o valor é o último que chegou antes de a carga sair (falha de aquisição
  ou firmware que só manda células com pressão).

Guarda a série do resíduo por setor (uma amostra por segundo, até 10 min) para ver
se ele decai (creep) ou fica parado (congelado).

A contagem começa sozinha quando a maca passa de carga para SEM_CARGA. Como um setor
congelado com carga pode impedir a maca de chegar a SEM_CARGA, o operador também
pode marcar "carga retirada agora" (start_manual): aí a contagem roda 10 min sem
olhar o estado.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from storage.calibration_store import BLOCK_REGIONS

RESIDUAL_SIGMA_K = 3.0
RESIDUAL_MIN_RAW = 3.0
SERIES_PERIOD_S = 1.0
SERIES_MAX_S = 600.0


class BaselineReturnTracker:
    def __init__(self) -> None:
        self._unload_t: float | None = None
        self._loaded_seen = False
        self._series: dict[int, list[tuple[float, float, int]]] = {}
        self._last_sample_t: float | None = None
        self._last: dict[str, Any] | None = None
        self._manual = False
        self._manual_requested = False

    def start_manual(self) -> None:
        """Operador retirou a carga agora; vale a partir do próximo frame."""
        self._manual_requested = True

    def _begin(self, now: float, manual: bool) -> None:
        self._unload_t = now
        self._manual = manual
        self._loaded_seen = False
        self._series = {bid: [] for bid in BLOCK_REGIONS}
        self._last_sample_t = None
        self._last = None

    def update(
        self,
        empty: bool,
        now: float,
        corrected: np.ndarray,
        baseline_std: np.ndarray | None,
        cell_age_s: np.ndarray | None,
        stale_limit_s: float,
    ) -> None:
        """Chamado a cada frame. `empty` é SEM_CARGA com a manta conectada."""
        if self._manual_requested:
            self._manual_requested = False
            self._begin(now, manual=True)
        if self._manual:
            if now - self._unload_t > SERIES_MAX_S:  # type: ignore[operator]
                # Fim da contagem manual: o último resultado fica, e volta o modo automático
                self._manual = False
                self._unload_t = None
                return
        elif not empty:
            self._loaded_seen = True
            self._unload_t = None
            return
        elif self._unload_t is None:
            # Começa a contar só quando a maca fica vazia depois de ter tido carga
            if not self._loaded_seen:
                return
            self._begin(now, manual=False)
        if self._last_sample_t is not None and now - self._last_sample_t < SERIES_PERIOD_S:
            return
        self._last_sample_t = now
        self._last = self._evaluate(now, corrected, baseline_std, cell_age_s, stale_limit_s)

    def _evaluate(
        self,
        now: float,
        corrected: np.ndarray,
        baseline_std: np.ndarray | None,
        cell_age_s: np.ndarray | None,
        stale_limit_s: float,
    ) -> dict[str, Any]:
        assert self._unload_t is not None
        since = now - self._unload_t
        sigma = baseline_std if baseline_std is not None else np.zeros_like(corrected)
        limit = np.maximum(RESIDUAL_SIGMA_K * sigma, RESIDUAL_MIN_RAW)
        residual = corrected > limit
        old = (cell_age_s > stale_limit_s) if (cell_age_s is not None and stale_limit_s > 0) else None
        sectors: dict[str, Any] = {}
        for bid, region in BLOCK_REGIONS.items():
            res = residual[region]
            n_res = int(np.count_nonzero(res))
            raw_sum = float(np.sum(corrected[region][res])) if n_res else 0.0
            if since <= SERIES_MAX_S:
                self._series[bid].append((round(since, 1), round(raw_sum, 1), n_res))
            n_old = int(np.count_nonzero(res & old[region])) if old is not None else None
            n_fresh = (n_res - n_old) if n_old is not None else None
            series = self._series[bid]
            first = series[0][1] if series else raw_sum
            if n_res == 0:
                verdict = "VOLTOU_A_TARA"
            elif n_old is None:
                verdict = "RESIDUO_SEM_TELEMETRIA"
            elif n_old > 0:
                verdict = "SEM_ATUALIZACAO"
            else:
                verdict = "MEMORIA_OU_CREEP"
            max_age = None
            if cell_age_s is not None and n_res:
                ages = cell_age_s[region][res]
                finite = ages[np.isfinite(ages)]
                max_age = round(float(np.max(finite)), 3) if finite.size else None
            sectors[str(bid)] = {
                "residual_cells": n_res,
                "residual_raw_sum": round(raw_sum, 1),
                "residual_raw_sum_at_unload": round(first, 1),
                "cells_not_updated": n_old,
                "cells_updated": n_fresh,
                "max_cell_age_s": max_age,
                "verdict": verdict,
                "series": series[-120:],
            }
        return {"since_unload_s": round(since, 1), "manual": self._manual, "sectors": sectors}

    def snapshot(self) -> dict[str, Any]:
        return {
            "criteria": {
                "residual": f"RAW corrigido > max({RESIDUAL_SIGMA_K:g}·σ da tara, {RESIDUAL_MIN_RAW:g} RAW)",
                "not_updated": "célula sem pacote há mais que stale_sector_s",
                "verdicts": {
                    "VOLTOU_A_TARA": "nenhuma célula acima da tara",
                    "MEMORIA_OU_CREEP": "células acima da tara, mas recebendo valores novos",
                    "SEM_ATUALIZACAO": "há células acima da tara sem receber pacote (valor antigo)",
                    "RESIDUO_SEM_TELEMETRIA": "resíduo, mas o leitor não informa a idade das células",
                },
            },
            "waiting": self._last is None,
            "running": self._unload_t is not None,
            **(self._last or {}),
        }
