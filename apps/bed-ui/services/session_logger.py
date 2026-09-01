"""Logger de sessões de calibração em formato CSV conforme Metodologia §11."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
import uuid

from config.settings import SOFTWARE_VERSION

logger = logging.getLogger(__name__)

CSV_HEADERS: list[str] = [
    "timestamp_iso",
    "session_id",
    "software_version",
    "block_id",
    "reference_kg",
    "reference_n",
    "position_tag",
    "repetition",
    "raw_sum_block",
    "net_sum_block",
    "mean_raw_block",
    "std_raw_block",
    "cv_pct",
    "estimated_kg",
    "estimated_n",
    "stability_state",
    "time_since_load_s",
    "tare_block_sum",
]

CSV_WIDE_HEADERS: list[str] = [
    "timestamp_iso",
    "session_id",
    "software_version",
    "peso_real_kg",
    "posicao",
    "repeticao",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "soma_total",
    "tempo_s",
    "estavel",
    "modelo_estimado",
]


class CalibrationSessionLogger:
    """Registra pontos de ensaio de calibração em arquivos CSV estruturados."""

    def __init__(
        self,
        output_dir: Path | str = "sessions",
        software_version: str = SOFTWARE_VERSION,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.software_version = software_version
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Garante que o diretório de destino exista."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_id(self) -> str:
        """Gera identificador único para a sessão."""
        return str(uuid.uuid4())

    def get_session_file_path(
        self,
        block_id: int,
        position_tag: str,
        session_date: datetime | None = None,
    ) -> Path:
        """Determina o caminho do arquivo CSV para a sessão de calibração."""
        date_str = (session_date or datetime.now(timezone.utc)).strftime("%Y%m%d")
        clean_tag = position_tag.strip().lower().replace(" ", "_")
        filename = f"calib_{date_str}_bloco{block_id}_{clean_tag}.csv"
        return self.output_dir / filename

    def _init_csv_file_if_needed(self, filepath: Path) -> None:
        """Escreve o cabeçalho CSV caso o arquivo seja novo ou vazio."""
        if filepath.exists() and filepath.stat().st_size > 0:
            return
        with filepath.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

    def _build_row(
        self,
        session_id: str,
        block_id: int,
        reference_kg: float | None,
        position_tag: str,
        repetition: int,
        raw_sum: float,
        net_sum: float,
        mean: float,
        std: float,
        cv_pct: float,
        estimated_kg: float | None,
        stability_state: str,
        time_since_load_s: float | None,
        tare_block_sum: float,
    ) -> list[Any]:
        """Formata os dados do ponto de calibração para a linha do CSV."""
        ref_n = round(reference_kg * 9.81, 4) if reference_kg is not None else None
        est_n = round(estimated_kg * 9.81, 4) if estimated_kg is not None else None
        timestamp_iso = datetime.now(timezone.utc).astimezone().isoformat()

        return [
            timestamp_iso,
            session_id,
            self.software_version,
            block_id,
            reference_kg,
            ref_n,
            position_tag,
            repetition,
            round(raw_sum, 4),
            round(net_sum, 4),
            round(mean, 4),
            round(std, 4),
            round(cv_pct, 4),
            estimated_kg,
            est_n,
            stability_state,
            time_since_load_s,
            round(tare_block_sum, 4),
        ]

    def log_calibration_point(
        self,
        session_id: str,
        block_id: int,
        reference_kg: float | None,
        position_tag: str,
        repetition: int,
        raw_sum: float,
        net_sum: float,
        mean: float,
        std: float,
        cv_pct: float,
        estimated_kg: float | None,
        stability_state: str = "stable",
        time_since_load_s: float | None = None,
        tare_block_sum: float = 0.0,
        custom_filepath: Path | None = None,
    ) -> Path | None:
        """Registra uma linha no arquivo CSV da sessão."""
        target_file = custom_filepath or self.get_session_file_path(block_id, position_tag)
        try:
            self._init_csv_file_if_needed(target_file)
            row = self._build_row(
                session_id=session_id,
                block_id=block_id,
                reference_kg=reference_kg,
                position_tag=position_tag,
                repetition=repetition,
                raw_sum=raw_sum,
                net_sum=net_sum,
                mean=mean,
                std=std,
                cv_pct=cv_pct,
                estimated_kg=estimated_kg,
                stability_state=stability_state,
                time_since_load_s=time_since_load_s,
                tare_block_sum=tare_block_sum,
            )
            with target_file.open(mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(row)
            return target_file
        except OSError as err:
            logger.error("falha ao gravar log de calibracao no csv %s: %s", target_file, err)
            return None

    def log_wide_point(
        self,
        session_id: str,
        peso_real_kg: float,
        posicao: str,
        repeticao: int,
        block_sums: list[float] | np.ndarray,
        tempo_s: float,
        estavel: int = 1,
        modelo_estimado: float | None = None,
        custom_filepath: Path | None = None,
    ) -> Path | None:
        """Registra ensaio completo da maca no formato largo [B1...B8]."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = custom_filepath or (self.output_dir / f"calib_wide_{date_str}.csv")
        try:
            if not target_file.exists() or target_file.stat().st_size == 0:
                with target_file.open(mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(CSV_WIDE_HEADERS)

            b_vals = [round(float(b), 4) for b in block_sums[:8]]
            while len(b_vals) < 8:
                b_vals.append(0.0)

            total_s = round(float(sum(b_vals)), 4)
            timestamp_iso = datetime.now(timezone.utc).astimezone().isoformat()

            row = [
                timestamp_iso,
                session_id,
                self.software_version,
                round(peso_real_kg, 4),
                posicao,
                repeticao,
                *b_vals,
                total_s,
                round(tempo_s, 2),
                estavel,
                round(modelo_estimado, 4) if modelo_estimado is not None else None,
            ]

            with target_file.open(mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(row)
            return target_file
        except OSError as err:
            logger.error("falha ao gravar log wide no csv %s: %s", target_file, err)
            return None
