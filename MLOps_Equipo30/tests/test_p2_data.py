"""
Pruebas unitarias para la configuración y el manifiesto de datos (Pregunta 2).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.p2_config import (
    P2_DATA_MANIFEST_PATH,
    LABEL_MAPPING,
    OVERFLOW_RATIO_THRESHOLD,
    TARGET_HORIZON_HOURS,
    UNDERFLOW_RATIO_THRESHOLD,
)
from src.data.p2_dataset import compute_sha256, get_current_data_version


def test_p2_config_thresholds_and_targets():
    """Valida que los umbrales de negocio del target de P2 estén bien configurados."""
    assert TARGET_HORIZON_HOURS == 2
    assert UNDERFLOW_RATIO_THRESHOLD == -0.50
    assert OVERFLOW_RATIO_THRESHOLD == 0.50
    assert 0 in LABEL_MAPPING
    assert 1 in LABEL_MAPPING
    assert 2 in LABEL_MAPPING


def test_p2_data_manifest_accessible():
    """Valida que el manifiesto de datos de P2 exista y tenga un formato válido."""
    assert P2_DATA_MANIFEST_PATH.exists(), "El manifiesto p2_data_manifest.json debe existir."
    version = get_current_data_version()
    assert version != "unversioned_p2_data"
    assert "p2-mibici-v2025_2026" in version


def test_p2_sha256_computation(tmp_path):
    """Valida el cálculo de hashes SHA-256 en archivos."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("MiBici Guadalajara MLOps Test P2", encoding="utf-8")
    hash_val = compute_sha256(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
