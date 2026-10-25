"""
Pruebas unitarias para el registro en Model Registry e inferencia de P3 (Clustering).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.p3_register import register_candidate
from src.p3_config import (
    P3_PRODUCTION_ALIAS,
    P3_REGISTRY_NAME,
    P3_REPORTS_DIR,
)


def test_p3_register_rejects_empty_reason():
    """Verifica que el registro falle si no se proporciona justificación deliberada."""
    with pytest.raises(ValueError, match="Es obligatorio documentar la justificación"):
        register_candidate(run_id="dummy_run_id", reason="   ")


def test_p3_reports_integrity():
    """Verifica la existencia y consistencia de los reportes comparativo, de selección y auditoría."""
    comp_file = P3_REPORTS_DIR / "comparacion.csv"
    sel_file = P3_REPORTS_DIR / "seleccion.json"
    reg_file = P3_REPORTS_DIR / "registry.json"

    assert comp_file.exists(), f"No se encontró {comp_file}"
    assert sel_file.exists(), f"No se encontró {sel_file}"
    assert reg_file.exists(), f"No se encontró {reg_file}"

    # Validar tabla de comparación
    df_comp = pd.read_csv(comp_file)
    assert len(df_comp) >= 4  # baseline + al menos 3 candidatos
    assert "silhouette_score" in df_comp.columns
    assert "davies_bouldin_index" in df_comp.columns
    assert "stage" in df_comp.columns

    # Validar archivo de selección
    sel_data = json.loads(sel_file.read_text(encoding="utf-8"))
    assert sel_data["selected_candidate"] == "hourly_gmm_k2"
    assert sel_data["metrics"]["test_silhouette_score"] > sel_data["baseline_metrics"]["test_silhouette_score"]
    assert sel_data["model_registry_name"] == P3_REGISTRY_NAME
    assert sel_data["alias"] == P3_PRODUCTION_ALIAS

    # Validar archivo de auditoría del Model Registry
    reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
    assert reg_data["model_name"] == P3_REGISTRY_NAME
    assert reg_data["alias"] == P3_PRODUCTION_ALIAS
    assert reg_data["load_and_predict_verified"] is True
    assert reg_data["source_run_id"] == sel_data["selected_run_id"]
