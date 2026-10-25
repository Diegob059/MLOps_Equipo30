"""
Pruebas unitarias para el pipeline de entrenamiento, candidatos y registro de P2.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.p2_train import _evaluate_predictions
from src.p2_config import P2_CANDIDATE_CONFIGS, P2_REGISTRY_NAME, P2_PRODUCTION_ALIAS
from src.models.p2_register import register_candidate


def test_p2_candidate_configs_diversity():
    """Verifica que existan al menos 3 candidatos con configuraciones sustancialmente diversas."""
    assert len(P2_CANDIDATE_CONFIGS) >= 3

    # Verificar que los candidatos tengan nombres descriptivos y parámetros clave
    depths = [c["params"]["max_depth"] for c in P2_CANDIDATE_CONFIGS.values()]
    trees = [c["params"]["n_estimators"] for c in P2_CANDIDATE_CONFIGS.values()]
    leaves = [c["params"]["num_leaves"] for c in P2_CANDIDATE_CONFIGS.values()]

    # Deben existir diferencias en profundidad y número de árboles
    assert len(set(depths)) >= 3
    assert len(set(trees)) >= 3
    assert len(set(leaves)) >= 3


def test_p2_evaluate_predictions_metrics():
    """Verifica el cómputo correcto de métricas multiclase balanceadas."""
    y_true = pd.Series([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])  # 1 error en clase 1

    metrics = _evaluate_predictions(y_true, y_pred)
    assert "test_macro_f1" in metrics
    assert "test_balanced_accuracy" in metrics
    assert "test_recall_normal" in metrics
    assert "test_recall_desabasto" in metrics
    assert "test_recall_saturacion" in metrics

    # Recall de normal y saturación debe ser 100%
    assert metrics["test_recall_normal"] == 1.0
    assert metrics["test_recall_saturacion"] == 1.0
    # Recall de desabasto debe ser 50%
    assert metrics["test_recall_desabasto"] == 0.5


def test_p2_register_rejects_empty_reason():
    """Verifica que el registro falle si no se proporciona justificación de negocio."""
    with pytest.raises(ValueError, match="Es obligatorio documentar la justificación"):
        register_candidate(run_id="dummy_run", reason="")
