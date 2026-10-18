"""
Módulo de Visualización Diagnóstica para la Pregunta 2.
======================================================
Genera y guarda figuras de evaluación operativa (matriz de confusión e importancia
de características) para documentación técnica y artefactos en MLflow.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.p2_config import (
    LABEL_MAPPING,
    P2_CONFUSION_MATRIX_PLOT,
    P2_FEATURE_IMPORTANCE_PLOT,
    logger,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = P2_CONFUSION_MATRIX_PLOT,
    labels: Optional[List[str]] = None,
) -> Path:
    """Genera y guarda la matriz de confusión normalizada por filas (recall)."""
    if labels is None:
        labels = [LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(
        "Matriz de Confusión Normalizada (Recall por Clase - P2)", fontsize=12, pad=12
    )
    ax.set_xlabel("Clase Predicha", fontsize=10)
    ax.set_ylabel("Clase Real (Ground Truth)", fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Matriz de confusión guardada en: {output_path}")
    return output_path


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    output_path: Path = P2_FEATURE_IMPORTANCE_PLOT,
    top_n: int = 15,
) -> Path:
    """Genera y guarda el gráfico de barras horizontales de importancia de variables."""
    s_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    s_top = s_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    s_top.plot(kind="barh", color="#1f77b4", ax=ax)
    ax.set_title(
        f"Top {top_n} Variables Más Importantes (LightGBM - P2)", fontsize=12, pad=12
    )
    ax.set_xlabel("Importancia de División (Splits)", fontsize=10)
    ax.set_ylabel("Variable", fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Gráfico de importancia de variables guardado en: {output_path}")
    return output_path
