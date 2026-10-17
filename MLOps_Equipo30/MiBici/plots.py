"""
Módulo de Visualizaciones y Diagnóstico de Modelos.
===================================================
Genera figuras estandarizadas para reportes y seguimiento de experimentos:
- Matriz de confusión dual (conteo absoluto y recall por clase).
- Gráfico de importancia de variables del modelo.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para ejecución en servidores y scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import typer

from MiBici.config import (
    CONFUSION_MATRIX_PLOT,
    FEATURE_IMPORTANCE_PLOT,
    FIGURES_DIR,
    LABEL_MAPPING,
    logger,
)

app = typer.Typer(help="Generación de figuras y diagnósticos de evaluación.")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
    output_path: Path = CONFUSION_MATRIX_PLOT,
) -> Path:
    """
    Genera y guarda una figura con la matriz de confusión en dos vistas:
    1. Conteos absolutos.
    2. Tasa de detección real (Recall normalizado por fila).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if class_labels is None:
        class_labels = [LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matriz en conteos
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    axes[0].set_title("Matriz de Confusión (Conteos Absolutos)")
    axes[0].set_ylabel("Etiqueta Real (Ground Truth)")
    axes[0].set_xlabel("Predicción del Modelo")

    # Matriz normalizada por fila (Recall)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Greens",
        ax=axes[1],
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    axes[1].set_title("Tasa de Detección Real (Recall por Clase)")
    axes[1].set_ylabel("Etiqueta Real (Ground Truth)")
    axes[1].set_xlabel("Predicción del Modelo")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"✓ Matriz de confusión guardada en: {output_path}")
    return output_path


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    output_path: Path = FEATURE_IMPORTANCE_PLOT,
    top_n: int = 15,
) -> Path:
    """
    Genera y guarda un gráfico de barras horizontales con las variables más influyentes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    if top_n and len(imp_series) > top_n:
        imp_series = imp_series.tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 5))
    imp_series.plot(kind="barh", color="#2b5c8f", ax=ax)
    ax.set_title(f"Top {len(imp_series)} Variables más Influyentes (Importancia Split)")
    ax.set_xlabel("Frecuencia de división (Split Importance)")
    ax.set_ylabel("Variable Predictora")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"✓ Gráfico de importancia de variables guardado en: {output_path}")
    return output_path
