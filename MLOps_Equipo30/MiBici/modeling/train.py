"""
Módulo de Entrenamiento, Evaluación y Registro de Experimentos (MLflow).
========================================================================
Entrena el modelo candidato (LightGBM con class_weight='balanced') y el baseline
heurístico para responder a la Pregunta 2 de MiBici Guadalajara.

Flujo:
    1. Carga del dataset procesado en Parquet (data/processed/mibici_features_target.parquet).
    2. Partición temporal estricta (Train < 2024-07-01 <= Test).
    3. Evaluación del Baseline Estratificado.
    4. Ajuste del modelo LightGBMClassifier con balanceo de clases.
    5. Cálculo de métricas: Macro F1, Balanced Accuracy, Recall y Precision por clase.
    6. Generación y almacenamiento de figuras (Matriz de confusión, Importancia de variables).
    7. Serialización del modelo entrenado en models/lgbm_risk_classifier.joblib.
    8. Registro automático del experimento, hiperparámetros, versión del dataset y artefactos en MLflow.

Ejecución desde terminal:
    python -m MiBici.modeling.train
    python -m MiBici.modeling.train --help
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from MiBici.config import (
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PLOT,
    FEATURE_IMPORTANCE_PLOT,
    LABEL_MAPPING,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    OVERFLOW_RATIO_THRESHOLD,
    PROCESSED_FEATURES_TARGET_PARQUET,
    RANDOM_SEED,
    TARGET_HORIZON_HOURS,
    UNDERFLOW_RATIO_THRESHOLD,
    logger,
)
from MiBici.dataset import get_current_data_version
from MiBici.features import (
    build_full_feature_pipeline,
    split_and_sanitize_data,
)
from MiBici.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
)

app = typer.Typer(help="CLI para Entrenamiento y Evaluación de Modelos MiBici.")


def train_and_evaluate_model(
    features_path: Path = PROCESSED_FEATURES_TARGET_PARQUET,
    model_output_path: Path = BEST_MODEL_PATH,
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """
    Ejecuta el ciclo completo de entrenamiento, evaluación visual y registro en MLflow.
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("INICIO DE ENTRENAMIENTO Y EVALUACIÓN (PREGUNTA 2 - RIESGO DE ESTACIONES)")
    logger.info("=" * 70)

    # 1. Cargar dataset procesado o generar si no existe
    if not features_path.exists():
        logger.warning(
            f"El dataset procesado no existe en {features_path}. Ejecutando pipeline de features..."
        )
        df, feature_cols = build_full_feature_pipeline()
    else:
        logger.info(f"Cargando dataset procesado desde: {features_path}")
        df = pd.read_parquet(features_path)
        # Identificar columnas predictoras excluyendo metadatos y target
        excluded_cols = {"station_id", "timestamp", "salidas", "llegadas", "balance_neto", "delta_net", "ratio", "target"}
        feature_cols = [c for c in df.columns if c not in excluded_cols]

    # 2. Partición temporal estricta (Train vs Test)
    X_train, y_train, X_test, y_test = split_and_sanitize_data(df, feature_cols)

    # 3. Baseline Estratificado
    logger.info("Evaluando Baseline Heurístico Estratificado...")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_macro_f1 = f1_score(y_test, y_pred_dummy, average="macro")
    logger.info(f"Baseline Estratificado Macro F1: {baseline_macro_f1:.4f}")

    # 4. Entrenamiento del Modelo Candidato (LightGBM)
    logger.info("Ajustando modelo candidato LightGBM con class_weight='balanced'...")
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)

    t_fit_start = time.time()
    clf.fit(X_train, y_train)
    fit_duration = time.time() - t_fit_start
    logger.success(f"✓ Entrenamiento completado en {fit_duration:.1f} segundos.")

    # 5. Evaluación cuantitativa sobre el conjunto de Test
    logger.info("Generando predicciones sobre el conjunto de Test...")
    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    recalls = recall_score(y_test, y_pred, average=None)
    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)

    logger.info("=" * 70)
    logger.info(f"RESULTADOS SOBRE TEST (Evaluación >= {TRAIN_END_DATE[:10]}):")
    logger.info(f"  Macro F1-Score       : {macro_f1:.4f} (vs Baseline: {baseline_macro_f1:.4f})")
    logger.info(f"  Balanced Accuracy    : {bal_acc:.4f}")
    logger.info(f"  Recall Desabasto (1) : {recalls[1]:.2%}")
    logger.info(f"  Recall Saturación (2): {recalls[2]:.2%}")
    logger.info(f"  Recall Normal (0)    : {recalls[0]:.2%}")
    logger.info("=" * 70)

    report_str = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=["Normal (0)", "Desabasto (1)", "Saturación (2)"],
    )
    logger.info("\nReporte de Clasificación Detallado:\n" + report_str)

    # 6. Serialización del Modelo en models/
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_output_path)
    logger.success(f"✓ Modelo serializado guardado en: {model_output_path}")

    # 7. Generación de Figuras en reports/figures/
    cm_path = plot_confusion_matrix(y_test.values, y_pred)
    fi_path = plot_feature_importance(clf.feature_importances_, feature_cols)

    # 8. Tracking y Registro en MLflow
    data_version = get_current_data_version()
    metrics_dict = {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "baseline_macro_f1": float(baseline_macro_f1),
        "recall_normal": float(recalls[0]),
        "recall_desabasto": float(recalls[1]),
        "recall_saturacion": float(recalls[2]),
        "precision_normal": float(precisions[0]),
        "precision_desabasto": float(precisions[1]),
        "precision_saturacion": float(precisions[2]),
        "training_time_seconds": float(fit_duration),
    }

    if log_to_mlflow:
        logger.info(f"Registrando experimento en MLflow ({MLFLOW_TRACKING_URI})...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="lgbm_balanced_v1") as run:
            # Parámetros e hiperparámetros
            mlflow.log_params(LGBM_PARAMS)
            mlflow.log_param("target_horizon_hours", TARGET_HORIZON_HOURS)
            mlflow.log_param("underflow_threshold", UNDERFLOW_RATIO_THRESHOLD)
            mlflow.log_param("overflow_threshold", OVERFLOW_RATIO_THRESHOLD)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features_count", len(feature_cols))

            # Métricas de rendimiento
            mlflow.log_metrics(metrics_dict)

            # Tags de negocio
            mlflow.set_tag("task", "pregunta_2_riesgo_estaciones")
            mlflow.set_tag("model_type", "LightGBMClassifier")
            mlflow.set_tag("data_manifest_version", data_version)

            # Artefactos: figuras y reporte de texto
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.log_artifact(str(fi_path), artifact_path="figures")
            mlflow.log_artifact(str(model_output_path), artifact_path="model")

            logger.success(f"✓ Corrida de MLflow registrada exitosamente con ID: {run.info.run_id}")

    elapsed_total = time.time() - t_start
    logger.success(f"Pipeline de modelado finalizado en {elapsed_total:.1f}s.")
    return metrics_dict


@app.command()
def main(
    mlflow_tracking: bool = typer.Option(
        True, help="Registrar parámetros, métricas y artefactos en MLflow."
    ),
):
    """Entrena el modelo LightGBM para predicción de riesgo y registra resultados."""
    train_and_evaluate_model(log_to_mlflow=mlflow_tracking)


if __name__ == "__main__":
    app()
