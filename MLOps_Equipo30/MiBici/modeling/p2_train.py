"""
Módulo de Entrenamiento, Evaluación y Registro en MLflow (Pregunta 2).
=====================================================================
Ajusta el Baseline heurístico estratificado y el modelo candidato LightGBM con
ponderación de clases (`class_weight='balanced'`).
Evalúa sobre el conjunto de Test sin fuga temporal y registra parámetros,
métricas y artefactos en SQLite MLflow.

Ejecución desde terminal:
    python -m MiBici.modeling.p2_train
"""

import time
from pathlib import Path
from typing import Dict, Optional

import joblib
from lightgbm import LGBMClassifier
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from MiBici.p2_config import (
    P2_MODEL_PATH,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_FEATURES_TARGET_PARQUET,
    RANDOM_SEED,
    TRAIN_END_DATE,
    logger,
)
from MiBici.p2_dataset import get_current_data_version
from MiBici.p2_features import get_train_test_data
from MiBici.p2_plots import plot_confusion_matrix, plot_feature_importance

app = typer.Typer(
    help="CLI de Entrenamiento y Evaluación de Modelos para la Pregunta 2 (MiBici)."
)


def train_and_evaluate(
    features_parquet: Path = P2_FEATURES_TARGET_PARQUET,
    model_output_path: Path = P2_MODEL_PATH,
    lgbm_params: Optional[Dict] = None,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> Dict:
    """
    Ejecuta el ciclo completo de entrenamiento, evaluación comparativa y tracking.
    """
    if lgbm_params is None:
        lgbm_params = LGBM_PARAMS

    logger.info("=" * 70)
    logger.info("INICIO DE ENTRENAMIENTO Y EVALUACIÓN (PREGUNTA 2 - RIESGO DE ESTACIONES)")
    logger.info("=" * 70)
    logger.info(f"Cargando dataset procesado desde: {features_parquet}")

    # 1. Partición y sanitización estricta
    logger.info("Aplicando sanitización y partición temporal estricta...")
    X_train, y_train, X_test, y_test, feature_cols, df_clean = get_train_test_data(
        features_parquet=features_parquet,
        split_date=TRAIN_END_DATE,
    )
    logger.info(
        f"Partición completada -> Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras."
    )

    # 2. Baseline Heurístico Estratificado
    logger.info("Evaluando Baseline Heurístico Estratificado...")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_macro_f1 = f1_score(y_test, y_pred_dummy, average="macro")
    logger.info(f"Baseline Estratificado Macro F1: {baseline_macro_f1:.4f}")

    # 3. Entrenamiento de LightGBM
    logger.info("Ajustando modelo candidato LightGBM con class_weight='balanced'...")
    t0 = time.time()
    clf = LGBMClassifier(**lgbm_params)
    clf.fit(X_train, y_train)
    fit_duration = time.time() - t0
    logger.info(f"✓ Entrenamiento completado en {fit_duration:.1f} segundos.")

    # 4. Evaluación en Test
    logger.info("Generando predicciones sobre el conjunto de Test...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

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

    # 5. Serialización del Modelo en models/
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_output_path)
    logger.success(f"✓ Modelo serializado guardado en: {model_output_path}")

    # 6. Generación de Figuras en reports/figures/
    cm_path = plot_confusion_matrix(y_test.values, y_pred)
    fi_path = plot_feature_importance(clf.feature_importances_, feature_cols)

    # 7. Tracking y Registro en MLflow
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

    try:
        logger.info(f"Registrando experimento en MLflow ({mlflow_tracking_uri})...")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="LGBM-P2-Risk-Predictor") as run:
            # Parámetros
            mlflow.log_params(lgbm_params)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.log_param("train_end_date", TRAIN_END_DATE)

            # Tags de trazabilidad y gobernanza de datos
            mlflow.set_tag("model_type", "LGBMClassifier")
            mlflow.set_tag("data_version", data_version)
            mlflow.set_tag("target_horizon", "2_hours")
            mlflow.set_tag("problem_type", "multiclass_imbalanced")
            mlflow.set_tag("question", "Pregunta 2 - Riesgo Desabasto/Saturacion")

            # Métricas
            mlflow.log_metrics(metrics_dict)

            # Artefactos
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path), artifact_path="figures")
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path), artifact_path="figures")
            if model_output_path.exists():
                mlflow.log_artifact(str(model_output_path), artifact_path="models")

            logger.info(f"✓ Corrida de MLflow registrada exitosamente con ID: {run.info.run_id}")

    except Exception as exc:
        logger.warning(f"No fue posible completar el registro en MLflow: {exc}")

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "recalls": recalls,
        "model_path": model_output_path,
        "feature_cols": feature_cols,
    }


@app.command()
def main():
    """Ejecuta el entrenamiento y evaluación para P2 desde CLI."""
    t0 = time.time()
    train_and_evaluate()
    logger.info(f"Pipeline de modelado finalizado en {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    app()
