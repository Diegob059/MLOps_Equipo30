"""
Módulo de Entrenamiento, Evaluación y Registro en MLflow (Pregunta 2).
=====================================================================
Ajusta el Baseline heurístico estratificado y el modelo candidato LightGBM con
ponderación de clases (`class_weight='balanced'`).
Evalúa sobre el conjunto de Test sin fuga temporal y registra parámetros,
métricas y artefactos en SQLite MLflow.

Ejecución desde terminal:
    python -m src.models.p2_train
    python -m src.models.p2_train --n-estimators 300 --max-depth 9
"""

from pathlib import Path
import time
from typing import Dict, Optional

import joblib
from lightgbm import LGBMClassifier
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from src.data.p2_dataset import get_current_data_version
from src.features.p2_features import get_train_test_data
from src.p2_config import (
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_FEATURES_TARGET_PARQUET,
    P2_MODEL_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_END_DATE,
    logger,
)
from src.visualization.p2_plots import plot_confusion_matrix, plot_feature_importance

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
    logger.info(
        "INICIO DE ENTRENAMIENTO Y EVALUACIÓN (PREGUNTA 2 - RIESGO DE ESTACIONES)"
    )
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

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    recalls = recall_score(y_test, y_pred, average=None)
    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)

    logger.info("=" * 70)
    logger.info(f"RESULTADOS SOBRE TEST (Evaluación >= {TRAIN_END_DATE[:10]}):")
    logger.info(
        f"  Macro F1-Score       : {macro_f1:.4f} (vs Baseline: {baseline_macro_f1:.4f})"
    )
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

    # 6. Generación de Figuras y Reportes en reports/
    cm_path = plot_confusion_matrix(y_test.values, y_pred)
    fi_path = plot_feature_importance(clf.feature_importances_, feature_cols)

    report_path = REPORTS_DIR / "p2_classification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_str, encoding="utf-8")

    # 7. Tracking y Registro en MLflow (Cumplimiento de Rúbrica TC5061)
    data_version = get_current_data_version()

    # Obtener commit de git para ligar la corrida con el código exacto
    import subprocess

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    dummy_bal_acc = balanced_accuracy_score(y_test, y_pred_dummy)
    recalls_dummy = recall_score(y_test, y_pred_dummy, average=None)

    try:
        logger.info(f"Registrando experimento en MLflow ({mlflow_tracking_uri})...")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        # ------------------------------------------------------------------
        # RUN 1: Baseline Heurístico Estratificado
        # ------------------------------------------------------------------
        with mlflow.start_run(run_name="p2_baseline_dummy_stratified") as run_dummy:
            mlflow.log_params(
                {
                    "model_class": "DummyClassifier",
                    "strategy": "stratified",
                    "random_seed": RANDOM_SEED,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "train_end_date": TRAIN_END_DATE,
                    "date_range": "2025-01-01 to 2026-09-30",
                    "data_version": data_version,
                }
            )
            mlflow.log_metrics(
                {
                    "test_macro_f1": float(baseline_macro_f1),
                    "test_balanced_accuracy": float(dummy_bal_acc),
                    "test_recall_normal": float(recalls_dummy[0]),
                    "test_recall_desabasto": float(recalls_dummy[1]),
                    "test_recall_saturacion": float(recalls_dummy[2]),
                }
            )
            baseline_summary_path = REPORTS_DIR / "p2_baseline_summary.txt"
            baseline_summary_path.write_text(
                f"BASELINE HEURÍSTICO ESTRATIFICADO (PREGUNTA 2)\n"
                f"================================================\n"
                f"Macro F1-Score      : {baseline_macro_f1:.4f}\n"
                f"Balanced Accuracy   : {dummy_bal_acc:.4f}\n"
                f"Recall Normal (0)   : {recalls_dummy[0]:.2%}\n"
                f"Recall Desabasto (1): {recalls_dummy[1]:.2%}\n"
                f"Recall Saturación (2): {recalls_dummy[2]:.2%}\n"
                f"Fecha Evaluación    : >= {TRAIN_END_DATE}\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(baseline_summary_path), artifact_path="reports")
            mlflow.set_tags(
                {
                    "equipo": "Equipo 30",
                    "pregunta": "P2",
                    "git_commit": git_commit,
                    "model_class": "DummyClassifier",
                    "author": "Ricardo Bahamon (Equipo 30)",
                    "target_horizon": "2_hours",
                    "problem_type": "multiclass_imbalanced",
                }
            )
            logger.info(f"✓ Run de Baseline registrado con ID: {run_dummy.info.run_id}")

        # ------------------------------------------------------------------
        # RUN 2: Modelo Candidato LightGBM con Class Weighting
        # ------------------------------------------------------------------
        with mlflow.start_run(run_name="p2_lgbm_risk_classifier_2025_2026") as run_lgbm:
            # Parámetros del modelo y del pipeline
            mlflow.log_params(lgbm_params)
            mlflow.log_params(
                {
                    "model_class": "LGBMClassifier",
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "num_features": len(feature_cols),
                    "train_end_date": TRAIN_END_DATE,
                    "date_range": "2025-01-01 to 2026-09-30",
                    "data_version": data_version,
                }
            )

            # Etiquetas del run: linaje de código, pregunta y gobernanza
            mlflow.set_tags(
                {
                    "equipo": "Equipo 30",
                    "pregunta": "P2",
                    "git_commit": git_commit,
                    "model_class": "LGBMClassifier",
                    "author": "Ricardo Bahamon (Equipo 30)",
                    "target_horizon": "2_hours",
                    "problem_type": "multiclass_imbalanced",
                }
            )

            # Métricas del run nombradas indicando el conjunto de datos (test_ / train_)
            mlflow.log_metrics(
                {
                    "test_macro_f1": float(macro_f1),
                    "test_balanced_accuracy": float(bal_acc),
                    "test_recall_normal": float(recalls[0]),
                    "test_recall_desabasto": float(recalls[1]),
                    "test_recall_saturacion": float(recalls[2]),
                    "test_precision_normal": float(precisions[0]),
                    "test_precision_desabasto": float(precisions[1]),
                    "test_precision_saturacion": float(precisions[2]),
                    "train_fit_duration_seconds": float(fit_duration),
                }
            )

            # Artefactos (Figuras diagnósticas, modelo serializado y reporte)
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path), artifact_path="figures")
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path), artifact_path="figures")
            if report_path.exists():
                mlflow.log_artifact(str(report_path), artifact_path="reports")
            if model_output_path.exists():
                mlflow.log_artifact(str(model_output_path), artifact_path="models")

            logger.info(f"✓ Run de LightGBM registrado con ID: {run_lgbm.info.run_id}")

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
def main(
    n_estimators: Optional[int] = typer.Option(None, help="Sobreescribe LGBM_PARAMS['n_estimators']."),
    max_depth: Optional[int] = typer.Option(None, help="Sobreescribe LGBM_PARAMS['max_depth']."),
    learning_rate: Optional[float] = typer.Option(None, help="Sobreescribe LGBM_PARAMS['learning_rate']."),
    num_leaves: Optional[int] = typer.Option(None, help="Sobreescribe LGBM_PARAMS['num_leaves']."),
):
    """Ejecuta el entrenamiento y evaluación para P2 desde CLI."""
    t0 = time.time()
    overrides = {
        key: value
        for key, value in {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
        }.items()
        if value is not None
    }
    lgbm_params = {**LGBM_PARAMS, **overrides} if overrides else None
    train_and_evaluate(lgbm_params=lgbm_params)
    logger.info(f"Pipeline de modelado finalizado en {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    app()
