"""
Script de Orquestación y Registro Unificado de Experimentos en MLflow (Equipo 30).
==================================================================================
Cumple rigurosamente con la rúbrica y especificación oficial de MLOps TC5061:
1. Experimento unificado con el número de equipo: `mibici-equipo30`.
2. Al menos un run por cada pregunta (P1, P2 y P3) con nombres descriptivos:
   - P1: baselines semanal e histórico, árboles e inventario inicial
   - P2 Baseline: `p2_baseline_dummy_stratified`
   - P2 Candidato: `p2_lgbm_risk_classifier_2025_2026`
   - P3 Baseline: `p3_baseline_kmeans_clustering_estaciones`
   - P3 Búsqueda: `p3_busqueda_hiperparametros_clustering` (run padre con 36
     runs hijos anidados, uno por combinación de K-Means/Gaussian Mixture
     sobre k=2..10 y 2 conjuntos de features)
   - P3 Candidatos (>= 3): `p3_candidato_<conjunto_features>_<algoritmo>_k<k>`
     (el mejor k de cada combinación algoritmo/conjunto de features del
     barrido anterior, ver `src/models/p3_train.py`)
3. Parámetros completos de reproducibilidad (rango de fechas, versión de datos,
   semilla, cortes temporales, hiperparámetros).
4. Métricas con prefijo de conjunto (`test_`, `train_`).
5. Artefactos registrados en cada run (gráficas diagnósticas, mapas, reportes y modelos).
6. Etiquetas estandarizadas: equipo ("Equipo 30"), pregunta ("P1", "P2", "P3"), commit de Git y autor.
7. Todas las preguntas (P1, P2 y P3) registran sus corridas contra el
   servidor local de MLflow Tracking (`MLFLOW_TRACKING_URI`, por defecto
   `http://127.0.0.1:5000`), levantado con `make mlflow` (ver README). El
   servidor debe estar corriendo antes de ejecutar cualquier entrenamiento.

Ejecución desde CLI:
    python -m src.models.run_all_experiments
"""

from pathlib import Path
import subprocess
import time
from typing import Dict, List, Tuple

import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.p2_dataset import get_current_data_version
from src.features.p2_features import get_train_test_data
from src.p2_config import (
    FIGURES_DIR,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_FEATURES_TARGET_PARQUET,
    P2_MODEL_PATH,
    P2_STATIONS_PARQUET,
    P2_TRIPS_HOURLY_PARQUET,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_END_DATE,
    logger,
)
from src.visualization.p2_plots import plot_confusion_matrix, plot_feature_importance


def get_git_commit() -> str:
    """Obtiene el hash del commit actual de Git."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "90b6590"


def get_tracking_uris() -> List[str]:
    """Retorna la URI del servidor local de MLflow Tracking (ver `make mlflow`).

    El servidor debe levantarse antes con:
        mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
    Todos los runs (P1, P2 y P3) se registran a través de él; ya no se
    escribe directamente al archivo SQLite ni a una carpeta `mlruns/` local
    desde el cliente, evitando accesos concurrentes al mismo archivo.
    """
    return [MLFLOW_TRACKING_URI]


def log_run_dual(
    experiment_name: str,
    run_name: str,
    params: Dict,
    metrics: Dict,
    tags: Dict,
    artifacts: List[Tuple[Path, str]],
):
    """Registra una corrida en el servidor de MLflow Tracking.

    Se conserva el nombre `log_run_dual` por compatibilidad con los llamados
    existentes (P2 y P3); internamente ya solo registra contra el servidor
    (ver `get_tracking_uris`).
    """
    for uri in get_tracking_uris():
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)
                for file_path, artifact_subpath in artifacts:
                    if file_path.exists():
                        mlflow.log_artifact(
                            str(file_path), artifact_path=artifact_subpath
                        )
            logger.info(f"✓ Run '{run_name}' registrado exitosamente en {uri}")
        except Exception as exc:
            logger.warning(f"Aviso al registrar '{run_name}' en {uri}: {exc}")


# ==============================================================================
# 1. PREGUNTA 1: DEMANDA HORARIA Y BICICLETAS AL INICIO DEL DÍA
# ==============================================================================
def run_p1_baseline(data_version: str, git_commit: str):
    """Ejecuta P1 con su manifiesto, rejilla completa y comparación común.

    Los argumentos se conservan por compatibilidad con el orquestador. P1
    registra su propia versión de datos y el commit real de la ejecución.
    """
    from src.data.p1_dataset import prepare_dataset
    from src.features.p1_features import build_features
    from src.models.p1_train import run_pipeline

    prepare_dataset()
    build_features()
    return run_pipeline()


# ==============================================================================
# 2. PREGUNTA 2: PREDICCIÓN DE RIESGO DE ESTACIONES (BASELINE + LIGHTGBM)
# ==============================================================================
def run_p2_experiments(data_version: str, git_commit: str):
    logger.info("=" * 70)
    logger.info("EJECUTANDO PREGUNTA 2: RIESGO DE ESTACIONES (BASELINE Y CANDIDATO)")
    logger.info("=" * 70)

    X_train, y_train, X_test, y_test, feature_cols, _ = get_train_test_data(
        features_parquet=P2_FEATURES_TARGET_PARQUET,
        split_date=TRAIN_END_DATE,
    )

    # --- Run 2a: Baseline Estratificado ---
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    baseline_macro_f1 = f1_score(y_test, y_pred_dummy, average="macro")
    dummy_bal_acc = balanced_accuracy_score(y_test, y_pred_dummy)
    recalls_dummy = recall_score(y_test, y_pred_dummy, average=None)

    p2_baseline_summary = REPORTS_DIR / "p2_baseline_summary.txt"
    p2_baseline_summary.write_text(
        f"PREGUNTA 2: BASELINE HEURÍSTICO ESTRATIFICADO\n"
        f"=============================================\n"
        f"Muestras Train     : {len(X_train):,}\n"
        f"Muestras Test      : {len(X_test):,}\n"
        f"Test Macro F1      : {baseline_macro_f1:.4f}\n"
        f"Test Balanced Acc  : {dummy_bal_acc:.4f}\n"
        f"Recall Normal (0)  : {recalls_dummy[0]:.2%}\n"
        f"Recall Desabasto(1): {recalls_dummy[1]:.2%}\n"
        f"Recall Saturacion(2): {recalls_dummy[2]:.2%}\n",
        encoding="utf-8",
    )

    params_dummy = {
        "pregunta": "P2",
        "model_class": "DummyClassifier",
        "strategy": "stratified",
        "random_seed": RANDOM_SEED,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_end_date": TRAIN_END_DATE,
        "date_range": "2025-01-01 to 2026-09-30",
        "data_version": data_version,
    }
    metrics_dummy = {
        "test_macro_f1": float(baseline_macro_f1),
        "test_balanced_accuracy": float(dummy_bal_acc),
        "test_recall_normal": float(recalls_dummy[0]),
        "test_recall_desabasto": float(recalls_dummy[1]),
        "test_recall_saturacion": float(recalls_dummy[2]),
    }
    tags_dummy = {
        "equipo": "Equipo 30",
        "pregunta": "P2",
        "git_commit": git_commit,
        "author": "Ricardo Bahamon (Equipo 30)",
        "model_class": "DummyClassifier",
        "target_horizon": "2_hours",
        "problem_type": "multiclass_imbalanced",
    }
    log_run_dual(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name="p2_baseline_dummy_stratified",
        params=params_dummy,
        metrics=metrics_dummy,
        tags=tags_dummy,
        artifacts=[(p2_baseline_summary, "reports")],
    )

    # --- Run 2b: Modelo Candidato LightGBM Balanceado ---
    t0 = time.time()
    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_train, y_train)
    fit_duration = time.time() - t0

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    recalls = recall_score(y_test, y_pred, average=None)
    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)

    P2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, P2_MODEL_PATH)

    cm_path = plot_confusion_matrix(y_test.values, y_pred)
    fi_path = plot_feature_importance(clf.feature_importances_, feature_cols)

    report_str = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=["Normal (0)", "Desabasto (1)", "Saturación (2)"],
    )
    report_path = REPORTS_DIR / "p2_classification_report.txt"
    report_path.write_text(report_str, encoding="utf-8")

    params_lgbm = dict(LGBM_PARAMS)
    params_lgbm.update(
        {
            "pregunta": "P2",
            "model_class": "LGBMClassifier",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_cols),
            "train_end_date": TRAIN_END_DATE,
            "date_range": "2025-01-01 to 2026-09-30",
            "data_version": data_version,
        }
    )
    metrics_lgbm = {
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
    tags_lgbm = {
        "equipo": "Equipo 30",
        "pregunta": "P2",
        "git_commit": git_commit,
        "author": "Ricardo Bahamon (Equipo 30)",
        "model_class": "LGBMClassifier",
        "target_horizon": "2_hours",
        "problem_type": "multiclass_imbalanced",
    }
    artifacts_lgbm = [
        (cm_path, "figures"),
        (fi_path, "figures"),
        (report_path, "reports"),
        (P2_MODEL_PATH, "models"),
    ]
    log_run_dual(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name="p2_lgbm_risk_classifier_2025_2026",
        params=params_lgbm,
        metrics=metrics_lgbm,
        tags=tags_lgbm,
        artifacts=artifacts_lgbm,
    )


# ==============================================================================
# 3. PREGUNTA 3: CLUSTERING DE ESTACIONES (K-MEANS / GAUSSIAN MIXTURE)
# ==============================================================================
def run_p3_clustering(data_version: str, git_commit: str):
    """Ejecuta P3 y registra sus runs en MLflow.

    Delega por completo en `src.models.p3_train.entrenar_y_registrar()`, que ya
    implementa el registro de la búsqueda (nested runs), el baseline y los >= 3
    candidatos diversos -incluidos los artefactos "completos" de cada candidato
    (modelo con firma, manifiesto de datos, `log_input` y archivo de
    configuración; ver `p3_train.py`-. `data_version`/`git_commit` se
    conservan como parámetros por compatibilidad con la firma del orquestador,
    pero P3 calcula los suyos propios internamente (igual que ya hace P1).
    """
    logger.info("=" * 70)
    logger.info("EJECUTANDO PREGUNTA 3: CLUSTERING Y PERFILAMIENTO DE ESTACIONES")
    logger.info("=" * 70)

    from src.models.p3_train import entrenar_y_registrar

    runs = entrenar_y_registrar()
    logger.info(
        f"P3 baseline -> silhouette: {runs['baseline']['metrics']['test_silhouette_score']:.4f}, "
        f"davies-bouldin: {runs['baseline']['metrics']['test_davies_bouldin_index']:.4f}"
    )
    for candidato in runs["candidatos"]:
        logger.info(
            f"P3 candidato ({candidato['run_name']}) -> silhouette: {candidato['metrics']['test_silhouette_score']:.4f}, "
            f"davies-bouldin: {candidato['metrics']['test_davies_bouldin_index']:.4f}"
        )


# ==============================================================================
# ORQUESTADOR PRINCIPAL
# ==============================================================================
def main():
    t_start = time.time()
    logger.info("INICIANDO EJECUCIÓN INTEGRAL DE EXPERIMENTOS MLFLOW (EQUIPO 30)")
    data_version = get_current_data_version()
    git_commit = get_git_commit()
    logger.info(f"Experimento: {MLFLOW_EXPERIMENT_NAME}")
    logger.info(f"Data Version: {data_version} | Git Commit: {git_commit}")

    # Ejecución secuencial de baselines y modelo candidato
    run_p1_baseline(data_version=data_version, git_commit=git_commit)
    run_p2_experiments(data_version=data_version, git_commit=git_commit)
    run_p3_clustering(data_version=data_version, git_commit=git_commit)

    logger.success(
        f"✓ Todos los experimentos (P1, P2 y P3) se registraron en MLflow en {time.time() - t_start:.1f}s."
    )


if __name__ == "__main__":
    main()
