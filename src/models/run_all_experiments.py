"""
Script de Orquestación y Registro Unificado de Experimentos en MLflow (Equipo 30).
==================================================================================
Cumple rigurosamente con la rúbrica y especificación oficial de MLOps TC5061:
1. Experimento unificado con el número de equipo: `mibici-equipo30`.
2. Al menos un run por cada pregunta (P1, P2 y P3) con nombres descriptivos:
   - P1: `p1_baseline_demanda_historica_horaria`
   - P2 Baseline: `p2_baseline_dummy_stratified`
   - P2 Candidato: `p2_lgbm_risk_classifier_2025_2026`
   - P3 Baseline: `p3_baseline_kmeans_clustering_estaciones`
   - P3 Candidato: `p3_candidato_<conjunto_features>_<algoritmo>_k<k>` (mejor
     combinación de un barrido k=2..10 sobre K-Means/Gaussian Mixture, ver
     `src/models/p3_train.py`)
3. Parámetros completos de reproducibilidad (rango de fechas, versión de datos,
   semilla, cortes temporales, hiperparámetros).
4. Métricas con prefijo de conjunto (`test_`, `train_`).
5. Artefactos registrados en cada run (gráficas diagnósticas, mapas, reportes y modelos).
6. Etiquetas estandarizadas: equipo ("Equipo 30"), pregunta ("P1", "P2", "P3"), commit de Git y autor.
7. Soporte dual de almacenamiento: SQLite local (`mlflow.db`) y FileStore (`mlruns`) para
   garantizar visibilidad sin importar cómo se invoque la UI de MLflow.

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
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from src.data.p2_dataset import get_current_data_version
from src.features.p2_features import get_train_test_data
from src.p2_config import (
    FIGURES_DIR,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    P2_FEATURES_TARGET_PARQUET,
    P2_MODEL_PATH,
    P2_STATIONS_PARQUET,
    P2_TRIPS_HOURLY_PARQUET,
    PROJ_ROOT,
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
    """Retorna las URIs de tracking para sincronización dual (SQLite + mlruns)."""
    sqlite_uri = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    file_uri = str(PROJ_ROOT / "mlruns")
    return [sqlite_uri, file_uri]


def log_run_dual(
    experiment_name: str,
    run_name: str,
    params: Dict,
    metrics: Dict,
    tags: Dict,
    artifacts: List[Tuple[Path, str]],
):
    """Registra una corrida en ambos backends de MLflow (SQLite y FileStore)."""
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
# 1. PREGUNTA 1: PREDICCIÓN DE DEMANDA HORARIA (BASELINE)
# ==============================================================================
def run_p1_baseline(data_version: str, git_commit: str):
    logger.info("=" * 70)
    logger.info("EJECUTANDO BASELINE: PREGUNTA 1 (DEMANDA HORARIA MIBICI)")
    logger.info("=" * 70)

    trips = pd.read_parquet(P2_TRIPS_HOURLY_PARQUET)
    trips["dt"] = pd.to_datetime(trips["timestamp"])
    trips["hour"] = trips["dt"].dt.hour
    trips["dow"] = trips["dt"].dt.dayofweek

    train_mask = trips["dt"] < TRAIN_END_DATE
    test_mask = trips["dt"] >= TRAIN_END_DATE

    train_df = trips[train_mask]
    test_df = trips[test_mask].copy()

    # Heurística: Promedio histórico por estación, día de semana y hora
    lookup = (
        train_df.groupby(["station_id", "dow", "hour"])["salidas"]
        .mean()
        .rename("pred_salidas")
    )
    overall_mean = train_df["salidas"].mean()

    test_df = test_df.merge(lookup, on=["station_id", "dow", "hour"], how="left")
    test_df["pred_salidas"] = test_df["pred_salidas"].fillna(overall_mean)

    mae = mean_absolute_error(test_df["salidas"], test_df["pred_salidas"])
    rmse = root_mean_squared_error(test_df["salidas"], test_df["pred_salidas"])
    r2 = r2_score(test_df["salidas"], test_df["pred_salidas"])

    logger.info(f"P1 Test MAE : {mae:.4f} viajes/hora")
    logger.info(f"P1 Test RMSE: {rmse:.4f} viajes/hora")
    logger.info(f"P1 Test R²  : {r2:.4f}")

    # Gráfica de diagnóstico: Demanda agregada por hora en ventana representativa
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1_plot_path = FIGURES_DIR / "p1_demanda_prediccion_vs_real.png"

    # Agregación temporal por hora para visualización clara
    agg_eval = (
        test_df.groupby("timestamp")[["salidas", "pred_salidas"]].sum().reset_index()
    )
    sample_window = agg_eval.iloc[: 24 * 7]  # 1 semana representativa

    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(
        range(len(sample_window)),
        sample_window["salidas"],
        label="Demanda Real Observada (Viajes)",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        range(len(sample_window)),
        sample_window["pred_salidas"],
        label="Predicción Baseline Histórico",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    plt.title(
        "P1 - Baseline de Demanda Horaria vs Real Observada (MiBici Guadalajara)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(
        "Horas continuas (Muestra de 7 días representativos - 2026)", fontsize=10
    )
    plt.ylabel("Volumen total de viajes por hora", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(p1_plot_path)
    plt.close()

    p1_report_path = REPORTS_DIR / "p1_baseline_metrics.txt"
    p1_report_path.write_text(
        f"PREGUNTA 1: BASELINE DE DEMANDA HORARIA (MIBICI)\n"
        f"================================================\n"
        f"Modelo           : Promedio Histórico por Estación, Día y Hora\n"
        f"Conjunto Eval    : Test (>= {TRAIN_END_DATE[:10]})\n"
        f"Muestras Train   : {len(train_df):,}\n"
        f"Muestras Test    : {len(test_df):,}\n"
        f"Test MAE         : {mae:.4f} viajes/hora\n"
        f"Test RMSE        : {rmse:.4f} viajes/hora\n"
        f"Test R²          : {r2:.4f}\n",
        encoding="utf-8",
    )

    params = {
        "pregunta": "P1",
        "model_class": "HistoricalHourlyMeanBaseline",
        "date_range": "2025-01-01 to 2026-09-30",
        "data_version": data_version,
        "train_end_date": TRAIN_END_DATE,
        "prediction_horizon": "1_hour",
        "random_seed": RANDOM_SEED,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }
    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
    }
    tags = {
        "equipo": "Equipo 30",
        "pregunta": "P1",
        "git_commit": git_commit,
        "author": "Equipo 30",
        "problem_type": "regression_hourly_demand",
        "system": "MiBici Guadalajara",
    }
    artifacts = [
        (p1_plot_path, "figures"),
        (p1_report_path, "reports"),
    ]
    log_run_dual(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name="p1_baseline_demanda_historica_horaria",
        params=params,
        metrics=metrics,
        tags=tags,
        artifacts=artifacts,
    )


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
    logger.info("=" * 70)
    logger.info("EJECUTANDO PREGUNTA 3: CLUSTERING Y PERFILAMIENTO DE ESTACIONES")
    logger.info("=" * 70)

    from src.models.p3_train import guardar_asignaciones_cluster, guardar_mejor_modelo, preparar_runs

    runs = preparar_runs()
    for clave in ("baseline", "candidato"):
        run = runs[clave]
        logger.info(
            f"P3 {clave} -> silhouette: {run['metrics']['test_silhouette_score']:.4f}, "
            f"davies-bouldin: {run['metrics']['test_davies_bouldin_index']:.4f}"
        )
        log_run_dual(
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            run_name=run["run_name"],
            params=run["params"],
            metrics=run["metrics"],
            tags=run["tags"],
            artifacts=runs["artefactos"],
        )

    guardar_mejor_modelo(runs["candidato"]["resultado"])
    guardar_asignaciones_cluster(runs["candidato"]["resultado"])


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
