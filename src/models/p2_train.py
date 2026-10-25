"""
Módulo de Entrenamiento, Evaluación y Registro en MLflow (Pregunta 2: Riesgo).
==============================================================================
Ejecuta la experimentación avanzada y el ciclo de modelado para la Pregunta 2:
1. Baseline heurístico estratificado (DummyClassifier).
2. Evaluación comparativa de al menos 3 candidatos con diferencias sustanciales
   en hiperparámetros y capacidad (balanceado base, alta capacidad profundo y
   regularizado compacto).
3. Registro de artefactos completos en MLflow para cada candidato:
   - Modelo serializado con `mlflow.sklearn.log_model`, firma (`signature`) e `input_example`.
   - Dataset de entrenamiento vía `mlflow.log_input`.
   - Manifiesto de datos (`data/`).
   - Configuración reproducible (`config/`).
   - Gráficas diagnósticas de evaluación (`figures/`).
   - Reporte de clasificación detallado (`reports/`).
4. Etiquetas estandarizadas para filtrado y gobernanza (`stage_of_work="candidate"`,
   `question="p2"`, `git_commit`, `git_dirty`, `data_version`, `equipo`).
5. Comparación multicriterio (`reports/p2/comparacion.csv`) y selección del
   Champion (`reports/p2/seleccion.json`).

Uso desde terminal:
    python -m src.models.p2_train
    python -m src.models.p2_train --n-estimators 200 --max-depth 9
"""

import json
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
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

from src.data.p2_dataset import get_current_data_version
from src.features.p2_features import get_train_test_data
from src.p2_config import (
    FIGURES_DIR,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_CANDIDATE_CONFIGS,
    P2_CONFUSION_MATRIX_PLOT,
    P2_DATA_MANIFEST_PATH,
    P2_FEATURE_IMPORTANCE_PLOT,
    P2_FEATURES_TARGET_PARQUET,
    P2_MODEL_PATH,
    P2_REPORTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_HORIZON_HOURS,
    TRAIN_CUTOFF_DATE,
    TRAIN_END_DATE,
    logger,
)
from src.visualization.p2_plots import plot_confusion_matrix, plot_feature_importance

app = typer.Typer(
    help="CLI de Experimentación Avanzada y Registro en MLflow para la Pregunta 2 (MiBici)."
)


def get_git_info() -> Tuple[str, str]:
    """Obtiene el hash del commit actual y el estado de limpieza del árbol de trabajo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        dirty = "True" if status else "False"
    except Exception:
        dirty = "unknown"
    return commit, dirty


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula las métricas estándar de evaluación sobre el conjunto de test."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    recalls = recall_score(y_true, y_pred, average=None)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "test_macro_f1": float(macro_f1),
        "test_balanced_accuracy": float(bal_acc),
        "test_recall_normal": float(recalls[0]),
        "test_recall_desabasto": float(recalls[1]),
        "test_recall_saturacion": float(recalls[2]),
        "test_precision_normal": float(precisions[0]),
        "test_precision_desabasto": float(precisions[1]),
        "test_precision_saturacion": float(precisions[2]),
    }


def train_and_evaluate(
    features_parquet: Path = P2_FEATURES_TARGET_PARQUET,
    model_output_path: Path = P2_MODEL_PATH,
    lgbm_params: Optional[Dict] = None,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> Dict:
    """
    Ejecuta la experimentación de P2:
    - Si se especifican overrides en `lgbm_params`, entrena esa configuración única.
    - Si `lgbm_params is None`, ejecuta la suite completa: Baseline Estratificado +
      los 3 candidatos diversos de `P2_CANDIDATE_CONFIGS`, compara resultados y
      selecciona el Champion.
    """
    logger.info("=" * 70)
    logger.info("INICIO DE EXPERIMENTACIÓN AVANZADA (PREGUNTA 2 - RIESGO DE ESTACIONES)")
    logger.info("=" * 70)
    logger.info(f"Cargando dataset procesado desde: {features_parquet}")

    # 1. Partición y sanitización estricta con corte anti-leakage
    X_train, y_train, X_test, y_test, feature_cols, df_clean = get_train_test_data(
        features_parquet=features_parquet,
        split_date=TRAIN_END_DATE,
        horizon_hours=TARGET_HORIZON_HOURS,
    )
    logger.info(
        f"Partición completada -> Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras."
    )

    data_version = get_current_data_version()
    git_commit, git_dirty = get_git_info()

    # Conexión con MLflow
    logger.info(f"Configurando servidor MLflow Tracking: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Evaluación y Registro del Baseline Heurístico Estratificado
    logger.info("Evaluando Baseline Heurístico Estratificado...")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    metrics_dummy = _evaluate_predictions(y_test, y_pred_dummy)

    baseline_summary_path = REPORTS_DIR / "p2_baseline_summary.txt"
    baseline_summary_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_summary_path.write_text(
        f"BASELINE HEURÍSTICO ESTRATIFICADO (PREGUNTA 2)\n"
        f"================================================\n"
        f"Macro F1-Score       : {metrics_dummy['test_macro_f1']:.4f}\n"
        f"Balanced Accuracy    : {metrics_dummy['test_balanced_accuracy']:.4f}\n"
        f"Recall Normal (0)    : {metrics_dummy['test_recall_normal']:.2%}\n"
        f"Recall Desabasto (1) : {metrics_dummy['test_recall_desabasto']:.2%}\n"
        f"Recall Saturación (2): {metrics_dummy['test_recall_saturacion']:.2%}\n"
        f"Fecha Corte Train    : < {TRAIN_CUTOFF_DATE} (Anti-Leakage)\n"
        f"Fecha Evaluación     : >= {TRAIN_END_DATE}\n",
        encoding="utf-8",
    )

    baseline_run_id = None
    try:
        with mlflow.start_run(run_name="p2_baseline_dummy_stratified") as run_b:
            baseline_run_id = run_b.info.run_id
            mlflow.log_params(
                {
                    "model_class": "DummyClassifier",
                    "strategy": "stratified",
                    "random_seed": RANDOM_SEED,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "train_end_date": TRAIN_END_DATE,
                    "train_cutoff_date": TRAIN_CUTOFF_DATE,
                    "date_range": "2025-01-01 to 2026-09-30",
                    "data_version": data_version,
                }
            )
            mlflow.log_metrics(metrics_dummy)
            mlflow.set_tags(
                {
                    "equipo": "Equipo 30",
                    "pregunta": "P2",
                    "question": "p2",
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "model_family": "DummyClassifier",
                    "stage_of_work": "baseline",
                    "model_role": "baseline",
                    "author": "Ricardo Bahamon (Equipo 30)",
                    "target_horizon": "2_hours",
                    "problem_type": "multiclass_imbalanced",
                }
            )
            if baseline_summary_path.exists():
                mlflow.log_artifact(str(baseline_summary_path), artifact_path="reports")
            logger.info(f"✓ Run de Baseline registrado con ID: {baseline_run_id}")
    except Exception as exc:
        logger.warning(f"Aviso al registrar baseline en MLflow: {exc}")

    # 3. Preparación de candidatos a evaluar
    if lgbm_params is not None:
        candidates_to_run = {
            "lgbm_custom": {
                "description": "LightGBM con parámetros personalizados",
                "params": lgbm_params,
            }
        }
    else:
        candidates_to_run = P2_CANDIDATE_CONFIGS

    results_table = []
    candidate_runs = {}

    # 4. Ciclo de entrenamiento, evaluación y registro para cada candidato
    for cand_name, cand_info in candidates_to_run.items():
        logger.info("-" * 70)
        logger.info(f"Entrenando Candidato: '{cand_name}' - {cand_info['description']}")
        logger.info("-" * 70)

        params = cand_info["params"]
        clf = LGBMClassifier(**params)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_duration = time.time() - t0
        logger.info(f"✓ Ajuste completado en {fit_duration:.1f} segundos.")

        # Inferencia y medición de latencia
        t_inf0 = time.time()
        y_pred = clf.predict(X_test)
        predict_duration = time.time() - t_inf0
        latency_ms_per_1k = (predict_duration / len(X_test)) * 1000 * 1000

        cand_metrics = _evaluate_predictions(y_test, y_pred)
        cand_metrics["train_fit_duration_seconds"] = float(fit_duration)
        cand_metrics["test_predict_latency_ms_per_1k"] = float(latency_ms_per_1k)

        logger.info(
            f"Métricas Candidato '{cand_name}': "
            f"Macro F1={cand_metrics['test_macro_f1']:.4f} | "
            f"Bal Acc={cand_metrics['test_balanced_accuracy']:.4f} | "
            f"Recall Desabasto={cand_metrics['test_recall_desabasto']:.2%} | "
            f"Recall Saturación={cand_metrics['test_recall_saturacion']:.2%}"
        )

        # Artefactos diagnósticos por candidato
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        cand_cm_path = FIGURES_DIR / f"p2_confusion_matrix_{cand_name}.png"
        cand_fi_path = FIGURES_DIR / f"p2_feature_importance_{cand_name}.png"
        cand_rep_path = REPORTS_DIR / f"p2_classification_report_{cand_name}.txt"
        cand_cfg_path = REPORTS_DIR / f"p2_config_{cand_name}.json"

        # Generar figuras
        plot_confusion_matrix(y_test.values, y_pred, output_path=cand_cm_path)
        plot_feature_importance(clf.feature_importances_, feature_cols, output_path=cand_fi_path)

        rep_str = classification_report(
            y_test,
            y_pred,
            digits=4,
            target_names=["Normal (0)", "Desabasto (1)", "Saturación (2)"],
        )
        cand_rep_path.write_text(rep_str, encoding="utf-8")

        config_payload = {
            "candidate_id": cand_name,
            "description": cand_info["description"],
            "params": params,
            "train_cutoff_date": TRAIN_CUTOFF_DATE,
            "train_end_date": TRAIN_END_DATE,
            "target_horizon_hours": TARGET_HORIZON_HOURS,
            "random_seed": RANDOM_SEED,
            "git_commit": git_commit,
            "data_version": data_version,
        }
        cand_cfg_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # Registro en MLflow como Candidato formal
        run_name = f"p2_candidato_{cand_name}"
        cand_run_id = None
        try:
            with mlflow.start_run(run_name=run_name) as run_c:
                cand_run_id = run_c.info.run_id

                # Parámetros
                mlflow.log_params(params)
                mlflow.log_params(
                    {
                        "candidate_id": cand_name,
                        "model_class": "LGBMClassifier",
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "num_features": len(feature_cols),
                        "train_end_date": TRAIN_END_DATE,
                        "train_cutoff_date": TRAIN_CUTOFF_DATE,
                        "date_range": "2025-01-01 to 2026-09-30",
                        "data_version": data_version,
                    }
                )

                # Métricas
                mlflow.log_metrics(cand_metrics)

                # Etiquetas de linaje y gobernanza
                mlflow.set_tags(
                    {
                        "equipo": "Equipo 30",
                        "pregunta": "P2",
                        "question": "p2",
                        "git_commit": git_commit,
                        "git_dirty": git_dirty,
                        "model_family": "LGBMClassifier",
                        "stage_of_work": "candidate",
                        "model_role": "candidate",
                        "candidate_id": cand_name,
                        "author": "Ricardo Bahamon (Equipo 30)",
                        "target_horizon": "2_hours",
                        "problem_type": "multiclass_imbalanced",
                    }
                )

                # 1. Logged Model con firma e input_example
                input_example = X_test.head(5)
                signature = infer_signature(input_example, y_pred[:5])
                mlflow.sklearn.log_model(
                    clf,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    serialization_format="cloudpickle",
                )

                # 2. Input Dataset vía mlflow.log_input
                try:
                    dataset = mlflow.data.from_pandas(
                        X_train.head(1000), targets=y_train.head(1000), name="p2_features_train"
                    )
                    mlflow.log_input(dataset, context="training")
                except Exception as exc:
                    logger.debug(f"Aviso al registrar dataset en MLflow: {exc}")

                # 3. Artefactos de configuración, datos y gráficas
                if cand_cfg_path.exists():
                    mlflow.log_artifact(str(cand_cfg_path), artifact_path="config")
                if P2_DATA_MANIFEST_PATH.exists():
                    mlflow.log_artifact(str(P2_DATA_MANIFEST_PATH), artifact_path="data")
                if cand_cm_path.exists():
                    mlflow.log_artifact(str(cand_cm_path), artifact_path="figures")
                if cand_fi_path.exists():
                    mlflow.log_artifact(str(cand_fi_path), artifact_path="figures")
                if cand_rep_path.exists():
                    mlflow.log_artifact(str(cand_rep_path), artifact_path="reports")

                logger.success(f"✓ Candidato '{cand_name}' registrado exitosamente con run_id: {cand_run_id}")

        except Exception as exc:
            logger.warning(f"Aviso al registrar candidato '{cand_name}' en MLflow: {exc}")

        candidate_runs[cand_name] = {
            "run_id": cand_run_id,
            "clf": clf,
            "metrics": cand_metrics,
            "report_str": rep_str,
            "cm_path": cand_cm_path,
            "fi_path": cand_fi_path,
        }

        results_table.append(
            {
                "candidato": cand_name,
                "run_id": cand_run_id,
                "n_estimators": params.get("n_estimators"),
                "max_depth": params.get("max_depth"),
                "num_leaves": params.get("num_leaves"),
                "learning_rate": params.get("learning_rate"),
                "macro_f1": cand_metrics["test_macro_f1"],
                "balanced_acc": cand_metrics["test_balanced_accuracy"],
                "recall_normal": cand_metrics["test_recall_normal"],
                "recall_desabasto": cand_metrics["test_recall_desabasto"],
                "recall_saturacion": cand_metrics["test_recall_saturacion"],
                "precision_desabasto": cand_metrics["test_precision_desabasto"],
                "precision_saturacion": cand_metrics["test_precision_saturacion"],
                "fit_duration_s": cand_metrics["train_fit_duration_seconds"],
                "latency_ms_per_1k": cand_metrics["test_predict_latency_ms_per_1k"],
            }
        )

    # 5. Generación de tabla comparativa de candidatos
    P2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_comparison = pd.DataFrame(results_table)
    comp_csv_path = P2_REPORTS_DIR / "comparacion.csv"
    df_comparison.to_csv(comp_csv_path, index=False)
    # Copia en reports/ para acceso directo
    df_comparison.to_csv(REPORTS_DIR / "p2_comparacion.csv", index=False)
    logger.info(f"✓ Tabla comparativa guardada en: {comp_csv_path}")

    # 6. Selección del Champion (Criterio Multicriterio: Macro F1, Balanced Accuracy y Recall de Desabasto)
    # Ordenar por macro_f1 y balanced_acc
    df_sorted = df_comparison.sort_values(
        by=["macro_f1", "balanced_acc", "recall_desabasto"], ascending=False
    )
    best_row = df_sorted.iloc[0]
    best_cand_name = best_row["candidato"]
    best_info = candidate_runs[best_cand_name]
    best_run_id = best_row["run_id"]

    selection_reason = (
        f"Candidato '{best_cand_name}' seleccionado como Champion de P2: "
        f"Mayor Macro F1 ({best_row['macro_f1']:.4f}) y Balanced Accuracy ({best_row['balanced_acc']:.4f}) "
        f"sobre Test limpio, con Recall de Desabasto del {best_row['recall_desabasto']:.2%} y "
        f"Saturación del {best_row['recall_saturacion']:.2%}, superando al Baseline Estratificado "
        f"(Macro F1={metrics_dummy['test_macro_f1']:.4f}) en +{(best_row['macro_f1'] - metrics_dummy['test_macro_f1']):.4f} puntos. "
        f"Tiempo de entrenamiento eficiente ({best_row['fit_duration_s']:.1f}s) y latencia óptima "
        f"({best_row['latency_ms_per_1k']:.4f} ms/1k filas)."
    )

    selection_payload = {
        "selected_candidate": best_cand_name,
        "selected_run_id": best_run_id,
        "selection_reason": selection_reason,
        "metrics": best_info["metrics"],
        "baseline_metrics": metrics_dummy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_cutoff_date": TRAIN_CUTOFF_DATE,
        "git_commit": git_commit,
        "data_version": data_version,
    }
    selection_json_path = P2_REPORTS_DIR / "seleccion.json"
    selection_json_path.write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.success(f"✓ Decisión de selección guardada en: {selection_json_path}")

    # 7. Persistir el modelo Champion en `models/` y actualizar reportes principales
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_info["clf"], model_output_path)
    logger.success(f"✓ Modelo Champion serializado guardado en: {model_output_path}")

    # Actualizar figuras y reportes principales de P2 con los del champion
    plot_confusion_matrix(y_test.values, best_info["clf"].predict(X_test), output_path=P2_CONFUSION_MATRIX_PLOT)
    plot_feature_importance(best_info["clf"].feature_importances_, feature_cols, output_path=P2_FEATURE_IMPORTANCE_PLOT)
    (REPORTS_DIR / "p2_classification_report.txt").write_text(best_info["report_str"], encoding="utf-8")

    logger.info("=" * 70)
    logger.info("RESUMEN DE EXPERIMENTACIÓN Y SELECCIÓN (PREGUNTA 2):")
    logger.info(f"  Baseline Run ID : {baseline_run_id}")
    logger.info(f"  Champion Elegido: {best_cand_name} (Run ID: {best_run_id})")
    logger.info(f"  Macro F1 Score  : {best_row['macro_f1']:.4f}")
    logger.info(f"  Balanced Acc    : {best_row['balanced_acc']:.4f}")
    logger.info(f"  Recall Desabasto: {best_row['recall_desabasto']:.2%}")
    logger.info(f"  Recall Saturac. : {best_row['recall_saturacion']:.2%}")
    logger.info("=" * 70)

    return {
        "champion_candidate": best_cand_name,
        "champion_run_id": best_run_id,
        "comparison_table": df_comparison,
        "selection": selection_payload,
        "model_path": model_output_path,
    }


@app.command()
def main(
    n_estimators: Optional[int] = typer.Option(
        None, help="Sobreescribe n_estimators para un run individual de LightGBM."
    ),
    max_depth: Optional[int] = typer.Option(
        None, help="Sobreescribe max_depth para un run individual de LightGBM."
    ),
    learning_rate: Optional[float] = typer.Option(
        None, help="Sobreescribe learning_rate para un run individual de LightGBM."
    ),
    num_leaves: Optional[int] = typer.Option(
        None, help="Sobreescribe num_leaves para un run individual de LightGBM."
    ),
):
    """Ejecuta la experimentación avanzada o un run individual para la Pregunta 2."""
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
    logger.info(f"Ciclo de experimentación de P2 finalizado en {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    app()
