#!/usr/bin/env python3
"""
Módulo de Validación de Datos y Detección de Drift con DeepChecks.
==================================================================
Ejecuta suites de validación automatizada de datos para el proyecto MiBici:
1. Suite de Integridad de Datos (data_integrity):
   - Tipos de datos consistentes, columnas con nulos inesperados, filas duplicadas
   - Detección de outliers y clases desbalanceadas en el dataset procesado de P2.
2. Suite de Validación Train vs. Test y Drift (train_test_validation):
   - Deriva de distribución (Feature Drift) entre 2025 (entrenamiento) y 2026 (prueba)
   - Deriva de etiquetas (Target Drift) y verificación de fuga entre particiones.
3. Exportación de reportes interactivos HTML en `reports/` y resumen JSON auditable.

Uso:
    python scripts/run_deepchecks.py
    python scripts/run_deepchecks.py --sample-size 15000
"""

import json
from pathlib import Path
import time
from typing import Optional

# Compatibilidad con NumPy 2.x para dependencias internas de DeepChecks
import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "bool"):
    np.bool = bool

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
from loguru import logger
import pandas as pd
import typer

from src.config import PROJ_ROOT, REPORTS_DIR
from src.p2_config import P2_FEATURES_TARGET_PARQUET, TRAIN_CUTOFF_DATE

app = typer.Typer(
    help="CLI para validación de datos e integridad con DeepChecks (MiBici Guadalajara)."
)


def load_dataset_samples(
    parquet_path: Path, sample_size: int = 15000, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Carga y particiona temporalmente una muestra representativa de los datos de P2."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos procesados en: {parquet_path}. "
            "Ejecute primero: python -m src.features.p2_features"
        )

    logger.info(f"Cargando dataset desde: {parquet_path}...")
    # Cargar y excluir el periodo inicial de calentamiento (168h de rezagos)
    df = pd.read_parquet(parquet_path).dropna(subset=["salidas_lag_168", "balance_rolling_12h"])

    # Columnas numéricas de características
    drop_cols = ["timestamp", "station_id", "target", "risk_ratio", "net_flow_target"]
    feature_cols = [
        c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Partición temporal respetando el corte anti-leakage del proyecto
    train_mask = df["timestamp"] <= TRAIN_CUTOFF_DATE
    test_mask = df["timestamp"] >= "2026-01-01"

    df_train = df[train_mask]
    df_test = df[test_mask]

    # Muestreo estratificado/aleatorio representativo para velocidad y cobertura
    half_sample = sample_size // 2
    if len(df_train) > half_sample:
        df_train_sample = df_train.sample(n=half_sample, random_state=random_state)
    else:
        df_train_sample = df_train

    if len(df_test) > half_sample:
        df_test_sample = df_test.sample(n=half_sample, random_state=random_state)
    else:
        df_test_sample = df_test

    logger.info(
        f"Muestras preparadas: Train (2025) = {len(df_train_sample):,} filas | "
        f"Test (2026) = {len(df_test_sample):,} filas | Features = {len(feature_cols)}"
    )

    return df_train_sample, df_test_sample, feature_cols


def run_data_integrity_suite(
    df: pd.DataFrame, feature_cols: list[str], output_html: Path
) -> dict:
    """Ejecuta la suite de integridad de datos y genera el reporte HTML."""
    logger.info("Iniciando Suite de Integridad de Datos (data_integrity)...")
    ds = Dataset(df[feature_cols + ["target"]], label="target", cat_features=[])

    suite = data_integrity()
    suite_result = suite.run(ds)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    if output_html.exists():
        output_html.unlink()
    suite_result.save_as_html(str(output_html))
    logger.success(f"✓ Reporte de Integridad guardado en: {output_html}")

    passed_checks = [c.check.name() for c in suite_result.get_passed_checks()]
    not_passed_checks = [c.check.name() for c in suite_result.get_not_passed_checks()]

    return {
        "suite_name": "data_integrity",
        "total_checks": len(suite_result.results),
        "passed_checks_count": len(passed_checks),
        "passed_checks": passed_checks,
        "not_passed_checks": not_passed_checks,
        "all_passed": bool(suite_result.passed()),
        "report_path": str(output_html.relative_to(PROJ_ROOT)),
    }


def run_train_test_drift_suite(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    output_html: Path,
) -> dict:
    """Ejecuta la suite de validación train vs test (drift) y genera el reporte HTML."""
    logger.info(
        "Iniciando Suite de Deriva de Datos y Validación Train/Test (train_test_validation)..."
    )
    train_ds = Dataset(df_train[feature_cols + ["target"]], label="target", cat_features=[])
    test_ds = Dataset(df_test[feature_cols + ["target"]], label="target", cat_features=[])

    suite = train_test_validation()
    suite_result = suite.run(train_dataset=train_ds, test_dataset=test_ds)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    if output_html.exists():
        output_html.unlink()
    suite_result.save_as_html(str(output_html))
    logger.success(f"✓ Reporte de Data Drift guardado en: {output_html}")

    passed_checks = [c.check.name() for c in suite_result.get_passed_checks()]
    not_passed_checks = [c.check.name() for c in suite_result.get_not_passed_checks()]

    return {
        "suite_name": "train_test_validation",
        "total_checks": len(suite_result.results),
        "passed_checks_count": len(passed_checks),
        "passed_checks": passed_checks,
        "not_passed_checks": not_passed_checks,
        "all_passed": bool(suite_result.passed()),
        "report_path": str(output_html.relative_to(PROJ_ROOT)),
    }


@app.command()
def main(
    sample_size: int = typer.Option(
        15000, help="Número total de observaciones a muestrear (50% train, 50% test)."
    ),
    data_path: Optional[Path] = typer.Option(
        None, help="Ruta opcional al archivo parquet procesado de P2."
    ),
):
    """Ejecuta las suites completas de DeepChecks sobre el dataset de MiBici."""
    t0 = time.time()
    parquet_path = data_path or P2_FEATURES_TARGET_PARQUET
    integrity_html = REPORTS_DIR / "deepchecks_data_integrity.html"
    drift_html = REPORTS_DIR / "deepchecks_data_drift.html"
    summary_json = REPORTS_DIR / "deepchecks_summary.json"

    df_train, df_test, feature_cols = load_dataset_samples(
        parquet_path=parquet_path, sample_size=sample_size
    )

    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # 1. Integridad
    integrity_summary = run_data_integrity_suite(
        df=df_combined, feature_cols=feature_cols, output_html=integrity_html
    )

    # 2. Train/Test Drift
    drift_summary = run_train_test_drift_suite(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        output_html=drift_html,
    )

    elapsed = time.time() - t0
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(parquet_path.name),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "features_evaluated": len(feature_cols),
        "elapsed_seconds": round(elapsed, 2),
        "suites": {
            "data_integrity": integrity_summary,
            "train_test_drift": drift_summary,
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logger.success(f"✓ Resumen auditable de DeepChecks guardado en: {summary_json}")
    logger.info(f"Proceso de DeepChecks finalizado con éxito en {elapsed:.1f}s.")


if __name__ == "__main__":
    app()
