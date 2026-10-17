"""
Módulo de Inferencia y Diagnóstico Operativo de Predicciones.
============================================================
Carga el modelo serializado (models/lgbm_risk_classifier.joblib), genera inferencias
sobre el conjunto de prueba o nuevos lotes de datos, y presenta una inspección cualitativa
comparando el valor predicho contra el valor esperado.

Ejecución desde terminal:
    python -m MiBici.modeling.predict
    python -m MiBici.modeling.predict --sample-size 5
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer

from MiBici.config import (
    BEST_MODEL_PATH,
    INTERIM_STATIONS_PARQUET,
    LABEL_MAPPING,
    PROCESSED_FEATURES_TARGET_PARQUET,
    TEST_PREDICTIONS_PARQUET,
    TRAIN_END_DATE,
    logger,
)

app = typer.Typer(help="CLI para Inferencia de Riesgo en Estaciones.")


def run_inference(
    model_path: Path = BEST_MODEL_PATH,
    dataset_path: Path = PROCESSED_FEATURES_TARGET_PARQUET,
    stations_path: Path = INTERIM_STATIONS_PARQUET,
    output_predictions_path: Path = TEST_PREDICTIONS_PARQUET,
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    Carga el modelo entrenado y ejecuta inferencia sobre el periodo de test,
    mostrando ejemplos representativos del valor predicho vs valor esperado.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {model_path}. "
            "Ejecute primero 'python -m MiBici.modeling.train'."
        )

    logger.info(f"Cargando modelo serializado desde: {model_path}")
    clf = joblib.load(model_path)

    logger.info(f"Cargando dataset para inferencia desde: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # Identificar columnas de características
    excluded = {"station_id", "timestamp", "salidas", "llegadas", "balance_neto", "delta_net", "ratio", "target"}
    feature_cols = [c for c in df.columns if c not in excluded]

    # Filtrar conjunto de test
    test_mask = (df["timestamp"] >= TRAIN_END_DATE) & (df["target"] != -1) & (df["salidas_lag_168"].notna())
    df_test = df.loc[test_mask].copy()

    logger.info(f"Generando inferencia para {len(df_test):,} registros del periodo Test...")
    X_test = df_test[feature_cols]
    y_test = df_test["target"].values

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Construcción del DataFrame de resultados
    df_results = df_test[["station_id", "timestamp", "capacity", "delta_net", "ratio"]].copy()
    df_results["valor_esperado"] = y_test
    df_results["valor_predicho"] = y_pred

    df_results["etiqueta_esperada"] = df_results["valor_esperado"].map(LABEL_MAPPING)
    df_results["etiqueta_predicha"] = df_results["valor_predicho"].map(LABEL_MAPPING)

    df_results["prob_normal"] = np.round(y_proba[:, 0], 3)
    df_results["prob_desabasto"] = np.round(y_proba[:, 1], 3)
    df_results["prob_saturacion"] = np.round(y_proba[:, 2], 3)

    # Cruzar con nombre de estaciones si existe el catálogo
    if stations_path.exists():
        df_stations = pd.read_parquet(stations_path)
        df_results = df_results.merge(
            df_stations[["station_id", "name"]], on="station_id", how="left"
        )
        df_results.rename(columns={"name": "nombre_estacion"}, inplace=True)
    else:
        df_results["nombre_estacion"] = "Estación #" + df_results["station_id"].astype(str)

    # Diagnóstico operativo
    df_results["diagnostico"] = np.where(
        df_results["valor_esperado"] == df_results["valor_predicho"],
        "✓ Acierto",
        np.where(
            (df_results["valor_esperado"] == 0) & (df_results["valor_predicho"] != 0),
            "⚠️ Alerta Preventiva (FP)",
            "❌ No Detectado (FN)",
        ),
    )

    # Guardar predicciones en Parquet
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(output_predictions_path, index=False)
    logger.success(f"✓ Predicciones completas guardadas en: {output_predictions_path}")

    # Mostrar muestra de 5 ejemplos representativos
    ejemplos_representativos = pd.concat([
        df_results[df_results["valor_esperado"] == 0].sample(1, random_state=42),
        df_results[(df_results["valor_esperado"] == 1) & (df_results["valor_predicho"] == 1)].sample(2, random_state=42),
        df_results[(df_results["valor_esperado"] == 2) & (df_results["valor_predicho"] == 2)].sample(2, random_state=42),
    ]).reset_index(drop=True)

    cols_show = [
        "timestamp",
        "station_id",
        "nombre_estacion",
        "capacity",
        "delta_net",
        "ratio",
        "etiqueta_esperada",
        "etiqueta_predicha",
        "prob_normal",
        "prob_desabasto",
        "prob_saturacion",
        "diagnostico",
    ]

    print("\n" + "=" * 115)
    print("5 EJEMPLOS DE TEST: VALOR ESPERADO (REAL) VS. VALOR PREDICHO (LIGHTGBM)")
    print("=" * 115)
    print(ejemplos_representativos[cols_show].to_string(index=True))
    print("=" * 115 + "\n")

    return df_results


@app.command()
def main(
    sample_size: int = typer.Option(
        5, help="Número de ejemplos representativos a inspeccionar en consola."
    ),
):
    """Ejecuta inferencia con el modelo serializado e inspecciona ejemplos cualitativos."""
    run_inference(sample_size=sample_size)


if __name__ == "__main__":
    app()
