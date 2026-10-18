"""
Módulo de Inferencia Operativa y Diagnóstico Cualitativo (Pregunta 2).
=====================================================================
Carga el modelo serializado entrenado (`p2_lgbm_risk_classifier.joblib`), ejecuta inferencia
por lotes (Batch Scoring) sobre el conjunto de Test y formatea una tabla legible
con 5 ejemplos (valor predicho vs esperado con probabilidades de cada clase).

Ejecución desde terminal:
    python -m src.models.p2_predict --sample-size 5
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer

from src.features.p2_features import get_train_test_data
from src.p2_config import (
    LABEL_MAPPING,
    P2_FEATURES_TARGET_PARQUET,
    P2_MODEL_PATH,
    P2_STATIONS_PARQUET,
    P2_TEST_PREDICTIONS_PARQUET,
    TRAIN_END_DATE,
    logger,
)

app = typer.Typer(
    help="CLI de Inferencia en Batch y Diagnóstico de Predicciones para P2 (MiBici)."
)


def run_batch_inference(
    model_path: Path = P2_MODEL_PATH,
    features_parquet: Path = P2_FEATURES_TARGET_PARQUET,
    stations_parquet: Path = P2_STATIONS_PARQUET,
    output_predictions_parquet: Path = P2_TEST_PREDICTIONS_PARQUET,
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    Ejecuta inferencia sobre el conjunto de Test de 2026, guarda predicciones en Parquet
    y despliega una tabla formateada en consola con casos cualitativos.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo serializado en: {model_path}. "
            "Ejecuta primero: python -m src.models.p2_train"
        )

    logger.info(f"Cargando modelo serializado desde: {model_path}")
    clf = joblib.load(model_path)

    logger.info(f"Cargando dataset para inferencia desde: {features_parquet}")
    X_train, y_train, X_test, y_test, feature_cols, df_clean = get_train_test_data(
        features_parquet=features_parquet,
        split_date=TRAIN_END_DATE,
    )

    logger.info(
        f"Generando inferencia para {len(X_test):,} registros del periodo Test..."
    )
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Construcción de tabla de resultados
    df_results = df_clean.loc[
        X_test.index,
        ["station_id", "timestamp", "capacity", "delta_net", "ratio", "target"],
    ].copy()
    df_results["pred_target"] = y_pred
    df_results["prob_normal"] = y_proba[:, 0].round(3)
    df_results["prob_desabasto"] = y_proba[:, 1].round(3)
    df_results["prob_saturacion"] = y_proba[:, 2].round(3)

    # Cruce con nombres de estaciones
    if stations_parquet.exists():
        df_stations = pd.read_parquet(stations_parquet)
        df_results = df_results.merge(
            df_stations[["station_id", "name"]], on="station_id", how="left"
        )
        df_results.rename(columns={"name": "nombre_estacion"}, inplace=True)
    else:
        df_results["nombre_estacion"] = "Estación " + df_results["station_id"].astype(
            str
        )

    # Mapeo a etiquetas legibles
    df_results["etiqueta_esperada"] = df_results["target"].map(LABEL_MAPPING)
    df_results["etiqueta_predicha"] = df_results["pred_target"].map(LABEL_MAPPING)
    df_results["diagnostico"] = np.where(
        df_results["target"] == df_results["pred_target"], "✓ Acierto", "✗ Discrepancia"
    )

    output_predictions_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(output_predictions_parquet, index=False)
    logger.info(f"✓ Predicciones completas guardadas en: {output_predictions_parquet}")

    # Selección intencional de 5 ejemplos diversos para inspección visual
    cols_display = [
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

    sample_normal = df_results[df_results["target"] == 0].head(1)
    sample_desabasto = df_results[df_results["target"] == 1].head(2)
    sample_saturacion = df_results[df_results["target"] == 2].head(2)

    df_sample = pd.concat(
        [sample_normal, sample_desabasto, sample_saturacion]
    ).reset_index(drop=True)
    df_sample = df_sample[cols_display]

    print("\n" + "=" * 115)
    print(
        "5 EJEMPLOS DE TEST (P2): VALOR ESPERADO (REAL) VS. VALOR PREDICHO (LIGHTGBM)"
    )
    print("=" * 115)
    print(df_sample.to_string())
    print("=" * 115 + "\n")

    return df_results


@app.command()
def main(
    sample_size: int = typer.Option(
        5, help="Número de ejemplos a desplegar en consola."
    ),
):
    """Ejecuta la inferencia en batch para la Pregunta 2."""
    run_batch_inference(sample_size=sample_size)


if __name__ == "__main__":
    app()
