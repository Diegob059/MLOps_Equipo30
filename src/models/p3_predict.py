"""
Módulo de Inferencia de Clusters de Estaciones (Pregunta 3).
============================================================
Carga el modelo de clustering (escalador + modelo de clustering) ya sea desde
el Model Registry de MLflow (`models:/mibici-equipo30-p3-grupos@champion`) o
desde un archivo serializado local (`models/p3_cluster_model.pkl`), y asigna el
cluster operacional correspondiente a cada estación en la tabla de features.

Ejecución desde terminal:
    python -m src.models.p3_predict --model-uri models:/mibici-equipo30-p3-grupos@champion
    python -m src.models.p3_predict  # usa el archivo local serializado por defecto
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import typer

from src.p3_config import (
    METADATA_COLS,
    MLFLOW_TRACKING_URI,
    P3_CLUSTERS_PARQUET,
    P3_HOURLY_FEATURES_PARQUET,
    P3_MODEL_PATH,
    logger,
)

app = typer.Typer(help="CLI de inferencia de clusters de estación (Pregunta 3).")


def predict(
    features_path: Path = P3_HOURLY_FEATURES_PARQUET,
    model_path: Path = P3_MODEL_PATH,
    model_uri: Optional[str] = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> pd.DataFrame:
    """Asigna un cluster a cada estación de `features_path` usando `model_uri` o `model_path`."""
    df = pd.read_parquet(features_path)

    if model_uri:
        if not tracking_uri.startswith("http"):
            env_uri = os.getenv("MLFLOW_TRACKING_URI", "")
            if env_uri.startswith("http"):
                tracking_uri = env_uri
            else:
                tracking_uri = "http://127.0.0.1:5001"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Cargando modelo desde MLflow Model Registry URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        preds = model.predict(df[feature_cols])
        if hasattr(preds, "values"):
            preds = preds.values
        df = df.copy()
        df["cluster"] = np.asarray(preds).ravel().astype(int)
    else:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo local en: {model_path}. "
                "Especifique --model-uri o ejecute primero: python -m src.models.p3_train"
            )
        logger.info(f"Cargando modelo serializado local desde: {model_path}")
        artefactos = joblib.load(model_path)
        X = artefactos["escalador"].transform(df[artefactos["columnas_caracteristicas"]])
        df = df.copy()
        df["cluster"] = artefactos["modelo"].predict(X)

    return df


@app.command()
def main(
    features_path: Optional[Path] = typer.Option(
        None, help="Ruta al Parquet con features de estación a predecir."
    ),
    model_path: Path = typer.Option(
        P3_MODEL_PATH, help="Ruta al archivo local del modelo serializado (.pkl)."
    ),
    model_uri: Optional[str] = typer.Option(
        None,
        "--model-uri",
        help="URI del modelo en MLflow (ej: models:/mibici-equipo30-p3-grupos@champion).",
    ),
    predictions_path: Path = typer.Option(
        P3_CLUSTERS_PARQUET, help="Ruta donde se guardará el Parquet con los clusters asignados."
    ),
    sample_size: int = typer.Option(
        5, help="Número de estaciones de ejemplo a mostrar en consola."
    ),
):
    """CLI: predice clusters de estaciones y persiste el resultado en parquet."""
    if features_path is None:
        if model_uri:
            features_path = P3_HOURLY_FEATURES_PARQUET
        else:
            try:
                artefactos = joblib.load(model_path)
                conjunto = artefactos.get("conjunto_caracteristicas", "hourly")
                features_path = Path(str(P3_HOURLY_FEATURES_PARQUET).replace("hourly", conjunto))
            except Exception:
                features_path = P3_HOURLY_FEATURES_PARQUET

    logger.info(f"Ejecutando inferencia sobre {features_path.name} ({len(pd.read_parquet(features_path))} estaciones)")
    df_clustered = predict(
        features_path=features_path,
        model_path=model_path,
        model_uri=model_uri,
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df_clustered.to_parquet(predictions_path)
    logger.success(f"✓ Predicciones guardadas exitosamente en: {predictions_path}")

    # Distribución de clusters
    distribucion = df_clustered["cluster"].value_counts().sort_index()
    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE ESTACIONES POR CLUSTER OPERACIONAL (P3)")
    print("=" * 60)
    for cluster_id, count in distribucion.items():
        pct = (count / len(df_clustered)) * 100
        desc = (
            "Origen neto residencial (pico de salidas matutino)"
            if cluster_id == 0
            else "Destino neto laboral/comercial (pico de llegadas matutino)"
        )
        print(f" Cluster {cluster_id}: {count:3d} estaciones ({pct:5.1f}%) -> {desc}")
    print("=" * 60)

    # Muestra de estaciones representativas
    df_display = df_clustered.reset_index() if "estacion_id" not in df_clustered.columns else df_clustered.copy()
    candidate_cols = ["estacion_id", "id_estacion", "name", "nombre", "dpcapacity", "capacidad", "salidas", "llegadas", "cluster"]
    cols_preview = [c for c in candidate_cols if c in df_display.columns]
    print(f"\nMUESTRA DE {sample_size} ESTACIONES CON CLUSTER ASIGNADO:")
    print(df_display[cols_preview].head(sample_size).to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    app()
