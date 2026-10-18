"""
Módulo de Inferencia para la Pregunta 3 (Clustering de Estaciones).
====================================================================
Carga el modelo de clustering guardado por `src/models/p3_train.py`
(escalador + modelo + columnas de features) y asigna un cluster a cada
estación de una tabla de features (por defecto, la del conjunto de features
con el que se entrenó el modelo candidato).

Ejecución desde terminal:
    python -m src.models.p3_predict
"""

from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from src.config import PROCESSED_DATA_DIR
from src.models.p3_train import P3_MODEL_PATH

app = typer.Typer(help="CLI de inferencia de clusters de estación (Pregunta 3).")


def predict(features_path: Path, model_path: Path = P3_MODEL_PATH) -> pd.DataFrame:
    """Asigna un cluster a cada estación de `features_path` usando el modelo guardado en `model_path`."""
    artefactos = joblib.load(model_path)
    df = pd.read_parquet(features_path)
    X = artefactos["escalador"].transform(df[artefactos["columnas_caracteristicas"]])
    df = df.copy()
    df["cluster"] = artefactos["modelo"].predict(X)
    return df


@app.command()
def main(
    features_path: Path = typer.Option(None, help="Parquet con features de estación a predecir."),
    model_path: Path = P3_MODEL_PATH,
    predictions_path: Path = PROCESSED_DATA_DIR / "p3_test_predictions.parquet",
):
    """CLI: predice los clusters de una tabla de features y guarda el resultado."""
    if features_path is None:
        artefactos = joblib.load(model_path)
        features_path = PROCESSED_DATA_DIR / f"estaciones_caracteristicas_{artefactos['conjunto_caracteristicas']}.parquet"

    logger.info(f"Prediciendo clusters con {model_path.name} sobre {features_path.name}")
    df = predict(features_path, model_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(predictions_path)
    logger.success(f"Predicciones guardadas: {predictions_path} ({len(df)} estaciones)")


if __name__ == "__main__":
    app()
