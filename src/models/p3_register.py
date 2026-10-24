"""
Módulo de Registro en el Model Registry de MLflow (Pregunta 3: Clustering).
============================================================================
Registra, en el Model Registry de MLflow, el modelo candidato de P3 que el
equipo eligió manualmente tras comparar las corridas en la interfaz (ver el
requisito "Seleccionar el mejor modelo": desempeño frente al baseline,
métrica de negocio, estabilidad y costo de entrenamiento/inferencia).

A propósito, este módulo NO se ejecuta automáticamente en cada entrenamiento
de `src/models/p3_train.py`: registrar un modelo es una decisión deliberada
del equipo, no algo que el código deba decidir solo (por ejemplo, eligiendo
siempre el candidato con mayor silueta). Se invoca explícitamente dando el
`run_id` de la corrida candidata elegida (visible en la UI de MLflow).

Ejecución desde terminal:
    python -m src.models.p3_register --run-id <run_id_de_la_corrida_candidata>
    python -m src.models.p3_register --run-id <run_id> --descripcion "..."
"""

import typer
import mlflow
from loguru import logger
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI

app = typer.Typer(help="CLI para registrar el modelo candidato elegido de P3 en el Model Registry de MLflow.")

MODEL_REGISTRY_NAME = "mibici-equipo30-p3-grupos"

REGISTERED_MODEL_DESCRIPTION = (
    "Modelo de clustering no supervisado de estaciones MiBici (Pregunta 3): agrupa "
    "estaciones por patrón horario de uso (salidas/llegadas) para apoyar decisiones de "
    "rebalanceo y mantenimiento diferenciadas por segmento. Cada versión corresponde a un "
    "candidato distinto (algoritmo, k y conjunto de features) seleccionado manualmente por "
    "el equipo tras comparar las corridas en la UI de MLflow. Ver `src/models/p3_train.py` "
    "para el detalle del barrido de hiperparámetros (K-Means/Gaussian Mixture, k=2..10, 2 "
    "conjuntos de features) del que salen los candidatos."
)


def registrar_modelo(
    run_id: str,
    descripcion_version: str = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    model_name: str = MODEL_REGISTRY_NAME,
) -> ModelVersion:
    """Registra el modelo de la corrida `run_id` como una nueva versión de `model_name`.

    El modelo debe haber sido registrado previamente en esa corrida con
    `mlflow.sklearn.log_model(..., name="model", ...)` (ver
    `p3_train._logear_modelo_y_datos_candidato`), lo cual solo ocurre en corridas
    candidatas, nunca en el baseline ni en los runs hijos de la búsqueda.

    Preserva el linaje de cada versión copiando, como etiquetas propias de la versión
    registrada, el experimento, la corrida de origen, el commit de git y la versión de
    datos -además del enlace nativo `run_id`/`source` que MLflow ya mantiene
    automáticamente entre una versión registrada y su corrida de origen-.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)

    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    # Linaje: experimento, corrida de origen, commit y versión de datos.
    client.set_model_version_tag(model_name, version.version, "experiment_id", run.info.experiment_id)
    client.set_model_version_tag(model_name, version.version, "source_run_id", run_id)
    client.set_model_version_tag(model_name, version.version, "git_commit", run.data.tags.get("git_commit", ""))
    client.set_model_version_tag(model_name, version.version, "data_version", run.data.params.get("data_version", ""))

    # Descripción del modelo registrado (a nivel de familia de versiones; idempotente).
    try:
        client.update_registered_model(model_name, description=REGISTERED_MODEL_DESCRIPTION)
    except Exception as exc:
        logger.warning(f"No se pudo actualizar la descripción del modelo registrado '{model_name}': {exc}")

    # Descripción específica de esta versión.
    if descripcion_version is None:
        silueta = run.data.metrics.get("test_silhouette_score")
        davies_bouldin = run.data.metrics.get("test_davies_bouldin_index")
        descripcion_version = (
            f"Candidato {run.data.params.get('algorithm')} (k={run.data.params.get('k')}) sobre el "
            f"conjunto de features '{run.data.params.get('feature_set')}'. "
            f"Silhouette={silueta:.4f}, Davies-Bouldin={davies_bouldin:.4f}. "
            f"Corrida de origen: {run_id} (experimento {run.info.experiment_id})."
        )
    client.update_model_version(model_name, version.version, description=descripcion_version)

    logger.success(f"Modelo registrado: '{model_name}' versión {version.version} (corrida de origen: {run_id})")
    return version


@app.command()
def main(
    run_id: str = typer.Option(
        ..., "--run-id", help="run_id de la corrida candidata elegida por el equipo (ver la UI de MLflow)."
    ),
    descripcion: str = typer.Option(
        None,
        "--descripcion",
        help="Descripción de esta versión (opcional; si se omite, se genera una automática con algoritmo/k/métricas).",
    ),
):
    """Registra en el Model Registry el modelo de la corrida `--run-id` elegida manualmente."""
    registrar_modelo(run_id=run_id, descripcion_version=descripcion)


if __name__ == "__main__":
    app()
