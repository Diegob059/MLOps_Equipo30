"""
Módulo de Configuración Centralizada para la Pregunta 3 (Clustering de Estaciones).
===================================================================================
Centraliza rutas, constantes, nombres del Model Registry, aliases y tracking de MLflow
para el clustering no supervisado de estaciones MiBici Guadalajara.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Carga de variables de entorno desde .env si existe (homologado con p1_config y p2_config)
load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("MiBici_P3")
    logger.success = logger.info  # type: ignore[attr-defined]

# Rutas del proyecto
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJ_ROOT / "models"
REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
REFERENCES_DIR: Path = PROJ_ROOT / "references"

# Rutas y artefactos específicos de P3
P3_REPORTS_DIR: Path = REPORTS_DIR / "p3"
P3_MODEL_PATH: Path = MODELS_DIR / "p3_cluster_model.pkl"
P3_CLUSTERS_PARQUET: Path = PROCESSED_DATA_DIR / "p3_estaciones_clusters.parquet"
P3_HOURLY_FEATURES_PARQUET: Path = PROCESSED_DATA_DIR / "estaciones_caracteristicas_hourly.parquet"
P3_BUCKETED_FEATURES_PARQUET: Path = PROCESSED_DATA_DIR / "estaciones_caracteristicas_bucketed.parquet"

# Configuración de MLflow
MLFLOW_EXPERIMENT_NAME: str = "mibici-equipo30"
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

P3_REGISTRY_NAME: str = "mibici-equipo30-p3-grupos"
P3_PRODUCTION_ALIAS: str = "champion"

# Metadatos del modelo registrado
P3_REGISTERED_MODEL_DESCRIPTION: str = (
    "P3: Clustering no supervisado de estaciones MiBici Guadalajara según su dinámica horaria de uso "
    "(salidas y llegadas). Segmenta las estaciones en perfiles operativamente diferenciados (origen residencial "
    "vs. destino laboral/comercial) para apoyar la planeación logística de camiones de rebalanceo y "
    "mantenimiento. El alias 'champion' identifica la versión activa en producción para el ejercicio académico."
)

# Hiperparámetros y conjuntos de features explorados
CONJUNTOS_CARACTERISTICAS = ("hourly", "bucketed")
ALGORITMOS = ("kmeans", "gmm")
K_MINIMO, K_MAXIMO = 2, 10
K_BASELINE = 4
RANDOM_SEED = 42

METADATA_COLS = [
    "id_estacion",
    "codigo",
    "nombre",
    "latitud",
    "longitud",
    "capacidad",
    "latitude",
    "longitude",
    "name",
    "dpcapacity",
    "total",
    "salidas",
    "llegadas",
]
