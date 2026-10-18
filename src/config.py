import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Subdirectorios de datos crudos (compartidos entre preguntas)
RAW_NOMENCLATURA_DIR = RAW_DATA_DIR / "Nomenclatura"
RAW_VIAJES_DIR = RAW_DATA_DIR / "Viajes"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

# ==============================================================================
# INGESTA Y VERSIONADO DE DATOS (COMPARTIDO ENTRE PREGUNTAS)
# ==============================================================================
# Rango de fechas usado por el proyecto: enero 2025 a septiembre 2026, el
# periodo completo disponible en el portal de datos abiertos de MiBici al
# momento de iniciar el proyecto.
TARGET_YEARS = [2025, 2026]
START_TIMESTAMP = "2025-01-01 00:00:00"
END_TIMESTAMP = "2026-09-30 23:59:59"

# Manifiesto de versión de datos (hashes SHA-256 de los archivos crudos),
# compartido por cualquier pregunta que ingiera los mismos datos crudos.
DATA_MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"

# Artefactos intermedios producidos por `dataset.py` (viajes y estaciones ya
# limpios), reutilizables por `features.py` y por cualquier pregunta.
INTERIM_TRIPS_PARQUET = INTERIM_DATA_DIR / "viajes_limpios.parquet"
INTERIM_STATIONS_PARQUET = INTERIM_DATA_DIR / "estaciones.parquet"

# ==============================================================================
# REPRODUCIBILIDAD Y TRACKING CON MLFLOW (COMPARTIDO ENTRE PREGUNTAS)
# ==============================================================================
RANDOM_SEED = 42

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
# Experimento único del equipo: agrupa los runs de P1, P2 y P3.
MLFLOW_EXPERIMENT_NAME = "mibici-equipo30"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
# Nota: se usa `logger.remove()` sin id (no `logger.remove(0)`) porque loguru
# es un logger global compartido entre módulos; si `src/p2_config.py` ya
# removió el handler por defecto en este proceso, `logger.remove(0)` lanzaría
# ValueError al no encontrar ese id.
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
