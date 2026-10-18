"""
Módulo de Configuración Centralizada del Proyecto MiBici.
=========================================================
Centraliza todas las rutas absolutas/relativas del proyecto, parámetros de fechas,
umbrales de negocio, hiperparámetros de modelos, semillas aleatorias y configuración
de tracking para MLflow.
"""

from pathlib import Path
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("MiBici")
    logger.success = logger.info  # type: ignore[attr-defined]

# Carga de variables de entorno desde archivo .env si existe
load_dotenv()

# ==============================================================================
# RUTAS PRINCIPALES DEL PROYECTO (Cookiecutter Data Science)
# ==============================================================================
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# Subdirectorios de datos crudos
RAW_NOMENCLATURA_DIR: Path = RAW_DATA_DIR / "Nomenclatura"
RAW_VIAJES_DIR: Path = RAW_DATA_DIR / "Viajes"

# Modelos y artefactos serializados
MODELS_DIR: Path = PROJ_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "lgbm_risk_classifier.joblib"

# Reportes y visualizaciones
REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
CONFUSION_MATRIX_PLOT: Path = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PLOT: Path = FIGURES_DIR / "feature_importance.png"

# Documentación técnica y artefactos de arquitectura
REFERENCES_DIR: Path = PROJ_ROOT / "references"
DATA_MANIFEST_PATH: Path = REFERENCES_DIR / "data_manifest.json"
ML_CANVAS_PATH: Path = REFERENCES_DIR / "ml_canvas_pregunta2.md"
ARCHITECTURE_DIAGRAM_PATH: Path = REFERENCES_DIR / "architecture_diagram.md"

# Artefactos procesados en Parquet para alto rendimiento
INTERIM_STATIONS_PARQUET: Path = INTERIM_DATA_DIR / "estaciones.parquet"
INTERIM_TRIPS_HOURLY_PARQUET: Path = INTERIM_DATA_DIR / "viajes_agregados_hora.parquet"
PROCESSED_FEATURES_TARGET_PARQUET: Path = PROCESSED_DATA_DIR / "mibici_features_target.parquet"
TEST_PREDICTIONS_PARQUET: Path = PROCESSED_DATA_DIR / "test_predictions.parquet"

# ==============================================================================
# CONFIGURACIÓN DE INGESTA Y FECHAS
# ==============================================================================
PORTAL_URL: str = "https://www.mibici.net/es/datos-abiertos/"
TARGET_YEARS: List[int] = [2025, 2026]
START_TIMESTAMP: str = "2025-01-01 00:00:00"
END_TIMESTAMP: str = "2026-09-30 23:00:00"
TRAIN_END_DATE: str = "2026-01-01 00:00:00"

# ==============================================================================
# DEFINICIÓN DEL TARGET OPERATIVO (PREGUNTA 2)
# ==============================================================================
# Flujo neto en 2 horas hacia adelante: (Llegadas[t+1, t+2] - Salidas[t+1, t+2]) / Capacidad
TARGET_HORIZON_HOURS: int = 2
UNDERFLOW_RATIO_THRESHOLD: float = -0.50  # Clase 1: Riesgo Desabasto / Vacío
OVERFLOW_RATIO_THRESHOLD: float = 0.50   # Clase 2: Riesgo Saturación / Lleno

LABEL_MAPPING: Dict[int, str] = {
    0: "Normal (0)",
    1: "Desabasto (1)",
    2: "Saturación (2)",
}

# ==============================================================================
# INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================
DEMAND_LAGS: List[int] = [1, 2, 24, 168]  # t-1, t-2, t-24, t-168
ROLLING_WINDOWS: List[int] = [3, 12]       # Ventanas rodantes de balance neto
WARMUP_HOURS: int = 168                   # 1 semana de calentamiento para lags

# ==============================================================================
# SEMILLAS Y MODELADO
# ==============================================================================
RANDOM_SEED: int = 42

# Hiperparámetros optimizados para LightGBM con balanceo de clases
LGBM_PARAMS: Dict = {
    "objective": "multiclass",
    "num_class": 3,
    "class_weight": "balanced",
    "n_estimators": 120,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 7,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ==============================================================================
# TRACKING CON MLFLOW
# ==============================================================================
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_NAME: str = "MiBici-Pregunta2-Riesgo-Estaciones"

# Configuración de loguru con tqdm si está disponible
if hasattr(logger, "remove"):
    try:
        from tqdm import tqdm

        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    except ModuleNotFoundError:
        pass
