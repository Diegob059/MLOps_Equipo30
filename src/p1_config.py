"""Rutas y decisiones del pronóstico de inventario inicial de P1."""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv

# Carga variables de entorno desde .env si existe (homologado con src/config.py y src/p2_config.py).
# Garantiza que sobreescrituras locales como MLFLOW_TRACKING_URI (ej. puerto 5001 en macOS
# para evitar colisión con AirPlay en 5000) sean leídas en cualquier invocación independiente
# de scripts, CLI o pruebas unitarias de P1 sin requerir export manual en terminal.
load_dotenv()

if sys.platform == "win32":
    # Evita UnicodeEncodeError en consolas Windows (cp1252) cuando MLflow
    # imprime el emoji del enlace de la corrida al registrar contra un
    # tracking server HTTP. Sin esto, una corrida puede quedar marcada
    # "RUNNING" en vez de "FINISHED".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data/raw/Viajes'
CATALOG = ROOT / 'data/raw/Nomenclatura/nomenclatura_2026_09.csv'
HOURLY = ROOT / 'data/interim/p1_flujos_hora.parquet'
FEATURES = ROOT / 'data/processed/p1_features.parquet'
MANIFEST = ROOT / 'references/p1_data_manifest.json'
MODEL = ROOT / 'models/p1_inventario_inicial.joblib'
REPORTS = ROOT / 'reports/p1'
FIGURES = ROOT / 'reports/figures/p1'
EXPERIMENT = 'mibici-equipo30'
# Apunta al servidor local de MLflow (ver `make mlflow`:
# `mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000`),
# requerido para que el Model Registry funcione. El servidor debe estar
# corriendo antes de ejecutar cualquier entrenamiento.
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
START = '2025-01-01'
END = '2026-09-30'
TRAIN_START = '2025-02-01'
TRAIN_END = '2025-12-31'
VALIDATION_START = '2026-01-01'
VALIDATION_END = '2026-06-30'
TEST_START = '2026-07-01'
HOURS = 12
PEAK_DURATION = 3
MARGIN_QUANTILE = 0.90
TREE_PARAMS = {'max_depth': 10, 'min_samples_leaf': 100, 'random_state': 42}
VARIABLES = ['hora', 'dia_semana', 'mes', 'salidas_ayer', 'salidas_semana',
             'llegadas_ayer', 'llegadas_semana']
FLOWS = ['salidas', 'llegadas']
METHODS = ['semanal', 'historico', 'arbol']
TREE_GRID = {'max_depth': [6, 10], 'min_samples_leaf': [50, 100]}
ADVANCED_REPORTS = REPORTS / 'avanzado'
REGISTRY_NAME = 'mibici-equipo30-p1-demanda'
PRODUCTION_ALIAS = 'champion'
