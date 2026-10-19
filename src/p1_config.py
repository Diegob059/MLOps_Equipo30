"""Rutas y decisiones del pronóstico de inventario inicial de P1."""
from pathlib import Path
import os

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
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f'sqlite:///{ROOT / "mlflow.db"}')
START = '2025-01-01'
END = '2026-09-30'
TRAIN_START = '2025-02-01'
VALIDATION_START = '2026-01-01'
TEST_START = '2026-07-01'
HOURS = 12
PEAK_DURATION = 3
MARGIN_QUANTILE = 0.90
TREE_PARAMS = {'max_depth': 10, 'min_samples_leaf': 100, 'random_state': 42}
VARIABLES = ['hora', 'dia_semana', 'mes', 'salidas_ayer', 'salidas_semana',
             'llegadas_ayer', 'llegadas_semana']
FLOWS = ['salidas', 'llegadas']
METHODS = ['semanal', 'historico', 'arbol']
