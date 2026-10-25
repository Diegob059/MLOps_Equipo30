"""
Módulo de Ingeniería de Características (Feature Store) para la Pregunta 2.
===========================================================================
Modulariza todas las transformaciones de series de tiempo para responder
a la Pregunta 2 de MiBici:
"¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 horas?"

Principios de Diseño:
    - Zero Data Leakage: Toda variable predictora utiliza estrictamente información pasada (t <= 0).
    - Tipado eficiente en memoria con downcasting (int16, float32, int8).
    - Funciones puras, desacopladas y reutilizables tanto en batch como en inferencia.

Ejecución desde terminal:
    python -m src.features.p2_features
"""

from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import typer

from src.p2_config import (
    DEMAND_LAGS,
    END_TIMESTAMP,
    OVERFLOW_RATIO_THRESHOLD,
    P2_FEATURES_TARGET_PARQUET,
    P2_STATIONS_PARQUET,
    P2_TRIPS_HOURLY_PARQUET,
    ROLLING_WINDOWS,
    START_TIMESTAMP,
    TARGET_HORIZON_HOURS,
    TRAIN_END_DATE,
    UNDERFLOW_RATIO_THRESHOLD,
    logger,
)

app = typer.Typer(
    help="CLI para Generación de Variables Predictoras y Target de Pregunta 2 (MiBici)."
)


def build_spatiotemporal_grid(
    df_stations: pd.DataFrame,
    df_hourly: pd.DataFrame,
    start_ts: str = START_TIMESTAMP,
    end_ts: str = END_TIMESTAMP,
) -> pd.DataFrame:
    """
    Construye la rejilla cartesiana continua (Estaciones x Horas) con imputación Zero-Filling.
    """
    logger.info(
        "Construyendo rejilla cartesiana espacio-temporal (Estaciones x Horas)..."
    )
    active_station_ids = np.intersect1d(
        df_stations["station_id"].values,
        df_hourly["station_id"].unique(),
    )
    active_station_ids.sort()

    date_range = pd.date_range(start_ts, end_ts, freq="h")
    grid_idx = pd.MultiIndex.from_product(
        [active_station_ids, date_range], names=["station_id", "timestamp"]
    )
    df_grid = pd.DataFrame(index=grid_idx).reset_index()

    # Cruce con flujos observados
    df_grid = df_grid.merge(df_hourly, on=["station_id", "timestamp"], how="left")

    # Zero-filling obligatorio
    df_grid["salidas"] = df_grid["salidas"].fillna(0).astype(np.int16)
    df_grid["llegadas"] = df_grid["llegadas"].fillna(0).astype(np.int16)

    # Cruce con metadatos fijos de la estación
    df_grid = df_grid.merge(
        df_stations[["station_id", "capacity", "latitude", "longitude"]],
        on="station_id",
        how="left",
    )
    df_grid.sort_values(["station_id", "timestamp"], inplace=True)
    df_grid.reset_index(drop=True, inplace=True)

    # Balance instantáneo de la hora (Llegadas - Salidas)
    df_grid["balance_neto"] = (df_grid["llegadas"] - df_grid["salidas"]).astype(
        np.int16
    )

    logger.info(
        f"✓ Rejilla generada: {len(df_grid):,} filas ({len(active_station_ids)} estaciones x {len(date_range)} horas)."
    )
    return df_grid


def compute_target_variable(
    df: pd.DataFrame,
    horizon_hours: int = TARGET_HORIZON_HOURS,
    underflow_thresh: float = UNDERFLOW_RATIO_THRESHOLD,
    overflow_thresh: float = OVERFLOW_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """
    Calcula el target multiclase de riesgo futuro a 2 horas (t+1, t+2):
    Clase 0: Operación Normal
    Clase 1: Riesgo de Desabasto (Ratio <= -0.50)
    Clase 2: Riesgo de Saturación (Ratio >= 0.50)
    """
    logger.info(f"Calculando variable Target a {horizon_hours} horas hacia adelante...")
    df = df.copy()

    # Desplazamientos hacia adelante por estación
    fut_llegadas = df.groupby("station_id")["llegadas"].shift(-1) + df.groupby(
        "station_id"
    )["llegadas"].shift(-2)
    fut_salidas = df.groupby("station_id")["salidas"].shift(-1) + df.groupby(
        "station_id"
    )["salidas"].shift(-2)

    df["delta_net"] = (fut_llegadas - fut_salidas).astype("float32")
    df["ratio"] = (df["delta_net"] / df["capacity"]).astype("float32")

    conditions = [
        df["ratio"] <= underflow_thresh,
        df["ratio"] >= overflow_thresh,
    ]
    df["target"] = np.select(conditions, [1, 2], default=0).astype(np.int8)

    # Las últimas 2 horas de cada estación no tienen horizonte futuro observable
    df.loc[df["ratio"].isna(), "target"] = -1

    valid_mask = df["target"] != -1
    dist = df.loc[valid_mask, "target"].value_counts(normalize=True).sort_index() * 100
    logger.info(
        f"Distribución del Target: Normal (0): {dist.get(0, 0):.2f}%, "
        f"Desabasto (1): {dist.get(1, 0):.2f}%, Saturación (2): {dist.get(2, 0):.2f}%"
    )
    return df


def add_demand_lags(
    df: pd.DataFrame,
    lags: List[int] = DEMAND_LAGS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Genera características de retardo (lags) de salidas y llegadas."""
    logger.info(f"Generando lags de demanda y oferta para retardos: {lags}...")
    df = df.copy()
    lag_cols = []

    for lag in lags:
        for col in ["salidas", "llegadas"]:
            feat_name = f"{col}_lag_{lag}"
            df[feat_name] = df.groupby("station_id")[col].shift(lag).astype("float32")
            lag_cols.append(feat_name)

    return df, lag_cols


def add_rolling_balance_features(
    df: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Calcula medias móviles rodantes del balance neto pasado con shift(1) anti-leakage."""
    logger.info(
        f"Calculando medias móviles rodantes de balance neto (ventanas: {windows}h)..."
    )
    df = df.copy()
    rolling_cols = []

    for w in windows:
        feat_name = f"balance_rolling_{w}h"
        df[feat_name] = (
            df.groupby("station_id")["balance_neto"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            .astype("float32")
        )
        rolling_cols.append(feat_name)

    return df, rolling_cols


def add_cyclical_calendar_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Codifica hora del día y día de la semana mediante transformaciones trigonométricas seno/coseno."""
    logger.info("Codificando variables temporales cíclicas (seno/coseno)...")
    df = df.copy()
    h = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek

    df["sin_hour"] = np.sin(2 * np.pi * h / 24).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * h / 24).astype("float32")
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7).astype("float32")

    cyc_cols = ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]
    return df, cyc_cols


def build_full_feature_pipeline(
    stations_parquet: Path = P2_STATIONS_PARQUET,
    trips_parquet: Path = P2_TRIPS_HOURLY_PARQUET,
    output_parquet: Path = P2_FEATURES_TARGET_PARQUET,
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de Feature Store para P2 y persiste en Parquet."""
    t0 = time.time()
    if not stations_parquet.exists() or not trips_parquet.exists():
        raise FileNotFoundError(
            f"Faltan artefactos intermedios para P2 ({stations_parquet} o {trips_parquet}). "
            "Ejecuta primero: python -m src.data.p2_dataset"
        )

    df_stations = pd.read_parquet(stations_parquet)
    df_hourly = pd.read_parquet(trips_parquet)

    # 1. Rejilla cartesiana con Zero-filling
    df_grid = build_spatiotemporal_grid(df_stations, df_hourly)

    # 2. Target Operativo multiclase
    df_with_target = compute_target_variable(df_grid)

    # 3. Lags
    df_with_lags, lag_cols = add_demand_lags(df_with_target)

    # 4. Rollings
    df_with_roll, roll_cols = add_rolling_balance_features(df_with_lags)

    # 5. Cíclicas
    df_full, cyc_cols = add_cyclical_calendar_features(df_with_roll)

    feature_cols = (
        lag_cols + roll_cols + cyc_cols + ["capacity", "latitude", "longitude"]
    )
    logger.info(
        f"✓ Creadas {len(feature_cols)} variables predictoras estrictamente del pasado."
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t0
    logger.info(
        f"✓ Dataset completo procesado en Parquet: {output_parquet} ({len(df_full):,} filas, {elapsed:.1f}s)"
    )
    return df_full


def get_train_test_data(
    features_parquet: Path = P2_FEATURES_TARGET_PARQUET,
    split_date: str = TRAIN_END_DATE,
    horizon_hours: int = TARGET_HORIZON_HOURS,
    warmup_lag_col: str = "salidas_lag_168",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Carga el dataset de características y realiza la partición temporal estricta anti-leakage:
    - Descarta periodo de calentamiento y filas terminales sin target.
    - Train: timestamp < (split_date - horizon_hours) -> Excluye las filas cuyas ventanas futuras
      de 2 horas (t+1, t+2) cruzan el corte hacia el periodo de prueba de 2026 (las que empiezan
      a partir de las 22:00 del 31 de diciembre de 2025).
    - Test : timestamp >= split_date
    """
    if not features_parquet.exists():
        raise FileNotFoundError(
            f"No existe el archivo de características: {features_parquet}"
        )

    df = pd.read_parquet(features_parquet)

    # Filtrado estricto
    valid_mask = (df["target"] != -1) & (df[warmup_lag_col].notna())
    df_clean = df[valid_mask].copy()

    feature_cols = [
        c
        for c in df_clean.columns
        if c
        not in [
            "station_id",
            "timestamp",
            "salidas",
            "llegadas",
            "balance_neto",
            "delta_net",
            "ratio",
            "target",
        ]
    ]

    # Prevención estricta de fuga temporal (Data Leakage):
    # Las filas cuya ventana futura cruza el corte (a partir de las 22:00 del 31 de diciembre)
    # se excluyen de entrenamiento para que ninguna etiqueta use viajes de enero de 2026.
    train_cutoff = pd.Timestamp(split_date) - pd.Timedelta(hours=horizon_hours)
    train_mask = df_clean["timestamp"] < train_cutoff
    test_mask = df_clean["timestamp"] >= split_date

    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, "target"]

    X_test = df_clean.loc[test_mask, feature_cols]
    y_test = df_clean.loc[test_mask, "target"]

    return X_train, y_train, X_test, y_test, feature_cols, df_clean


@app.command()
def main():
    """Ejecuta la pipeline de Feature Store para P2 desde CLI."""
    logger.info("=" * 70)
    logger.info("INICIO DEL PIPELINE DE FEATURE ENGINEERING (PREGUNTA 2)")
    logger.info("=" * 70)
    build_full_feature_pipeline()


if __name__ == "__main__":
    app()
