"""
Módulo de Ingeniería de Características (Feature Store) para MiBici Guadalajara.
================================================================================
Traslada y modulariza todas las transformaciones de series de tiempo para responder
a la Pregunta 2 de MiBici:
"¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 horas?"

Principios de Diseño:
    - Zero Data Leakage: Toda variable predictora utiliza estrictamente información pasada (t <= 0).
    - Tipado eficiente en memoria con downcasting (int16, float32, int8).
    - Funciones puras, desacopladas y reutilizables tanto en batch como en inferencia.

Ejecución desde terminal:
    python -m MiBici.features
    python -m MiBici.features --help
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from MiBici.config import (
    DEMAND_LAGS,
    END_TIMESTAMP,
    INTERIM_STATIONS_PARQUET,
    INTERIM_TRIPS_HOURLY_PARQUET,
    OVERFLOW_RATIO_THRESHOLD,
    PROCESSED_FEATURES_TARGET_PARQUET,
    ROLLING_WINDOWS,
    START_TIMESTAMP,
    TARGET_HORIZON_HOURS,
    TRAIN_END_DATE,
    UNDERFLOW_RATIO_THRESHOLD,
    WARMUP_HOURS,
    logger,
)

app = typer.Typer(
    help="CLI para Generación de Variables Predictoras y Target de MiBici."
)


def build_spatiotemporal_grid(
    df_stations: pd.DataFrame,
    df_hourly: pd.DataFrame,
    start_ts: str = START_TIMESTAMP,
    end_ts: str = END_TIMESTAMP,
) -> pd.DataFrame:
    """
    Construye la rejilla cartesiana continua (Estaciones x Horas) con imputación Zero-Filling.

    ¿Por qué es pertinente para la Pregunta 2?
    En movilidad compartida, las estaciones durante la madrugada o periodos de baja afluencia
    no registran transacciones en los logs abiertos. Si omitiéramos esas horas, la serie temporal
    se fragmentaría, rompiendo el cálculo de lags periódicos (t-24, t-168) y alterando la inercia
    de demanda. La ausencia de viaje es un cero físico real, no un valor faltante.

    Parámetros:
        df_stations: Catálogo normalizado de estaciones con id y capacidad.
        df_hourly: Registros de viajes agregados por estación y hora.
        start_ts: Timestamp inicial del horizonte histórico.
        end_ts: Timestamp final del horizonte histórico.

    Retorna:
        DataFrame ordenado con la rejilla espacio-temporal y balance neto instantáneo.
    """
    logger.info("Construyendo rejilla cartesiana espacio-temporal (Estaciones x Horas)...")
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
    df_grid["balance_neto"] = (df_grid["llegadas"] - df_grid["salidas"]).astype(np.int16)

    logger.success(
        f"✓ Rejilla generada: {len(df_grid):,} filas ({len(active_station_ids)} estaciones x {len(date_range)} horas)."
    )
    return df_grid


def compute_target_variable(
    df_grid: pd.DataFrame,
    horizon_hours: int = TARGET_HORIZON_HOURS,
    underflow_threshold: float = UNDERFLOW_RATIO_THRESHOLD,
    overflow_threshold: float = OVERFLOW_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """
    Calcula el flujo neto en un horizonte futuro de 2 horas (t+1, t+2) normalizado por la capacidad.

    ¿Por qué es pertinente para la Pregunta 2?
    Dado que MiBici no publica sensores de inventario en tiempo real minuto a minuto, la física
    del flujo neto permite modelar el riesgo inminente:
        Δ_net = sum(Llegadas[t+1, t+2] - Salidas[t+1, t+2])
        Ratio = Δ_net / Capacidad
    - Clase 1 (Riesgo Desabasto / Vacío): Ratio <= -0.50 (pierde >= 50% de anclajes netos).
    - Clase 2 (Riesgo Saturación / Lleno): Ratio >= 0.50 (recibe >= 50% de anclajes netos).
    - Clase 0 (Operación Normal): -0.50 < Ratio < 0.50.

    Nota de Fuga: Se aplica desplazamiento negativo por estación para mirar exclusivamente hacia el futuro.
    Las últimas 2 horas sin horizonte futuro se marcan con -1 para su descarte posterior.
    """
    logger.info(f"Calculando variable Target a {horizon_hours} horas hacia adelante...")
    fut_llegadas = (
        df_grid.groupby("station_id")["llegadas"].shift(-1)
        + df_grid.groupby("station_id")["llegadas"].shift(-2)
    )
    fut_salidas = (
        df_grid.groupby("station_id")["salidas"].shift(-1)
        + df_grid.groupby("station_id")["salidas"].shift(-2)
    )

    df_grid["delta_net"] = (fut_llegadas - fut_salidas).astype("float32")
    df_grid["ratio"] = (df_grid["delta_net"] / df_grid["capacity"]).astype("float32")

    conditions = [
        df_grid["ratio"] <= underflow_threshold,
        df_grid["ratio"] >= overflow_threshold,
    ]
    df_grid["target"] = np.select(conditions, [1, 2], default=0).astype(np.int8)

    # Horas sin futuro completo
    df_grid.loc[df_grid["ratio"].isna(), "target"] = -1

    dist = df_grid[df_grid["target"] != -1]["target"].value_counts(normalize=True).sort_index() * 100
    logger.info(
        f"Distribución del Target: Normal (0): {dist.get(0, 0):.2f}%, "
        f"Desabasto (1): {dist.get(1, 0):.2f}%, Saturación (2): {dist.get(2, 0):.2f}%"
    )
    return df_grid


def add_demand_lags(
    df_grid: pd.DataFrame,
    lags: List[int] = DEMAND_LAGS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Genera retardos (lags) multiescala para la demanda (salidas) y la oferta (llegadas).

    ¿Por qué es pertinente para la Pregunta 2?
    - Lags t-1 y t-2: Capturan la inercia y momentum inmediato de uso en la estación.
    - Lag t-24: Captura el patrón diario a la misma hora del día anterior (hora pico de oficina, entrada/salida escolar).
    - Lag t-168: Captura la estacionalidad semanal exacta (el comportamiento de un lunes no es igual al de un domingo).

    Garantía de Leakage: Todos los lags son desplazamientos positivos (shift(lag) con lag >= 1),
    utilizando información observable estrictamente en el pasado t <= 0.
    """
    logger.info(f"Generando lags de demanda y oferta para retardos: {lags}...")
    lag_cols = []
    for lag in lags:
        for col in ["salidas", "llegadas"]:
            col_name = f"{col}_lag_{lag}"
            df_grid[col_name] = (
                df_grid.groupby("station_id")[col].shift(lag).astype("float32")
            )
            lag_cols.append(col_name)
    return df_grid, lag_cols


def add_rolling_balance_features(
    df_grid: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcula medias móviles rodantes del balance neto (Llegadas - Salidas) en ventanas de 3h y 12h.

    ¿Por qué es pertinente para la Pregunta 2?
    Suaviza la volatilidad horaria y permite que el modelo perciba si una estación viene acumulando
    bicicletas de forma sostenida (riesgo de llenado) o desocupándose progresivamente (riesgo de vaciado).

    Garantía de Leakage: Se aplica OBLIGATORIAMENTE `.shift(1)` antes de `.rolling()`, asegurando
    que la observación de la hora actual 't' no contamine la media del pasado.
    """
    logger.info(f"Calculando medias móviles rodantes de balance neto (ventanas: {windows}h)...")
    rolling_cols = []
    for w in windows:
        col_name = f"balance_rolling_{w}h"
        df_grid[col_name] = df_grid.groupby("station_id")["balance_neto"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        ).astype("float32")
        rolling_cols.append(col_name)
    return df_grid, rolling_cols


def add_cyclical_calendar_features(
    df_grid: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Codifica las variables temporales (hora del día y día de la semana) usando funciones seno y coseno.

    ¿Por qué es pertinente para la Pregunta 2?
    Los modelos de árboles o lineales interpretan números enteros linealmente (23 y 0 parecerían extremos opuestos).
    La codificación trigonométrica proyecta el tiempo en un círculo continuo, enseñándole al algoritmo
    que las 23:00 y las 00:00 son horas contiguas, al igual que el domingo (día 6) y el lunes (día 0).
    """
    logger.info("Codificando variables temporales cíclicas (seno/coseno)...")
    h = df_grid["timestamp"].dt.hour
    dow = df_grid["timestamp"].dt.dayofweek

    df_grid["sin_hour"] = np.sin(2 * np.pi * h / 24).astype("float32")
    df_grid["cos_hour"] = np.cos(2 * np.pi * h / 24).astype("float32")
    df_grid["sin_dow"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    df_grid["cos_dow"] = np.cos(2 * np.pi * dow / 7).astype("float32")

    cyclical_cols = ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]
    return df_grid, cyclical_cols


def build_full_feature_pipeline(
    stations_parquet: Path = INTERIM_STATIONS_PARQUET,
    trips_parquet: Path = INTERIM_TRIPS_HOURLY_PARQUET,
    output_parquet: Path = PROCESSED_FEATURES_TARGET_PARQUET,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Orquesta la construcción integral del dataset de características y target sin fuga de información.
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("INICIO DEL PIPELINE DE FEATURE ENGINEERING")
    logger.info("=" * 70)

    if not stations_parquet.exists() or not trips_parquet.exists():
        raise FileNotFoundError(
            "Faltan artefactos intermedios en Parquet. Ejecute primero 'python -m MiBici.dataset'."
        )

    df_stations = pd.read_parquet(stations_parquet)
    df_hourly = pd.read_parquet(trips_parquet)

    # 1. Rejilla cartesiana con Zero-filling
    df = build_spatiotemporal_grid(df_stations, df_hourly)

    # 2. Target multiclase
    df = compute_target_variable(df)

    # 3. Lags
    df, lag_cols = add_demand_lags(df)

    # 4. Rollings
    df, rolling_cols = add_rolling_balance_features(df)

    # 5. Cíclicas
    df, cyclical_cols = add_cyclical_calendar_features(df)

    # 6. Características estáticas de la estación
    static_cols = ["capacity", "latitude", "longitude"]

    feature_cols = lag_cols + rolling_cols + cyclical_cols + static_cols
    logger.info(f"✓ Creadas {len(feature_cols)} variables predictoras estrictamente del pasado.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t_start
    logger.success(
        f"✓ Dataset completo procesado en Parquet: {output_parquet} ({len(df):,} filas, {elapsed:.1f}s)"
    )
    return df, feature_cols


def split_and_sanitize_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end_date: str = TRAIN_END_DATE,
    warmup_lag_col: str = "salidas_lag_168",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Limpia el periodo de calentamiento y realiza una partición temporal estricta (Train vs Test).

    Garantía de Leakage: La partición cronológica impide que datos del futuro en Test influyan
    sobre el entrenamiento de Train.
    """
    logger.info("Aplicando sanitización y partición temporal estricta...")
    valid = (df["target"] != -1) & (df[warmup_lag_col].notna())
    df_clean = df[valid]

    train_mask = df_clean["timestamp"] < train_end_date
    test_mask = df_clean["timestamp"] >= train_end_date

    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, "target"]

    X_test = df_clean.loc[test_mask, feature_cols]
    y_test = df_clean.loc[test_mask, "target"]

    logger.info(
        f"Partición completada -> Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras."
    )
    return X_train, y_train, X_test, y_test


@app.command()
def main():
    """Ejecuta el pipeline completo de Feature Engineering y guarda el dataset procesado."""
    build_full_feature_pipeline()


if __name__ == "__main__":
    app()
