"""
Módulo de Ingeniería de Características (Compartido).
=======================================================
Punto único de *feature engineering* del proyecto (ver instrucciones de la
actividad: "...funciones reutilizables en el archivo `features.py`..."). Lee
los artefactos limpios que produce ``src/data/dataset.py`` en
``data/interim/`` y construye un perfil de comportamiento por estación,
reutilizable por cualquier pregunta que necesite caracterizar estaciones
(actualmente, la Pregunta 3 - clustering - en ``src/models/p3_train.py``).

Las variables se calculan sobre el periodo histórico completo disponible
(2025-01 a 2026-09): como no se trata de una tarea de pronóstico con un
"momento de predicción" (no hay fuga de información hacia el futuro posible
en una tarea de agrupamiento descriptivo, no hay recorte temporal train/test
que respetar aquí).

Ejecución desde terminal:
    python -m src.features.features
"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.config import (
    INTERIM_STATIONS_PARQUET,
    INTERIM_TRIPS_PARQUET,
    PROCESSED_DATA_DIR,
)

app = typer.Typer(help="CLI de ingeniería de características de estaciones de MiBici (compartido).")

# Hora -> franja horaria usada por el perfil agregado por franjas.
FRANJAS_HORARIAS: dict[str, tuple[int, ...]] = {
    "madrugada": (0, 1, 2, 3, 4, 5),
    "pico_am": (6, 7, 8, 9),
    "medio_dia": (10, 11, 12, 13, 14, 15),
    "pico_pm": (16, 17, 18, 19),
    "noche": (20, 21, 22, 23),
}

# Umbral de actividad mínima para que las proporciones horarias de una
# estación sean estadísticamente estables (estaciones con pocas salidas
# producen perfiles horarios ruidosos).
MIN_SALIDAS = 500

# Columnas que describen/dan contexto a una estación pero no son variables de
# entrada a un modelo (volumen crudo, ubicación, nombre).
METADATA_COLS: list[str] = ["name", "latitude", "longitude", "dpcapacity", "salidas", "llegadas", "total"]

SUMMARY_FEATURE_COLS: list[str] = ["log_total", "balance_rel", "rotacion_por_anclaje", "duracion_mediana"]


def load_clean_trips(path: Path = INTERIM_TRIPS_PARQUET) -> pd.DataFrame:
    """Carga la tabla de viajes limpia escrita por `src/data/dataset.py`."""
    trips = pd.read_parquet(path)
    logger.info(f"Viajes cargados desde {path}: {len(trips):,}")
    return trips


def load_clean_stations(path: Path = INTERIM_STATIONS_PARQUET) -> pd.DataFrame:
    """Carga el catálogo de estaciones (ya restringido a estaciones comunes a todos los meses)."""
    stations = pd.read_parquet(path)
    logger.info(f"Estaciones cargadas desde {path}: {len(stations)}")
    return stations


def add_time_fields(trips: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas de hora del día / bandera de fin de semana con las que se construyen los perfiles."""
    df = trips.copy()
    df["hora"] = df["inicio"].dt.hour
    df["fin_de_semana"] = df["inicio"].dt.dayofweek >= 5
    return df


def build_hourly_profile(trips: pd.DataFrame) -> pd.DataFrame:
    """Proporción de salidas de cada estación por hora, separado entre semana (lv) / fin de semana (fds).

    Cada fila suma 1 (distribución conjunta sobre las 48 combinaciones
    hora x tipo de día).
    """
    tipo_dia = np.where(trips["fin_de_semana"], "fds", "lv")
    perfiles = (
        trips.assign(tipo_dia=tipo_dia)
        .groupby(["origen_id", "tipo_dia", "hora"])
        .size()
        .unstack(["tipo_dia", "hora"], fill_value=0)
        .reindex(columns=pd.MultiIndex.from_product([["lv", "fds"], range(24)]), fill_value=0)
    )
    perfiles = perfiles.div(perfiles.sum(axis=1), axis=0)
    perfiles.columns = [f"{t}_{h:02d}" for t, h in perfiles.columns]
    return perfiles


def build_bucketed_profile(trips: pd.DataFrame) -> pd.DataFrame:
    """Proporción de salidas de cada estación por franja horaria, separado lv / fds."""
    hora_a_franja = {h: franja for franja, horas in FRANJAS_HORARIAS.items() for h in horas}
    tipo_dia = np.where(trips["fin_de_semana"], "fds", "lv")
    franja = trips["hora"].map(hora_a_franja)
    perfiles = (
        trips.assign(tipo_dia=tipo_dia, franja=franja)
        .groupby(["origen_id", "tipo_dia", "franja"])
        .size()
        .unstack(["tipo_dia", "franja"], fill_value=0)
        .reindex(columns=pd.MultiIndex.from_product([["lv", "fds"], FRANJAS_HORARIAS.keys()]), fill_value=0)
    )
    perfiles = perfiles.div(perfiles.sum(axis=1), axis=0)
    perfiles.columns = [f"{t}_{b}" for t, b in perfiles.columns]
    return perfiles


def build_station_features(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    granularity: str = "hourly",
    min_salidas: int = MIN_SALIDAS,
) -> tuple[pd.DataFrame, list[str]]:
    """Construye la tabla de features por estación a partir de viajes y catálogo ya limpios.

    Parámetros
    ----------
    granularity: "hourly" para el perfil completo de 24h o "bucketed" para
        las franjas horarias reducidas. Ambas comparten las mismas features resumen.
    min_salidas: estaciones con menos salidas que este umbral se descartan.

    Regresa
    -------
    (features_df, feature_cols): `features_df` indexado por `estacion_id`,
    con columnas de metadatos (`METADATA_COLS`) y columnas de features;
    `feature_cols` lista exactamente las segundas (las que deben escalarse y
    alimentarse a un modelo).
    """
    if granularity not in {"hourly", "bucketed"}:
        raise ValueError(f"granularity debe ser 'hourly' o 'bucketed', recibido: {granularity!r}")

    trips = add_time_fields(trips)

    salidas = trips.groupby("origen_id").size().rename("salidas")
    llegadas = trips.groupby("destino_id").size().rename("llegadas")
    duracion_mediana = trips.groupby("origen_id")["duracion_min"].median().rename("duracion_mediana")

    df = (
        stations.set_index("estacion_id")[["name", "latitude", "longitude", "dpcapacity"]]
        .join(salidas)
        .join(llegadas)
        .join(duracion_mediana)
        .fillna({"salidas": 0, "llegadas": 0})
    )
    df["total"] = df["salidas"] + df["llegadas"]
    df["log_total"] = np.log1p(df["total"])
    df["balance_rel"] = (df["llegadas"] - df["salidas"]) / df["total"].replace(0, np.nan)
    df["rotacion_por_anclaje"] = df["salidas"] / df["dpcapacity"].replace(0, np.nan)
    df["duracion_mediana"] = df["duracion_mediana"].fillna(df["duracion_mediana"].median())
    df[["balance_rel", "rotacion_por_anclaje"]] = df[["balance_rel", "rotacion_por_anclaje"]].fillna(0)

    activas = df.index[df["salidas"] >= min_salidas]
    logger.info(f"Estaciones con >= {min_salidas} salidas: {len(activas)} (descartadas: {len(df) - len(activas)})")

    trips_activas = trips[trips["origen_id"].isin(activas)]
    perfiles = build_hourly_profile(trips_activas) if granularity == "hourly" else build_bucketed_profile(trips_activas)

    out = df.loc[activas].join(perfiles)
    feature_cols = SUMMARY_FEATURE_COLS + list(perfiles.columns)
    logger.success(f"Features ({granularity}): {out.shape[0]} estaciones x {len(feature_cols)} features")
    return out, feature_cols


@app.command()
def main(output_dir: Path = PROCESSED_DATA_DIR, min_salidas: int = MIN_SALIDAS):
    """Construye y escribe ambos conjuntos de features de estación ("hourly" y "bucketed") a `data/processed/`."""
    trips = load_clean_trips()
    stations = load_clean_stations()
    output_dir.mkdir(parents=True, exist_ok=True)

    for granularity in ("hourly", "bucketed"):
        features, feature_cols = build_station_features(trips, stations, granularity=granularity, min_salidas=min_salidas)
        out_path = output_dir / f"estaciones_caracteristicas_{granularity}.parquet"
        features.to_parquet(out_path)
        logger.success(f"Guardado: {out_path} ({features.shape[0]} estaciones x {len(feature_cols)} features)")


if __name__ == "__main__":
    app()
