"""
Módulo de Ingesta, Normalización y Versionado de Datos (Pregunta 2).
===================================================================
Este módulo automatiza completamente el proceso de ETL reproducible para la Pregunta 2 de MiBici:
1. Descarga desatendida de datos crudos (2025 y 2026 + Nomenclatura) si no están en disco.
2. Normalización de esquemas de columnas, codificación (latin1 -> UTF-8) y tipos temporales.
3. Almacenamiento optimizado de datos limpios en formato Parquet desacoplado (p2_estaciones.parquet, p2_viajes_agregados_hora.parquet).
4. Generación y versionado de un manifiesto de datos en Git (references/p2_data_manifest.json)
   con conteos de registros y hashes criptográficos SHA-256 para trazabilidad con MLflow.

Ejecución desde terminal:
    python -m src.data.p2_dataset
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import typer

from src.data.p2_download_raw_data import run_pipeline as run_raw_download
from src.p2_config import (
    P2_DATA_MANIFEST_PATH,
    P2_STATIONS_PARQUET,
    P2_TRIPS_HOURLY_PARQUET,
    RAW_DATA_DIR,
    RAW_NOMENCLATURA_DIR,
    RAW_VIAJES_DIR,
    TARGET_YEARS,
    logger,
)

app = typer.Typer(
    help="CLI de Ingesta, Limpieza y Versionado de Datos para Pregunta 2 (MiBici Guadalajara)."
)


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash criptográfico SHA-256 de un archivo en bloques de 64 KB."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def ensure_raw_data_present(years: List[int] = TARGET_YEARS) -> None:
    """
    Verifica que los datos crudos existan en disco. Si falta alguno, ejecuta
    el pipeline de descarga automatizada sin requerir interacción manual.
    """
    raw_files = (
        [
            f
            for f in RAW_VIAJES_DIR.glob("*.csv")
            if any(f"_{y}_" in f.name for y in years)
        ]
        if RAW_VIAJES_DIR.exists()
        else []
    )

    has_nomenclature = (
        bool(list(RAW_NOMENCLATURA_DIR.glob("*.csv")))
        if RAW_NOMENCLATURA_DIR.exists()
        else False
    )

    if len(raw_files) < 20 or not has_nomenclature:
        logger.warning(
            f"Faltan datos crudos ({len(raw_files)} archivos encontrados para años {years}, nomenclatura={has_nomenclature}). "
            "Iniciando descarga desatendida..."
        )
        run_raw_download(
            years=years,
            output_dir=RAW_DATA_DIR,
            force=False,
            skip_nomenclatura=False,
            organize_subdirs=True,
        )
    else:
        logger.info(
            f"✓ Todos los datos crudos requeridos ya están presentes ({len(raw_files)} viajes + nomenclatura)."
        )


def process_stations_catalog(
    raw_nomenclatura_dir: Path = RAW_NOMENCLATURA_DIR,
    output_parquet: Path = P2_STATIONS_PARQUET,
) -> pd.DataFrame:
    """
    Lee y normaliza el catálogo de nomenclatura de estaciones de MiBici:
    - Renombra columnas a estándar snake_case: station_id, name, latitude, longitude, capacity.
    - Filtra registros con capacidad no válida o identificador nulo.
    - Aplica downcasting a tipos de memoria eficientes (int16, float32).
    - Serializa en formato Parquet comprimido (Snappy).
    """
    logger.info("Normalizando catálogo de estaciones...")
    csv_candidates = list(raw_nomenclatura_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No se encontró archivo de nomenclatura en {raw_nomenclatura_dir}."
        )

    # Si hay varios, seleccionamos el más reciente por orden alfabético
    nomenclatura_file = sorted(csv_candidates)[-1]
    logger.info(f"Leyendo nomenclatura desde: {nomenclatura_file.name}")

    df_nom = pd.read_csv(nomenclatura_file)

    # Identificación flexible de columnas
    col_map = {}
    for col in df_nom.columns:
        c_low = col.lower().strip()
        if c_low in ["id", "station_id", "estacion_id"]:
            col_map[col] = "station_id"
        elif c_low in ["name", "nombre", "nom"]:
            col_map[col] = "name"
        elif c_low in ["latitude", "latitud", "lat"]:
            col_map[col] = "latitude"
        elif c_low in ["longitude", "longitud", "lon", "lng"]:
            col_map[col] = "longitude"
        elif c_low in ["dpcapacity", "capacity", "capacidad", "anclajes"]:
            col_map[col] = "capacity"

    df_nom.rename(columns=col_map, inplace=True)
    required = ["station_id", "capacity", "latitude", "longitude"]
    missing = [c for c in required if c not in df_nom.columns]
    if missing:
        raise ValueError(
            f"Columnas obligatorias ausentes en catálogo de estaciones: {missing}"
        )

    # Limpieza: coerción numérica y filtro de integridad física
    df_nom["station_id"] = pd.to_numeric(df_nom["station_id"], errors="coerce")
    df_nom["capacity"] = pd.to_numeric(df_nom["capacity"], errors="coerce")
    df_nom["latitude"] = pd.to_numeric(df_nom["latitude"], errors="coerce")
    df_nom["longitude"] = pd.to_numeric(df_nom["longitude"], errors="coerce")

    df_nom.dropna(subset=["station_id", "capacity"], inplace=True)
    df_nom.drop_duplicates(subset=["station_id"], inplace=True)
    df_nom = df_nom[df_nom["capacity"] > 0].copy()

    # Downcasting para optimización de RAM
    df_nom["station_id"] = df_nom["station_id"].astype(np.int16)
    df_nom["capacity"] = df_nom["capacity"].astype(np.int16)
    df_nom["latitude"] = df_nom["latitude"].astype(np.float32)
    df_nom["longitude"] = df_nom["longitude"].astype(np.float32)

    # Persistencia en Parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_nom.to_parquet(output_parquet, index=False)
    logger.info(
        f"✓ Estaciones guardadas en Parquet: {output_parquet} ({len(df_nom)} estaciones)."
    )
    return df_nom


def process_hourly_trips(
    raw_viajes_dir: Path = RAW_VIAJES_DIR,
    output_parquet: Path = P2_TRIPS_HOURLY_PARQUET,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Procesa, limpia y agrega los registros de viajes mensuales a nivel (estación, hora):
    - Maneja codificación latin1 para soporte de caracteres especiales.
    - Procesa en chunks para mantener un consumo bajo y predecible de RAM.
    - Normaliza timestamps con truncamiento a nivel hora base (floor 'h').
    - Agrega salidas (origen) y llegadas (destino) por estación y hora.
    - Serializa la tabla agregada en Parquet.
    """
    csv_files = sorted(
        [
            f
            for f in raw_viajes_dir.glob("*.csv")
            if any(f"_{y}_" in f.name for y in TARGET_YEARS)
        ]
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontraron archivos de viajes para los años {TARGET_YEARS} en {raw_viajes_dir}."
        )

    logger.info(
        f"Agregando viajes de {len(csv_files)} archivos mensuales ({TARGET_YEARS})..."
    )
    t0 = time.time()
    salidas_chunks, llegadas_chunks = [], []
    total_trips = 0

    usecols = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]

    for file_path in csv_files:
        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            encoding="latin1",
            chunksize=chunksize,
        ):
            chunk = chunk.dropna()
            total_trips += len(chunk)

            # Normalización y redondeo temporal a la hora base
            t_start = pd.to_datetime(
                chunk["Inicio_del_viaje"], errors="coerce", format="mixed"
            ).dt.floor("h")
            t_end = pd.to_datetime(
                chunk["Fin_del_viaje"], errors="coerce", format="mixed"
            ).dt.floor("h")

            # Demanda (salidas)
            s = (
                chunk.groupby(
                    [chunk["Origen_Id"].astype(np.int16), t_start], observed=False
                )
                .size()
                .reset_index(name="salidas")
            )
            s.columns = ["station_id", "timestamp", "salidas"]
            salidas_chunks.append(s)

            # Oferta (llegadas)
            chunk_llegadas = (
                chunk.groupby(
                    [chunk["Destino_Id"].astype(np.int16), t_end], observed=False
                )
                .size()
                .reset_index(name="llegadas")
            )
            chunk_llegadas.columns = ["station_id", "timestamp", "llegadas"]
            llegadas_chunks.append(chunk_llegadas)

    df_salidas = (
        pd.concat(salidas_chunks, ignore_index=True)
        .groupby(["station_id", "timestamp"], as_index=False)["salidas"]
        .sum()
    )
    df_llegadas = (
        pd.concat(llegadas_chunks, ignore_index=True)
        .groupby(["station_id", "timestamp"], as_index=False)["llegadas"]
        .sum()
    )

    df_salidas["salidas"] = df_salidas["salidas"].astype(np.int16)
    df_llegadas["llegadas"] = df_llegadas["llegadas"].astype(np.int16)

    # Consolidación de salidas y llegadas en una tabla horaria única
    df_hourly = pd.merge(
        df_salidas, df_llegadas, on=["station_id", "timestamp"], how="outer"
    )
    df_hourly["salidas"] = df_hourly["salidas"].fillna(0).astype(np.int16)
    df_hourly["llegadas"] = df_hourly["llegadas"].fillna(0).astype(np.int16)
    df_hourly.sort_values(["station_id", "timestamp"], inplace=True)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t0
    logger.info(
        f"✓ {total_trips:,} viajes consolidados en {len(df_hourly):,} registros horarios ({elapsed:.1f}s) -> {output_parquet}"
    )
    return df_hourly


def generate_data_manifest(
    manifest_path: Path = P2_DATA_MANIFEST_PATH,
    raw_viajes_dir: Path = RAW_VIAJES_DIR,
    raw_nomenclatura_dir: Path = RAW_NOMENCLATURA_DIR,
) -> Dict:
    """
    Genera un archivo JSON de manifiesto con los hashes SHA-256 de todos los datos
    crudos e intermedios para garantizar reproducibilidad estricta y linaje con MLflow.
    """
    logger.info("Generando manifiesto criptográfico de versión de datos para P2...")
    manifest_files = {}

    # 1. Catálogo de Nomenclatura
    for p in sorted(raw_nomenclatura_dir.glob("*.csv")):
        manifest_files[f"raw/Nomenclatura/{p.name}"] = {
            "sha256": compute_sha256(p),
            "size_bytes": p.stat().st_size,
        }

    # 2. Archivos de Viajes crudos correspondientes a los años objetivo
    for p in sorted(raw_viajes_dir.glob("*.csv")):
        if any(f"_{y}_" in p.name for y in TARGET_YEARS):
            manifest_files[f"raw/Viajes/{p.name}"] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }

    # 3. Artefactos Parquet generados para P2
    for p in [P2_STATIONS_PARQUET, P2_TRIPS_HOURLY_PARQUET]:
        if p.exists():
            df_temp = pd.read_parquet(p)
            manifest_files[f"interim/{p.name}"] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
                "rows_count": len(df_temp),
            }

    # Hash combinado del dataset completo para versionado único en MLflow
    combined_hash = hashlib.sha256(
        "".join(sorted([f["sha256"] for f in manifest_files.values()])).encode()
    ).hexdigest()

    years_str = "_".join(str(y) for y in sorted(TARGET_YEARS))
    data_version = f"p2-mibici-v{years_str}-{combined_hash[:10]}"

    manifest_data = {
        "dataset_name": "MiBici Guadalajara Open Data (Pregunta 2)",
        "data_version": data_version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_years": TARGET_YEARS,
        "total_files_tracked": len(manifest_files),
        "files": manifest_files,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.success(
        f"✓ Manifiesto de datos guardado: {manifest_path} (Versión: {data_version})"
    )
    return manifest_data


def get_current_data_version() -> str:
    """Devuelve el identificador de versión del manifiesto para tracking en MLflow."""
    if P2_DATA_MANIFEST_PATH.exists():
        with open(P2_DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("data_version", "unknown_p2_data_version")
    return "unversioned_p2_data"


@app.command()
def main(
    download: bool = typer.Option(
        True, help="Verificar y descargar datos crudos automáticamente si faltan."
    ),
    generate_manifest: bool = typer.Option(
        True, help="Generar manifiesto de versionado con hashes SHA-256."
    ),
):
    """
    Ejecuta el pipeline completo de ingesta, normalización y versionado para la Pregunta 2.
    """
    logger.info("=" * 70)
    logger.info("INICIO DE INGESTA Y NORMALIZACIÓN DE DATOS (PREGUNTA 2 - MiBici)")
    logger.info("=" * 70)

    if download:
        ensure_raw_data_present(TARGET_YEARS)

    process_stations_catalog()
    process_hourly_trips()

    if generate_manifest:
        manifest = generate_data_manifest()
        logger.info(f"Dataset versionado exitosamente: {manifest['data_version']}")

    logger.info("=" * 70)
    logger.info("PIPELINE DE INGESTA Y VERSIONADO (P2) COMPLETADO EXITOSAMENTE")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
