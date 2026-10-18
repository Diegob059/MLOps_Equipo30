"""
Módulo de Ingesta, Limpieza y Versionado de Datos (Compartido).
================================================================
Punto de ingesta único del proyecto (ver instrucciones de la actividad: "En
el archivo `dataset.py` del módulo..."). Cualquier pregunta (P1, P2 o P3)
puede construir su propio *feature engineering* a partir de los artefactos
que este módulo produce en ``data/interim/``:

1. Descarga desatendida de los datos crudos (viajes 2025-2026 y nomenclatura
   de estaciones) si no están presentes en disco, reutilizando el *scraper*
   ya implementado en ``src/data/p2_download_raw_data.py``.
2. Normalización de columnas, codificación (Latin-1 con encabezados
   inconsistentes) y tipos de fecha.
3. Limpieza: elimina duplicados, filas con `genero`/`anio_nacimiento`
   faltante, viajes con duración fuera de rango, y estaciones que no están
   presentes en todos los meses descargados (para no contar estaciones
   recién abiertas).
4. Generación y versionado en Git de un manifiesto de datos
   (``references/data_manifest.json``) con conteos de registros y hashes
   SHA-256 de cada archivo crudo, consultable desde MLflow vía
   `get_current_data_version()`.

Ejecución desde terminal:
    python -m src.data.dataset
"""

import hashlib
import json
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import typer

from src.config import (
    DATA_MANIFEST_PATH,
    INTERIM_STATIONS_PARQUET,
    INTERIM_TRIPS_PARQUET,
    RAW_DATA_DIR,
    RAW_NOMENCLATURA_DIR,
    RAW_VIAJES_DIR,
    TARGET_YEARS,
    logger,
)
from src.data.p2_download_raw_data import run_pipeline as descargar_datos_crudos

app = typer.Typer(help="CLI de ingesta, limpieza y versionado de datos de MiBici (compartido).")

SUBDIR_VIAJES = "Viajes"
SUBDIR_ESTACIONES = "Nomenclatura"

# La codificación del encabezado "Año_de_nacimiento" es inconsistente entre
# archivos mensuales (UTF-8 vs. Latin-1 con caracteres mal interpretados), por
# lo que las columnas se renombran por posición en lugar de depender de los
# nombres del encabezado.
COLUMNAS_VIAJES = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]

DURACION_MINIMA_DEFECTO = 1.0
DURACION_MAXIMA_DEFECTO = 180.0


def ensure_raw_data_present(years: list[int] = TARGET_YEARS) -> None:
    """Verifica que los datos crudos existan en disco; si faltan, dispara la descarga desatendida."""
    raw_files = (
        [f for f in RAW_VIAJES_DIR.glob("*.csv") if any(f"_{y}_" in f.name for y in years)]
        if RAW_VIAJES_DIR.exists()
        else []
    )
    has_nomenclature = bool(list(RAW_NOMENCLATURA_DIR.glob("*.csv"))) if RAW_NOMENCLATURA_DIR.exists() else False

    if len(raw_files) < 20 or not has_nomenclature:
        logger.warning(
            f"Faltan datos crudos ({len(raw_files)} archivos para años {years}, "
            f"nomenclatura={has_nomenclature}). Iniciando descarga desatendida..."
        )
        descargar_datos_crudos(years=years, output_dir=RAW_DATA_DIR, force=False, skip_nomenclatura=False)
    else:
        logger.info(f"Datos crudos ya presentes ({len(raw_files)} archivos de viajes + nomenclatura).")


def _leer_archivo_viajes(ruta: Path) -> pd.DataFrame:
    """Lee un único CSV mensual de viajes de MiBici con la codificación/columnas correctas."""
    df = pd.read_csv(
        ruta,
        encoding="latin-1",
        header=0,
        names=COLUMNAS_VIAJES,
        dtype={
            "viaje_id": "int64",
            "usuario_id": "int64",
            "genero": "string",
            "origen_id": "int32",
            "destino_id": "int32",
        },
    )
    df["archivo"] = ruta.stem[-7:]  # p. ej. "2025_01"
    return df


def cargar_viajes(raw_viajes_dir: Path = RAW_VIAJES_DIR, patron: str = "datos_abiertos_202[0-9]_*.csv") -> pd.DataFrame:
    """Carga y concatena todos los archivos mensuales de viajes encontrados en ``raw_viajes_dir``."""
    archivos = sorted(raw_viajes_dir.glob(patron))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos de viajes en {raw_viajes_dir}")
    logger.info(f"Cargando {len(archivos)} archivos de viajes desde {raw_viajes_dir}")
    viajes = pd.concat([_leer_archivo_viajes(f) for f in archivos], ignore_index=True)
    viajes["inicio"] = pd.to_datetime(viajes["inicio"], errors="coerce")
    viajes["fin"] = pd.to_datetime(viajes["fin"], errors="coerce")
    logger.info(f"Viajes cargados: {len(viajes):,}")
    return viajes


def cargar_estaciones(raw_nomenclatura_dir: Path = RAW_NOMENCLATURA_DIR, patron: str = "nomenclatura_*.csv") -> pd.DataFrame:
    """Carga el catálogo de estaciones más reciente encontrado en ``raw_nomenclatura_dir``."""
    archivos = sorted(raw_nomenclatura_dir.glob(patron))
    if not archivos:
        raise FileNotFoundError(f"No se encontró catálogo de estaciones en {raw_nomenclatura_dir}")
    estaciones = pd.read_csv(archivos[-1], encoding="latin-1").rename(columns={"id": "estacion_id"})
    logger.info(f"Catálogo de estaciones: {archivos[-1].name} ({len(estaciones)} estaciones)")
    return estaciones


def obtener_estaciones_comunes(viajes: pd.DataFrame) -> set:
    """Regresa los ids de estación presentes (como origen o destino) en todos los meses de ``viajes``.

    Las estaciones abiertas después del primer mes (p. ej. nuevas estaciones
    agregadas durante 2025/2026) están ausentes en los archivos más antiguos y
    por lo tanto quedan excluidas por esta intersección, evitando contar
    estaciones recién abiertas como si tuvieran historia completa.
    """
    comunes: set | None = None
    for _, df_mes in viajes.groupby("archivo"):
        ids = set(df_mes["origen_id"]) | set(df_mes["destino_id"])
        comunes = ids if comunes is None else (comunes & ids)
    comunes = comunes or set()
    logger.info(f"Estaciones presentes en los {viajes['archivo'].nunique()} meses: {len(comunes)}")
    return comunes


def limpiar_viajes(
    viajes: pd.DataFrame,
    estaciones_comunes: Iterable[int] | None = None,
    duracion_minima_min: float = DURACION_MINIMA_DEFECTO,
    duracion_maxima_min: float = DURACION_MAXIMA_DEFECTO,
) -> pd.DataFrame:
    """Aplica los filtros de calidad de datos compartidos por cualquier pregunta que use viajes limpios.

    Filtros:
    - Elimina filas exactamente duplicadas.
    - Elimina filas con ``genero`` o ``anio_nacimiento`` faltante.
    - Conserva solo viajes con duración en ``[duracion_minima_min, duracion_maxima_min]``.
    - Si se da ``estaciones_comunes``, conserva solo viajes donde tanto el
      origen como el destino estén en ese conjunto.
    """
    df = viajes.copy()
    n0 = len(df)

    columnas_dedup = [c for c in df.columns if c != "archivo"]
    df = df.drop_duplicates(subset=columnas_dedup)
    logger.info(f"Tras eliminar duplicados: {len(df):,} ({n0 - len(df):,} descartados)")

    n1 = len(df)
    df = df.dropna(subset=["genero", "anio_nacimiento"])
    logger.info(f"Tras eliminar genero/anio_nacimiento nulos: {len(df):,} ({n1 - len(df):,} descartados)")

    n2 = len(df)
    duracion_min = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    mascara_duracion = duracion_min.between(duracion_minima_min, duracion_maxima_min)
    df = df.loc[mascara_duracion].copy()
    df["duracion_min"] = duracion_min.loc[mascara_duracion]
    logger.info(
        f"Tras filtrar duración [{duracion_minima_min}, {duracion_maxima_min}] min: "
        f"{len(df):,} ({n2 - len(df):,} descartados)"
    )

    if estaciones_comunes is not None:
        n3 = len(df)
        estaciones_comunes = set(estaciones_comunes)
        df = df[df["origen_id"].isin(estaciones_comunes) & df["destino_id"].isin(estaciones_comunes)]
        logger.info(f"Tras filtrar estaciones comunes a todos los meses: {len(df):,} ({n3 - len(df):,} descartados)")

    logger.success(f"Limpieza completa: {len(df):,}/{n0:,} viajes conservados ({len(df) / n0 * 100:.1f}%)")
    return df.reset_index(drop=True)


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash criptográfico SHA-256 de un archivo en bloques de 64 KB."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def generate_data_manifest(
    manifest_path: Path = DATA_MANIFEST_PATH,
    raw_viajes_dir: Path = RAW_VIAJES_DIR,
    raw_nomenclatura_dir: Path = RAW_NOMENCLATURA_DIR,
    years: list[int] = TARGET_YEARS,
) -> dict:
    """Genera/actualiza el manifiesto JSON con los hashes SHA-256 de los datos crudos, versionado en Git."""
    logger.info("Generando manifiesto criptográfico de versión de datos...")
    manifest_files = {}

    for p in sorted(raw_nomenclatura_dir.glob("*.csv")):
        manifest_files[f"raw/Nomenclatura/{p.name}"] = {"sha256": compute_sha256(p), "size_bytes": p.stat().st_size}

    for p in sorted(raw_viajes_dir.glob("*.csv")):
        if any(f"_{y}_" in p.name for y in years):
            manifest_files[f"raw/Viajes/{p.name}"] = {"sha256": compute_sha256(p), "size_bytes": p.stat().st_size}

    concatenated_hashes = "".join(v["sha256"][:8] for v in manifest_files.values())
    data_version = f"mibici-v{years[0]}_{years[-1]}-{hashlib.sha256(concatenated_hashes.encode()).hexdigest()[:10]}"

    manifest = {
        "dataset_name": "MiBici Guadalajara Open Data",
        "data_version": data_version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_years": years,
        "total_files_tracked": len(manifest_files),
        "files": manifest_files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.success(f"Manifiesto guardado: {manifest_path} ({len(manifest_files)} archivos, versión {data_version})")
    return manifest


def get_current_data_version(manifest_path: Path = DATA_MANIFEST_PATH) -> str:
    """Devuelve el identificador de versión del manifiesto para registrar en MLflow."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f).get("data_version", "unknown_data_version")
    return "unversioned_data"


@app.command()
def main(
    download: bool = typer.Option(True, help="Verificar y descargar datos crudos automáticamente si faltan."),
    generate_manifest: bool = typer.Option(True, help="Generar/actualizar el manifiesto de versionado de datos."),
    duracion_minima_min: float = DURACION_MINIMA_DEFECTO,
    duracion_maxima_min: float = DURACION_MAXIMA_DEFECTO,
):
    """Ejecuta el pipeline completo de ingesta, limpieza y versionado, compartido por todas las preguntas."""
    logger.info("=" * 70)
    logger.info("INICIO DE INGESTA, LIMPIEZA Y VERSIONADO DE DATOS (MiBici, compartido)")
    logger.info("=" * 70)

    if download:
        ensure_raw_data_present()

    viajes = cargar_viajes()
    estaciones = cargar_estaciones()

    estaciones_comunes = obtener_estaciones_comunes(viajes) & set(estaciones["estacion_id"])
    viajes_limpios = limpiar_viajes(
        viajes,
        estaciones_comunes=estaciones_comunes,
        duracion_minima_min=duracion_minima_min,
        duracion_maxima_min=duracion_maxima_min,
    )
    estaciones_limpias = estaciones[estaciones["estacion_id"].isin(estaciones_comunes)].reset_index(drop=True)

    INTERIM_TRIPS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    viajes_limpios.to_parquet(INTERIM_TRIPS_PARQUET, index=False)
    estaciones_limpias.to_parquet(INTERIM_STATIONS_PARQUET, index=False)
    logger.success(f"Guardado: {INTERIM_TRIPS_PARQUET} ({len(viajes_limpios):,} filas)")
    logger.success(f"Guardado: {INTERIM_STATIONS_PARQUET} ({len(estaciones_limpias):,} estaciones)")

    if generate_manifest:
        generate_data_manifest()


if __name__ == "__main__":
    app()
