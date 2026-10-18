#!/usr/bin/env python3
"""
Módulo de Extracción de Datos Abiertos de MiBici Guadalajara.
=============================================================
Este script automatiza la descarga de los datos históricos de viajes (CSV)
para los años solicitados (por defecto 2025 y 2026), así como el catálogo de
nomenclatura y capacidad de las estaciones del sistema MiBici.

Características:
    - Scraping dinámico del portal oficial (https://www.mibici.net/es/datos-abiertos/).
    - Mecanismo de fallback con URLs determinísticas de assets en caso de contingencia.
    - Idempotencia: comprueba existencia y tamaño previo para evitar descargas redundantes.
    - Sesión HTTP resiliente con reintentos y backoff exponencial (urllib3/requests).
    - Barras de progreso por archivo (tqdm) y validación de integridad (anti-HTML de error).
    - Descompresión automática si algún archivo se distribuye como .zip.
    - Soporte CLI completo mediante argparse (Cookiecutter Data Science).

Uso:
    python src/data/download_raw_data.py --years 2025 2026
    python src/data/download_raw_data.py --years 2026 --output_dir data/raw --force
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
import os
import re
import sys
import time
import urllib.parse
import zipfile

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Configuración de logging legible y profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mibici_downloader")

BASE_URL = "https://www.mibici.net"
PORTAL_URL = "https://www.mibici.net/es/datos-abiertos/"

# Fallback determinístico con URLs conocidas de assets del servidor de MiBici
# Garantiza que el script funcione aun si la estructura HTML del portal llegara a modificarse
FALLBACK_TRIPS: Dict[Tuple[int, int], str] = {
    # 2023
    (2023, 1): "https://www.mibici.net/site/assets/files/36762/datos_abiertos_2023_01.csv",
    (2023, 2): "https://www.mibici.net/site/assets/files/40197/datos_abiertos_2023_02.csv",
    (2023, 3): "https://www.mibici.net/site/assets/files/43035/datos_abiertos_2023_03.csv",
    (2023, 4): "https://www.mibici.net/site/assets/files/46758/datos_abiertos_2023_04.csv",
    (2023, 5): "https://www.mibici.net/site/assets/files/46759/datos_abiertos_2023_05.csv",
    (2023, 6): "https://www.mibici.net/site/assets/files/48116/datos_abiertos_2023_06.csv",
    (2023, 7): "https://www.mibici.net/site/assets/files/49448/datos_abiertos_2023_07.csv",
    (2023, 8): "https://www.mibici.net/site/assets/files/51295/datos_abiertos_2023_08.csv",
    (2023, 9): "https://www.mibici.net/site/assets/files/53344/datos_abiertos_2023_09.csv",
    (2023, 10): "https://www.mibici.net/site/assets/files/54935/datos_abiertos_2023_10.csv",
    (2023, 11): "https://www.mibici.net/site/assets/files/57344/datos_abiertos_2023_11.csv",
    (2023, 12): "https://www.mibici.net/site/assets/files/58715/datos_abiertos_2023_12.csv",
    # 2024
    (2024, 1): "https://www.mibici.net/site/assets/files/61450/datos_abiertos_2024_01.csv",
    (2024, 2): "https://www.mibici.net/site/assets/files/77251/datos_abiertos_2024_02.csv",
    (2024, 3): "https://www.mibici.net/site/assets/files/77252/datos_abiertos_2024_03.csv",
    (2024, 4): "https://www.mibici.net/site/assets/files/80028/datos_abiertos_2024_04.csv",
    (2024, 5): "https://www.mibici.net/site/assets/files/81663/datos_abiertos_2024_05.csv",
    (2024, 6): "https://www.mibici.net/site/assets/files/83901/datos_abiertos_2024_06.csv",
    (2024, 7): "https://www.mibici.net/site/assets/files/85500/datos_abiertos_2024_07.csv",
    (2024, 8): "https://www.mibici.net/site/assets/files/87665/datos_abiertos_2024_08.csv",
    (2024, 9): "https://www.mibici.net/site/assets/files/89736/datos_abiertos_2024_09.csv",
    (2024, 10): "https://www.mibici.net/site/assets/files/91758/datos_abiertos_2024_10.csv",
    (2024, 11): "https://www.mibici.net/site/assets/files/93516/datos_abiertos_2024_11.csv",
    (2024, 12): "https://www.mibici.net/site/assets/files/94962/datos_abiertos_2024_12.csv",
    # 2025
    (2025, 1): "https://www.mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    (2025, 2): "https://www.mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv",
    (2025, 3): "https://www.mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv",
    (2025, 4): "https://www.mibici.net/site/assets/files/103813/datos_abiertos_2025_04.csv",
    (2025, 5): "https://www.mibici.net/site/assets/files/105513/datos_abiertos_2025_05.csv",
    (2025, 6): "https://www.mibici.net/site/assets/files/107088/datos_abiertos_2025_06.csv",
    (2025, 7): "https://www.mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv",
    (2025, 8): "https://www.mibici.net/site/assets/files/111021/datos_abiertos_2025_08.csv",
    (2025, 9): "https://www.mibici.net/site/assets/files/113216/datos_abiertos_2025_09.csv",
    (2025, 10): "https://www.mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv",
    (2025, 11): "https://www.mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv",
    (2025, 12): "https://www.mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv",
    # 2026
    (2026, 1): "https://www.mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    (2026, 2): "https://www.mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    (2026, 3): "https://www.mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    (2026, 4): "https://www.mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    (2026, 5): "https://www.mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    (2026, 6): "https://www.mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    (2026, 7): "https://www.mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    (2026, 8): "https://www.mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    (2026, 9): "https://www.mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
}

FALLBACK_NOMENCLATURA_URL = (
    "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
)


@dataclass
class DownloadItem:
    """Representa un archivo objetivo para descarga."""

    filename: str
    url: str
    file_type: str  # 'trip' o 'nomenclature'
    year: Optional[int] = None
    month: Optional[int] = None


def create_resilient_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Configura y devuelve una sesión de requests con política de reintentos
    exponencial y cabeceras estándar de navegador para evitar códigos 406/403.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "application/octet-stream,text/csv,*/*;q=0.8"
            ),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        }
    )
    return session


def scrape_portal_links(
    portal_url: str,
    years: List[int],
    session: requests.Session,
    include_nomenclatura: bool = True,
) -> List[DownloadItem]:
    """
    Realiza scraping del portal de datos abiertos de MiBici para extraer
    los hipervínculos a los archivos de viajes y nomenclatura.
    Si algún mes de los años requeridos no se encuentra en el DOM, activa
    el mecanismo de fallback determinístico.
    """
    items_by_filename: Dict[str, DownloadItem] = {}
    years_set = set(years)
    found_trips: Dict[Tuple[int, int], DownloadItem] = {}
    nomenclatura_found = False

    logger.info(f"Conectando al portal oficial: {portal_url}")
    try:
        response = session.get(portal_url, timeout=(10, 30))
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            anchors = soup.find_all("a", href=True)
            logger.info(f"Analizando {len(anchors)} enlaces en el documento HTML...")

            for a in anchors:
                href = a["href"].strip()
                full_url = urllib.parse.urljoin(BASE_URL, href)
                filename = Path(urllib.parse.urlparse(full_url).path).name

                # 1. Catálogo de Nomenclatura de estaciones
                if include_nomenclatura and "nomenclatura" in filename.lower():
                    if not nomenclatura_found:
                        items_by_filename[filename] = DownloadItem(
                            filename=filename,
                            url=full_url,
                            file_type="nomenclature",
                        )
                        nomenclatura_found = True
                        logger.debug(f"Encontrada nomenclatura en DOM: {filename}")

                # 2. Archivos mensuales de viajes: datos_abiertos_AAAA_MM.csv
                match = re.search(r"datos_abiertos_(\d{4})_(\d{2})", filename, re.IGNORECASE)
                if match:
                    y = int(match.group(1))
                    m = int(match.group(2))
                    if y in years_set and (y, m) not in found_trips:
                        item = DownloadItem(
                            filename=filename,
                            url=full_url,
                            file_type="trip",
                            year=y,
                            month=m,
                        )
                        found_trips[(y, m)] = item
                        items_by_filename[filename] = item
                        logger.debug(f"Encontrado viaje en DOM: {filename}")

        else:
            logger.warning(
                f"El portal respondió con HTTP {response.status_code}. Se activará fallback."
            )
    except Exception as exc:
        logger.warning(f"Error al conectar con {portal_url}: {exc}. Se utilizará fallback.")

    # Mecanismo de Fallback para viajes faltantes
    for y in sorted(years_set):
        for m in range(1, 13):
            key = (y, m)
            if key not in found_trips and key in FALLBACK_TRIPS:
                fallback_url = FALLBACK_TRIPS[key]
                fallback_name = Path(urllib.parse.urlparse(fallback_url).path).name
                logger.info(
                    f"Activando fallback para {y}-{m:02d} -> {fallback_name}"
                )
                item = DownloadItem(
                    filename=fallback_name,
                    url=fallback_url,
                    file_type="trip",
                    year=y,
                    month=m,
                )
                items_by_filename[fallback_name] = item

    # Mecanismo de Fallback para nomenclatura si no se encontró en el DOM
    if include_nomenclatura and not nomenclatura_found:
        nom_name = Path(urllib.parse.urlparse(FALLBACK_NOMENCLATURA_URL).path).name
        logger.info(f"Activando fallback para nomenclatura -> {nom_name}")
        items_by_filename[nom_name] = DownloadItem(
            filename=nom_name,
            url=FALLBACK_NOMENCLATURA_URL,
            file_type="nomenclature",
        )

    return list(items_by_filename.values())


def verify_file_integrity(file_path: Path) -> bool:
    """
    Valida que el archivo descargado no esté vacío ni corresponda a una página
    HTML de error devuelta con código 200 (como errores 404 o 500 personalizados).
    """
    if not file_path.exists():
        return False

    size = file_path.stat().st_size
    if size == 0:
        logger.error(f"Fallo de integridad: {file_path.name} tiene tamaño 0 bytes.")
        return False

    # Inspeccionamos los primeros 512 bytes buscando encabezados HTML sospechosos
    try:
        with open(file_path, "rb") as f:
            header_bytes = f.read(512).strip().lower()
        if (
            header_bytes.startswith(b"<!doctype html")
            or header_bytes.startswith(b"<html")
            or b"<title>404" in header_bytes
            or b"<title>error" in header_bytes
        ):
            logger.error(
                f"Fallo de integridad: {file_path.name} contiene HTML en lugar de datos CSV/ZIP."
            )
            return False
    except Exception as err:
        logger.error(f"Error al leer cabecera de {file_path.name}: {err}")
        return False

    return True


def extract_if_zip(file_path: Path, target_dir: Path) -> List[Path]:
    """
    Si el archivo descargado es un archivo comprimido .zip, lo extrae en el
    directorio de destino y devuelve la lista de archivos resultantes.
    """
    if file_path.suffix.lower() == ".zip":
        logger.info(f"Descomprimiendo archivo ZIP: {file_path.name} en {target_dir}...")
        extracted: List[Path] = []
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(target_dir)
                for member in zf.namelist():
                    extracted_path = target_dir / member
                    extracted.append(extracted_path)
                    logger.info(f"  -> Extraído: {member}")
            return extracted
        except Exception as exc:
            logger.error(f"Error al descomprimir {file_path.name}: {exc}")
    return [file_path]


def resolve_local_target(
    item: DownloadItem,
    output_dir: Path,
    organize_subdirs: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Determina la ruta de destino deseada y comprueba si el archivo ya existe
    previamente en la ruta estándar o en subcarpetas (Viajes / Nomenclatura).
    """
    if organize_subdirs:
        subfolder = "Viajes" if item.file_type == "trip" else "Nomenclatura"
        preferred_path = output_dir / subfolder / item.filename
    else:
        preferred_path = output_dir / item.filename

    # Rutas candidatas donde podría existir el archivo ya descargado
    candidates = [
        preferred_path,
        output_dir / item.filename,
        output_dir / "Viajes" / item.filename,
        output_dir / "Nomenclatura" / item.filename,
    ]

    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return preferred_path, cand

    return preferred_path, None


def download_item(
    item: DownloadItem,
    output_dir: Path,
    session: requests.Session,
    force: bool = False,
    organize_subdirs: bool = True,
    timeout: Tuple[int, int] = (10, 90),
) -> bool:
    """
    Descarga un archivo específico aplicando idempotencia, streaming,
    barra de progreso tqdm, reintentos y validación de integridad.
    """
    target_path, existing_path = resolve_local_target(
        item, output_dir, organize_subdirs=organize_subdirs
    )

    # Idempotencia: omitir si ya existe y tiene tamaño válido
    if not force and existing_path is not None:
        size_mb = existing_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"✓ Omitiendo {item.filename}: ya existe en {existing_path.relative_to(output_dir.parent)} "
            f"({size_mb:.2f} MB)"
        )
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    logger.info(f"Iniciando descarga: {item.filename} desde {item.url}")
    try:
        with session.get(item.url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                logger.error(
                    f"Código HTTP {resp.status_code} al solicitar {item.filename}."
                )
                return False

            total_bytes = int(resp.headers.get("Content-Length", 0))
            chunk_size = 64 * 1024  # 64 KB

            with open(temp_path, "wb") as f, tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"↓ {item.filename[:26]}",
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Validación de integridad antes de consolidar el archivo
        if not verify_file_integrity(temp_path):
            if temp_path.exists():
                temp_path.unlink()
            return False

        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)

        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Descargado exitosamente: {item.filename} ({size_mb:.2f} MB)")

        # Si vino en formato ZIP, descomprimir
        extract_if_zip(target_path, target_path.parent)
        return True

    except Exception as exc:
        logger.error(f"Fallo durante la descarga de {item.filename}: {exc}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def run_pipeline(
    years: List[int],
    output_dir: Path,
    force: bool = False,
    skip_nomenclatura: bool = False,
    organize_subdirs: bool = True,
) -> None:
    """
    Ejecuta el flujo completo de descubrimiento y descarga de datos crudos.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE DE DESCARGA DE DATOS ABIERTOS MIBICI (2025 - 2026)")
    logger.info(f"Años objetivo       : {years}")
    logger.info(f"Directorio de salida: {output_dir.resolve()}")
    logger.info(f"Incluir nomenclatura: {not skip_nomenclatura}")
    logger.info(f"Forzar re-descarga  : {force}")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    session = create_resilient_session()

    items = scrape_portal_links(
        portal_url=PORTAL_URL,
        years=years,
        session=session,
        include_nomenclatura=not skip_nomenclatura,
    )

    if not items:
        logger.error("No se encontraron elementos para descargar. Abortando.")
        sys.exit(1)

    logger.info(f"Total de archivos a procesar: {len(items)}")

    # Ordenamos: primero nomenclatura, luego viajes por año y mes
    items.sort(
        key=lambda x: (
            0 if x.file_type == "nomenclature" else 1,
            x.year or 0,
            x.month or 0,
        )
    )

    success_count = 0
    fail_count = 0

    for item in items:
        ok = download_item(
            item=item,
            output_dir=output_dir,
            session=session,
            force=force,
            organize_subdirs=organize_subdirs,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(
        f"Pipeline finalizado en {elapsed:.1f}s. "
        f"Exitosos: {success_count}/{len(items)}, Fallos: {fail_count}"
    )
    logger.info("=" * 70)

    if fail_count > 0:
        logger.warning("Uno o más archivos no pudieron descargarse.")
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    """Configuración de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Descarga automatizada de datos abiertos de MiBici (Años 2025 y 2026)."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help="Años de viajes a descargar (ej. --years 2025 2026). Por defecto: 2025 2026.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directorio de destino para los archivos crudos (por defecto: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar la descarga incluso si los archivos ya existen localmente.",
    )
    parser.add_argument(
        "--skip_nomenclatura",
        action="store_true",
        help="Omitir la descarga del catálogo de estaciones / nomenclatura.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Descargar archivos directamente en output_dir sin subcarpetas Viajes/ y Nomenclatura/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolución inteligente del directorio de salida:
    # Si 'data/raw' se pasa por defecto y no existe en CWD pero existe en MLOps_Equipo30/data/raw,
    # se auto-resuelve para conveniencia de ejecución desde cualquier nivel del repositorio.
    raw_path = Path(args.output_dir)
    if not raw_path.is_absolute():
        if not raw_path.exists() and (Path("MLOps_Equipo30") / raw_path).exists():
            raw_path = Path("MLOps_Equipo30") / raw_path
        elif not raw_path.exists():
            script_parent = Path(__file__).resolve().parents[2] / "data" / "raw"
            if script_parent.exists():
                raw_path = script_parent

    run_pipeline(
        years=args.years,
        output_dir=raw_path,
        force=args.force,
        skip_nomenclatura=args.skip_nomenclatura,
        organize_subdirs=not args.flat,
    )


if __name__ == "__main__":
    main()
