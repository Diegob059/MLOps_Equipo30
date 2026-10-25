"""Conteos horarios de P1, deduplicación y manifiesto de datos.

python -m src.data.p1_dataset [--raw-dir /ruta/Viajes --catalog /ruta/catalogo.csv]
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from src import p1_config as cfg


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, encoding='utf-8-sig', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp1252', **kwargs)


def prepare_dataset(raw_dir=cfg.RAW_DIR, catalog=cfg.CATALOG):
    """Lee los 21 meses completos; no descarga ni modifica los CSV originales."""
    raw_dir, catalog = Path(raw_dir), Path(catalog)
    files = [raw_dir / f'datos_abiertos_{m.year}_{m.month:02d}.csv'
             for m in pd.period_range(cfg.START, cfg.END, freq='M')]
    missing = [str(p) for p in [*files, catalog] if not p.exists()]
    if missing:
        raise FileNotFoundError('Faltan archivos: ' + ', '.join(missing))
    summaries, manifest_files, seen = [], [], set()
    columns = ['Viaje_Id', 'Inicio_del_viaje', 'Fin_del_viaje', 'Origen_Id', 'Destino_Id']
    for path in files:
        month = read_csv(path, usecols=columns)
        raw_rows = len(month)
        month = month.drop_duplicates()
        if month['Viaje_Id'].isna().any() or month['Viaje_Id'].duplicated().any():
            raise ValueError(f'Identificadores ausentes o conflictivos en {path.name}')
        ids = set(month['Viaje_Id'])
        if seen.intersection(ids):
            raise ValueError(f'Viaje_Id repetido entre archivos: {path.name}')
        seen.update(ids)
        begin = pd.to_datetime(month['Inicio_del_viaje'], errors='raise')
        end = pd.to_datetime(month['Fin_del_viaje'], errors='raise')
        if begin.isna().any() or end.isna().any() or (end < begin).any():
            raise ValueError(f'Tiempos incompletos o invertidos en {path.name}')
        if month[['Origen_Id', 'Destino_Id']].isna().any().any():
            raise ValueError(f'Estaciones ausentes en {path.name}')
        for flow, station, moment in [('salidas', 'Origen_Id', begin),
                                       ('llegadas', 'Destino_Id', end)]:
            events = pd.DataFrame({'estacion': month[station], 'timestamp': moment.dt.floor('h')})
            count = events.groupby(['estacion', 'timestamp']).size().rename(flow)
            summaries.append(count.to_frame())
        manifest_files.append({'file': path.name, 'sha256': sha256(path),
                               'bytes': path.stat().st_size, 'raw_rows': raw_rows,
                               'clean_rows': len(month), 'duplicates_removed': raw_rows - len(month)})
        print(f'{path.name}: {len(month):,} viajes', flush=True)
    hourly = pd.concat(summaries).groupby(level=['estacion', 'timestamp']).sum().fillna(0)
    hourly = hourly.astype('int32').reset_index().sort_values(['estacion', 'timestamp'])
    hourly = hourly[hourly['timestamp'].between(cfg.START, cfg.END + ' 23:00:00')]
    hourly['estacion'] = hourly['estacion'].astype('int32')
    cfg.HOURLY.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(cfg.HOURLY, index=False)
    catalog_hash = sha256(catalog)
    identity = {p['file']: p['sha256'] for p in manifest_files}
    identity['nomenclatura_2026_09.csv'] = catalog_hash
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {'data_version': 'p1-' + version, 'period': [cfg.START, cfg.END],
                'files': manifest_files, 'catalog_sha256': catalog_hash,
                'hourly_sha256': sha256(cfg.HOURLY), 'catalog_file': catalog.name,
                'output_rows': len(hourly), 'clean_trips': sum(p['clean_rows'] for p in manifest_files),
                'deduplication': 'Exact operational duplicates removed; conflicting/repeated IDs rejected',
                'arrival_boundary': 'Jan 1 incomplete for starts in Dec 2024; train begins Feb 2025'}
    cfg.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return hourly, manifest


def load_prepared():
    manifest = json.loads(cfg.MANIFEST.read_text())
    if sha256(cfg.HOURLY) != manifest['hourly_sha256']:
        raise ValueError('El agregado horario no corresponde al manifiesto de P1.')
    return pd.read_parquet(cfg.HOURLY), manifest


def ensure_raw_data(raw_dir=cfg.RAW_DIR, catalog=cfg.CATALOG):
    """Descarga solo los meses fijados en config y verifica el manifiesto previo.

    Se reutiliza el descargador del equipo. El catálogo de septiembre no se
    sustituye por el más reciente, pues cambiaría la versión de datos.
    """
    from src.data.p2_download_raw_data import (
        DownloadItem, FALLBACK_TRIPS, FALLBACK_NOMENCLATURA_URL,
        create_resilient_session, download_item,
    )
    raw_dir, catalog = Path(raw_dir), Path(catalog)
    expected = json.loads(cfg.MANIFEST.read_text()) if cfg.MANIFEST.exists() else {}
    hashes = {item['file']: item['sha256'] for item in expected.get('files', [])}
    if expected.get('catalog_sha256'):
        hashes[catalog.name] = expected['catalog_sha256']
    sources = [(raw_dir / f'datos_abiertos_{month.year}_{month.month:02d}.csv',
                FALLBACK_TRIPS.get((month.year, month.month)))
               for month in pd.period_range(cfg.START, cfg.END, freq='M')]
    sources.append((catalog, FALLBACK_NOMENCLATURA_URL
                    if catalog.name == Path(FALLBACK_NOMENCLATURA_URL).name else None))
    with create_resilient_session() as session:
        for path, url in sources:
            if not path.exists():
                if not url:
                    raise ValueError(f'No hay fuente oficial configurada para {path.name}')
                item = DownloadItem(filename=path.name, url=url, file_type='trip')
                if not download_item(item, path.parent, session, organize_subdirs=False):
                    raise RuntimeError(f'No se pudo descargar {path.name}; no se inicia entrenamiento.')
            if path.name in hashes and sha256(path) != hashes[path.name]:
                raise ValueError(f'{path.name} difiere del manifiesto versionado; revisa la fuente.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=cfg.RAW_DIR)
    parser.add_argument('--catalog', type=Path, default=cfg.CATALOG)
    parser.add_argument('--download-missing', action='store_true')
    args = parser.parse_args()
    if args.download_missing:
        ensure_raw_data(args.raw_dir, args.catalog)
    prepare_dataset(args.raw_dir, args.catalog)
