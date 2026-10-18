"""Rejilla horaria completa y rezagos disponibles a medianoche para P1."""
import pandas as pd
from src import p1_config as cfg
from src.data.p1_dataset import load_prepared, sha256


def january_stations(hourly):
    timestamp = hourly['timestamp']
    return sorted(hourly.loc[(timestamp < cfg.TRAIN_START) &
                              (hourly['salidas'] > 0), 'estacion'].unique().tolist())


def choose_peaks(hourly):
    """Selecciona tres horas consecutivas entre 06:00 y 12:00 usando solo 2025."""
    history = hourly[hourly['timestamp'] < cfg.VALIDATION_START]
    index = pd.date_range(cfg.START, '2025-12-31 23:00:00', freq='h')
    daily = history.groupby('timestamp')['salidas'].sum().reindex(index, fill_value=0)
    profile = daily.groupby([daily.index.dayofweek < 5, daily.index.hour]).mean()
    peaks = {}
    for weekday in [True, False]:
        windows = {h: profile.xs(weekday, level=0).loc[h:h + cfg.PEAK_DURATION - 1].sum()
                   for h in range(6, 12 - cfg.PEAK_DURATION + 1)}
        start = max(windows, key=windows.get)
        peaks['Laborable' if weekday else 'Fin de semana'] = {
            'inicio_pico': start, 'hora_fin': start + cfg.PEAK_DURATION}
    return pd.DataFrame.from_dict(peaks, orient='index')


def make_features(hourly, stations, start=cfg.START, end=cfg.END):
    """Cada fila corresponde a una estación, fecha y hora (00–11)."""
    index = pd.MultiIndex.from_product([stations, pd.date_range(start, end), range(cfg.HOURS)],
                                       names=['estacion', 'fecha', 'hora'])
    events = hourly.copy()
    events['fecha'] = events['timestamp'].dt.normalize()
    events['hora'] = events['timestamp'].dt.hour
    table = events.set_index(['estacion', 'fecha', 'hora'])[cfg.FLOWS]
    table = table.reindex(index, fill_value=0).astype('float32').reset_index()
    for flow in cfg.FLOWS:
        group = table.groupby(['estacion', 'hora'])[flow]
        table[f'{flow}_ayer'] = group.shift(1)
        table[f'{flow}_semana'] = group.shift(7)
    table['dia_semana'] = table['fecha'].dt.dayofweek.astype('int8')
    table['mes'] = table['fecha'].dt.month.astype('int8')
    return table.dropna(subset=cfg.VARIABLES)


def build_features():
    hourly, manifest = load_prepared()
    table = make_features(hourly, january_stations(hourly))
    cfg.FEATURES.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg.FEATURES, index=False)
    import json
    manifest['features_sha256'] = sha256(cfg.FEATURES)
    cfg.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'Características P1: {len(table):,} filas', flush=True)
    return table


if __name__ == '__main__':
    build_features()
