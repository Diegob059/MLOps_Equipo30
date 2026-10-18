"""Inferencia diaria usando únicamente movimientos de fechas anteriores.

python -m src.models.p1_predict --date 2026-10-01
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src import p1_config as cfg
from src.data.p1_dataset import load_prepared, sha256
from src.features.p1_features import make_features
from src.models.p1_train import predict_flows
from src.models.p1_balance import recommend, read_catalog


def predict_day(date, catalog=cfg.CATALOG, output=None):
    day = pd.Timestamp(date).normalize()
    hourly, manifest = load_prepared()
    bundle = joblib.load(cfg.MODEL)
    if bundle['data_version'] != manifest['data_version']:
        raise ValueError('Los datos y el modelo pertenecen a versiones diferentes.')
    if sha256(catalog) != manifest['catalog_sha256']:
        raise ValueError('El catálogo no corresponde al manifiesto de P1.')
    last_day = pd.Timestamp(bundle['history_complete_through'])
    if day <= pd.Timestamp(bundle['calibration_end']) or day > last_day + pd.Timedelta(days=1):
        raise ValueError('La fecha debe ser posterior a la calibración y tener historia hasta ayer.')
    history = hourly[hourly['timestamp'] < day].copy()
    features = make_features(history, bundle['stations'], day - pd.Timedelta(days=7), day)
    features = features[features['fecha'] == day]
    predictions = predict_flows(features, bundle['models'], bundle['selected'])
    result = recommend(predictions, history, bundle, day, read_catalog(catalog))
    path = Path(output) if output else cfg.REPORTS / f'sugerencia_{day.date()}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f'Recomendación guardada en {path}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--catalog', type=Path, default=cfg.CATALOG)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    predict_day(args.date, args.catalog, args.output)
