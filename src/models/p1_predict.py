"""Inferencia diaria usando únicamente movimientos de fechas anteriores.

python -m src.models.p1_predict --date 2026-10-01
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src import p1_config as cfg
from src.data.p1_dataset import load_prepared, sha256
from src.features.p1_features import january_stations, make_features
from src.models.p1_balance import read_catalog, recommend
from src.models.p1_train import predict_flows


def predict_day(date, catalog=cfg.CATALOG, output=None, model_uri=None):
    day = pd.Timestamp(date).normalize()
    hourly, manifest = load_prepared()
    registry_model = None
    if model_uri:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(cfg.TRACKING_URI)
        registry_model = mlflow.pyfunc.load_model(model_uri)
        run = MlflowClient().get_run(registry_model.metadata.run_id)
        if run.data.tags.get('question') != 'p1':
            raise ValueError('La URI debe corresponder a un modelo de P1.')
        params = run.data.params
        bundle = {'data_version': params['data_version'], 'stations': january_stations(hourly),
                  'calibration_end': params['validation_end'],
                  'history_complete_through': params['data_end'],
                  'peaks': pd.DataFrame.from_dict(json.loads(params['peaks']), orient='index'),
                  'margin': float(params['margin'])}
    else:
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
    if registry_model is not None:
        from src.models.p1_estimator import model_input
        predictions = features.copy()
        predicted = registry_model.predict(model_input(features))
        predictions[predicted.columns] = predicted
    else:
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
    parser.add_argument('--model-uri', help='Ejemplo: models:/mibici-equipo30-p1-demanda@champion')
    args = parser.parse_args()
    predict_day(args.date, args.catalog, args.output, args.model_uri)
