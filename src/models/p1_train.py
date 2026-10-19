"""Entrena P1 sobre una evaluación común y registra resultados en MLflow.

python -m src.models.p1_train
"""
import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src import p1_config as cfg
from src.data.p1_dataset import load_prepared, sha256
from src.features.p1_features import choose_peaks, january_stations
from src.models.p1_balance import calculate_balance, summarize_balance, read_catalog, recommend


def regression_metrics(real, predicted):
    return {'mae': float(mean_absolute_error(real, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(real, predicted)))}


def fit_models(train):
    models = {}
    for flow in cfg.FLOWS:
        tree = DecisionTreeRegressor(**cfg.TREE_PARAMS)
        tree.fit(train[cfg.VARIABLES], train[flow])
        lookup = train.groupby(['estacion', 'dia_semana', 'hora'])[flow].mean()
        models[flow] = {'tree': tree, 'lookup': lookup, 'fallback': float(train[flow].mean())}
    return models


def predict_flows(table, models, selected=None):
    out = table.copy()
    keys = pd.MultiIndex.from_frame(out[['estacion', 'dia_semana', 'hora']])
    for flow in cfg.FLOWS:
        out[f'{flow}_semanal'] = out[f'{flow}_semana']
        out[f'{flow}_arbol'] = models[flow]['tree'].predict(out[cfg.VARIABLES])
        # El promedio histórico también incluye las horas con cero de la rejilla.
        out[f'{flow}_historico'] = models[flow]['lookup'].reindex(keys).fillna(
            models[flow]['fallback']).to_numpy()
        if selected:
            out[f'{flow}_previstas'] = out[f'{flow}_{selected[flow]}']
    return out


def flow_metrics(table, method):
    result = {}
    for flow in cfg.FLOWS:
        for metric, value in regression_metrics(table[flow], table[f'{flow}_{method}']).items():
            result[f'{flow}_{metric}'] = value
    return result


def evaluate_stock(validation, test, peaks, method):
    val_balance = calculate_balance(validation, peaks, f'salidas_{method}', f'llegadas_{method}')
    test_balance = calculate_balance(test, peaks, f'salidas_{method}', f'llegadas_{method}')
    val_pred = summarize_balance(val_balance)
    val_real = summarize_balance(calculate_balance(validation, peaks, 'salidas', 'llegadas'))
    margin = max(0.0, float((val_real['necesidad'] - val_pred['necesidad']).quantile(cfg.MARGIN_QUANTILE)))
    real = summarize_balance(calculate_balance(test, peaks, 'salidas', 'llegadas'))
    result = summarize_balance(test_balance).rename(columns={'necesidad': 'necesidad_prevista'})
    result['necesidad_observada'] = real['necesidad']
    result['sin_contar_llegadas'] = test_balance.groupby(['estacion', 'fecha'])[f'salidas_{method}'].sum()
    result['bicicletas_referencia'] = np.ceil(result['necesidad_prevista'] + margin).astype(int)
    metrics = {f'inventario_{k}': v for k, v in regression_metrics(
        result['necesidad_observada'], result['necesidad_prevista']).items()}
    metrics['cobertura_conteos_sin_limite_capacidad'] = float((
        result['bicicletas_referencia'] >= result['necesidad_observada']).mean())
    return result, margin, metrics


def log_experiments(scores, bundle, manifest):
    import mlflow
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cfg.ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=cfg.ROOT, text=True).strip())
    runs = []
    names = {'semanal': 'p1_baseline_semanal_balance', 'historico': 'p1_baseline_historico_balance',
             'arbol': 'p1_arboles_balance', 'seleccionado': 'p1_inventario_inicial'}
    for method in [*cfg.METHODS, 'seleccionado']:
        with mlflow.start_run(run_name=names[method]) as run:
            mlflow.set_tags({'equipo': 'Equipo 30', 'pregunta': 'P1', 'author': 'Gregorio Escobedo',
                             'git_commit': commit, 'git_dirty': str(dirty),
                             'target': 'inventario_horario_conservador', 'code_snapshot': 'source/'})
            mlflow.log_params({'metodo': method, 'data_version': manifest['data_version'],
                'train_start': cfg.TRAIN_START, 'validation_start': cfg.VALIDATION_START,
                'test_start': cfg.TEST_START, 'data_end': cfg.END,
                'forecast_origin': '00:00 local; siguiente manana', 'interval_hours': 1,
                'cohort': 'origenes_enero_2025', 'stations': len(bundle['stations']),
                'features': json.dumps(cfg.VARIABLES), 'tree_params': json.dumps(cfg.TREE_PARAMS),
                'margin_quantile': cfg.MARGIN_QUANTILE, 'margin': bundle['margins'][method],
                'selected_methods': json.dumps(bundle['selected']),
                'peak_weekday': str(bundle['peaks'].loc['Laborable'].to_dict()),
                'peak_weekend': str(bundle['peaks'].loc['Fin de semana'].to_dict()),
                'within_hour_order': 'departures_before_arrivals',
                'validation_reused_for_selection_and_margin': True})
            for partition in ['validation', 'test']:
                row = scores[(scores['metodo'] == method) & (scores['particion'] == partition)].iloc[0]
                metrics = {f'{partition}_{key}': float(value) for key, value in row.items()
                           if key not in ['metodo', 'particion'] and pd.notna(value)}
                mlflow.log_metrics(metrics)
            for path in [cfg.MANIFEST, cfg.REPORTS / 'metricas.csv', cfg.REPORTS / 'resumen.json',
                         cfg.REPORTS / 'franjas.csv', cfg.ROOT / 'requirements-p1.txt',
                         cfg.ROOT / 'requirements-p1-lock.txt']:
                mlflow.log_artifact(str(path), artifact_path='reports')
            # Snapshot del código ejecutado: también identifica cambios aún sin commit.
            source = [cfg.ROOT / 'src/p1_config.py', cfg.ROOT / 'src/data/p1_dataset.py',
                      cfg.ROOT / 'src/features/p1_features.py', *Path(__file__).parent.glob('p1_*.py'),
                      cfg.ROOT / 'src/visualization/p1_plots.py']
            for path in source:
                mlflow.log_artifact(str(path), artifact_path='source/' + str(path.relative_to(cfg.ROOT).parent))
            if method == 'seleccionado':
                mlflow.log_artifact(str(cfg.MODEL), artifact_path='models')
                mlflow.log_artifacts(str(cfg.FIGURES), artifact_path='figures')
                mlflow.log_artifact(str(cfg.REPORTS / 'sugerencia_ejemplo.csv'), artifact_path='reports')
            runs.append({'metodo': method, 'run_id': run.info.run_id, 'experiment': cfg.EXPERIMENT})
    (cfg.REPORTS / 'mlflow_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    return runs


def run_pipeline(log_mlflow=True, catalog=cfg.CATALOG):
    hourly, manifest = load_prepared()
    if not cfg.FEATURES.exists() or sha256(cfg.FEATURES) != manifest.get('features_sha256'):
        raise ValueError('Genera las características con python -m src.features.p1_features')
    if sha256(catalog) != manifest['catalog_sha256']:
        raise ValueError('El catálogo no corresponde al manifiesto de P1.')
    data = pd.read_parquet(cfg.FEATURES)
    train = data[data['fecha'].between(cfg.TRAIN_START, '2025-12-31')]
    validation = data[data['fecha'].between(cfg.VALIDATION_START, '2026-06-30')]
    test = data[data['fecha'].between(cfg.TEST_START, cfg.END)]
    models = fit_models(train)
    validation, test = predict_flows(validation, models), predict_flows(test, models)
    selected = {}
    for flow in cfg.FLOWS:
        selected[flow] = min(cfg.METHODS, key=lambda m: mean_absolute_error(
            validation[flow], validation[f'{flow}_{m}']))
        for table in [validation, test]:
            table[f'{flow}_seleccionado'] = table[f'{flow}_{selected[flow]}']
            table[f'{flow}_previstas'] = table[f'{flow}_seleccionado']
    peaks = choose_peaks(hourly)
    rows, margins, evaluations = [], {}, {}
    for method in [*cfg.METHODS, 'seleccionado']:
        evaluation, margin, stock = evaluate_stock(validation, test, peaks, method)
        evaluations[method], margins[method] = evaluation, margin
        for partition, table in [('validation', validation), ('test', test)]:
            metrics = flow_metrics(table, method)
            if partition == 'test':
                metrics.update(stock)
            rows.append({'metodo': method, 'particion': partition, **metrics})
    scores = pd.DataFrame(rows)
    bundle = {'models': models, 'selected': selected, 'stations': january_stations(hourly),
              'peaks': peaks, 'margin': margins['seleccionado'], 'margins': margins,
              'data_version': manifest['data_version'], 'variables': cfg.VARIABLES,
              'training_end': '2025-12-31', 'calibration_end': '2026-06-30',
              'history_complete_through': cfg.END}
    cfg.MODEL.parent.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.MODEL)
    example_day = pd.Timestamp(cfg.END)
    predictions = test[test['fecha'] == example_day]
    recommendation = recommend(predictions, hourly, bundle, example_day, read_catalog(catalog))
    summary = {'selected_methods': selected, 'margin': bundle['margin'], 'stations': len(bundle['stations']),
               'data_version': manifest['data_version'], 'example_date': cfg.END,
               'states': recommendation['estado'].value_counts().to_dict(),
               'training_rows': len(train), 'validation_rows': len(validation), 'test_rows': len(test)}
    scores.to_csv(cfg.REPORTS / 'metricas.csv', index=False)
    peaks.to_csv(cfg.REPORTS / 'franjas.csv')
    recommendation.to_csv(cfg.REPORTS / 'sugerencia_ejemplo.csv', index=False)
    evaluations['seleccionado'].to_csv(cfg.REPORTS / 'evaluacion_inventario_prueba.csv')
    # Conservar los flujos de prueba para la comparación visual; están fuera de Git.
    test.to_parquet(cfg.ROOT / 'data/processed/p1_predicciones_prueba.parquet', index=False)
    (cfg.REPORTS / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    from src.visualization.p1_plots import create_plots
    create_plots(hourly, train, test, evaluations['seleccionado'], recommendation, bundle)
    if log_mlflow:
        log_experiments(scores, bundle, manifest)
    print(scores.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return scores, recommendation, bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=cfg.CATALOG)
    parser.add_argument('--no-mlflow', action='store_true', help='Ejecutar solo el análisis local, sin tracking')
    args = parser.parse_args()
    run_pipeline(log_mlflow=not args.no_mlflow, catalog=args.catalog)
