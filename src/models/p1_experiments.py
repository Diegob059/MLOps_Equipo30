"""Experimentación de P1 desde el entry point train del MLflow Project.

    mlflow run . -e train -P question=p1 --env-manager local
"""
import json
import os
import subprocess
import time
from itertools import product

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from src import p1_config as cfg
from src.data.p1_dataset import load_prepared, sha256
from src.features.p1_features import choose_peaks
from src.models.p1_balance import calculate_balance, summarize_balance
from src.models.p1_estimator import FlowRegressor, model_input
from src.models.p1_train import regression_metrics


def temporal_split(data):
    partitions = [data[data.fecha.between(start, end)].copy() for start, end in [
        (cfg.TRAIN_START, cfg.TRAIN_END),
        (cfg.VALIDATION_START, cfg.VALIDATION_END), (cfg.TEST_START, cfg.END)]]
    if any(table.empty for table in partitions):
        raise ValueError('P1 requiere datos en las tres particiones temporales.')
    return partitions


def evaluate(table, predicted, peaks, margin=None):
    frame = table[['estacion', 'fecha', 'hora', *cfg.FLOWS]].copy()
    frame[predicted.columns] = predicted
    observed = summarize_balance(calculate_balance(frame, peaks, 'salidas', 'llegadas'))
    forecast = summarize_balance(calculate_balance(frame, peaks))
    inventory = observed[['necesidad']].rename(columns={'necesidad': 'observada'})
    inventory['prevista'] = forecast['necesidad']
    if margin is None:
        margin = max(0.0, float((inventory.observada - inventory.prevista).quantile(
            cfg.MARGIN_QUANTILE)))
    inventory['referencia'] = np.ceil(inventory.prevista + margin)
    metrics = {f'inventario_{key}': value for key, value in regression_metrics(
        inventory.observada, inventory.prevista).items()}
    metrics['cobertura_conteos_sin_limite_capacidad'] = float(
        (inventory.referencia >= inventory.observada).mean())
    for flow in cfg.FLOWS:
        metrics.update({f'{flow}_{key}': value for key, value in regression_metrics(
            frame[flow], predicted[f'{flow}_previstas']).items()})
    monthly = inventory.assign(error=(inventory.observada - inventory.prevista).abs())
    monthly = monthly.groupby(monthly.index.get_level_values('fecha').to_period('M')).error.mean()
    metrics['inventario_mae_mensual_std'] = float(monthly.std(ddof=0))
    metrics['inventario_mae_peor_mes'] = float(monthly.max())
    return metrics, margin, inventory, monthly


def select_candidate(rows):
    """Decide solo con validación; desempata con estabilidad, complejidad y nombre."""
    eligible = [row for row in rows if row['stage_of_work'] == 'candidate']
    if not eligible:
        raise ValueError('No hay candidatos para seleccionar.')
    return min(eligible, key=lambda row: (
        row['validation_inventario_mae'], row['validation_inventario_mae_peor_mes'],
        row.get('max_depth', 0), row['candidate']))


def _plots(table, predicted, inventory, monthly, run_id):
    directory = cfg.ADVANCED_REPORTS / 'plots' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    sample = table.index[::max(1, len(table) // 3000)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, flow in zip(axes, cfg.FLOWS):
        axis.scatter(table.loc[sample, flow], predicted.loc[sample, f'{flow}_previstas'],
                     s=4, alpha=0.2)
        high = max(table.loc[sample, flow].max(), predicted.loc[sample, f'{flow}_previstas'].max())
        axis.plot([0, high], [0, high], '--', color='gray')
        axis.set(title=flow.capitalize(), xlabel='Real', ylabel='Predicho')
    fig.suptitle('Prueba: flujos horarios (muestra determinista)')
    fig.tight_layout()
    fig.savefig(directory / 'real_predicho.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(inventory.observada - inventory.prevista, bins=50)
    axes[0].set(title='Prueba: residuales de inventario', xlabel='Observado − previsto',
                ylabel='Estaciones/día')
    monthly.plot.bar(ax=axes[1], rot=30)
    axes[1].set(title='Prueba: estabilidad mensual', xlabel='Mes', ylabel='MAE inventario')
    fig.tight_layout()
    fig.savefig(directory / 'residuales_estabilidad.png', dpi=130)
    plt.close(fig)
    mlflow.log_artifacts(str(directory), artifact_path='plots')


def _log_candidate(name, method, params, train, validation, test, peaks, manifest,
                   lineage, baseline=None, nested=False, dataset=None):
    stage = 'baseline' if method == 'semanal' else 'candidate'
    with mlflow.start_run(run_name=f'p1_{name}', nested=nested) as run:
        mlflow.set_tags({**lineage, 'question': 'p1', 'pregunta': 'P1',
                         'model_family': method, 'stage_of_work': stage})
        model = FlowRegressor(method=method, **params)
        start = time.perf_counter()
        model.fit(model_input(train), train[cfg.FLOWS])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_prediction = model.predict(model_input(validation))
        predict_ms = (time.perf_counter() - start) / len(validation) * 1_000_000
        val_metrics, margin, _, _ = evaluate(validation, val_prediction, peaks)
        test_prediction = model.predict(model_input(test))
        test_metrics, _, inventory, monthly = evaluate(test, test_prediction, peaks, margin)
        configuration = {
            'candidate': name, 'method': method, **model.effective_params_,
            'data_version': manifest['data_version'], 'features_sha256': manifest['features_sha256'],
            'train_start': cfg.TRAIN_START, 'train_end': cfg.TRAIN_END,
            'validation_start': cfg.VALIDATION_START, 'validation_end': cfg.VALIDATION_END,
            'test_start': cfg.TEST_START, 'data_end': cfg.END,
            'margin_quantile': cfg.MARGIN_QUANTILE, 'margin': margin,
            'forecast_origin': '00:00 local; siguiente manana', 'interval_hours': 1,
            'cohort': 'origenes_enero_2025', 'stations': int(train.estacion.nunique()),
            'within_hour_order': 'departures_before_arrivals',
            'selection_metric': 'validation_inventario_mae',
            'validation_reused_for_selection_and_margin': True,
            'train_rows': len(train), 'validation_rows': len(validation), 'test_rows': len(test),
            'variables': cfg.VARIABLES, 'peaks': peaks.to_dict(orient='index'),
        }
        mlflow.log_params({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                           for key, value in configuration.items()})
        metrics = {**{f'validation_{k}': v for k, v in val_metrics.items()},
                   **{f'test_{k}': v for k, v in test_metrics.items()},
                   'fit_seconds': fit_seconds, 'predict_ms_per_1000_rows': predict_ms}
        if baseline is not None:
            for partition in ('validation', 'test'):
                key = f'{partition}_inventario_mae'
                reference = baseline[key]
                metrics[f'{partition}_mejora_inventario_pct'] = (
                    100 * (reference - metrics[key]) / reference if reference else 0.0)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cfg.MANIFEST), artifact_path='data')
        mlflow.log_input(dataset, context='training')
        mlflow.log_dict(configuration, 'config/ejecucion.json')
        for path in ('requirements.txt', 'python_env.yaml', 'MLproject', 'src/p1_config.py'):
            mlflow.log_artifact(str(cfg.ROOT / path), artifact_path='config')
        for path in ('src/models/p1_experiments.py', 'src/models/p1_estimator.py',
                     'src/models/p1_balance.py', 'src/models/p1_train.py',
                     'src/features/p1_features.py', 'src/data/p1_dataset.py', 'scripts/train.py'):
            mlflow.log_artifact(str(cfg.ROOT / path), artifact_path='source/' +
                                (cfg.ROOT / path).relative_to(cfg.ROOT).parent.as_posix())
        # Se especifica serialization_format='cloudpickle' explícitamente: en MLflow 3, si la librería
        # 'skops' está instalada en el entorno, MLflow intenta usar skops por defecto. Sin embargo,
        # la clase personalizada FlowRegressor (src.models.p1_estimator) y los nodos internos de
        # árboles de decisión (sklearn.tree._tree.Tree) no pertenecen a la lista blanca de skops,
        # arrojando UntrustedTypesFoundException. Cloudpickle permite serializar y deserializar
        # de forma íntegra clases custom y closures de Python sin restricciones artificiales.
        example = model_input(validation.head(5))
        info = mlflow.sklearn.log_model(
            sk_model=model, name='model', input_example=example,
            signature=infer_signature(example, model.predict(example)),
            code_paths=[str(cfg.ROOT / 'src')],
            pip_requirements=[f'mlflow=={mlflow.__version__}',
                           f'numpy=={np.__version__}', f'pandas=={pd.__version__}',
                           f'scikit-learn=={__import__("sklearn").__version__}'],
            serialization_format="cloudpickle")
        _plots(test, test_prediction, inventory, monthly, run.info.run_id)
        return {'candidate': name, 'stage_of_work': stage, 'model_family': method,
                **model.effective_params_, **metrics, 'run_id': run.info.run_id,
                'model_uri': info.model_uri, 'data_version': manifest['data_version']}


def run_experiments(model=None, tree_params=None):
    hourly, manifest = load_prepared()
    if not cfg.FEATURES.exists() or sha256(cfg.FEATURES) != manifest.get('features_sha256'):
        raise ValueError('Ejecuta mlflow run . -e ingest -P question=p1 primero.')
    train, validation, test = temporal_split(pd.read_parquet(cfg.FEATURES))
    peaks = choose_peaks(hourly)
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)
    lineage = {'equipo': 'Equipo 30', 'author': 'Gregorio Escobedo',
               'git_commit': subprocess.check_output(
                   ['git', 'rev-parse', 'HEAD'], cwd=cfg.ROOT, text=True).strip(),
               'git_dirty': str(bool(subprocess.check_output(
                   ['git', 'status', '--porcelain'], cwd=cfg.ROOT, text=True).strip())),
               'project_run_id': os.getenv('MIBICI_PROJECT_RUN_ID', ''),
               'data_version': manifest['data_version']}
    # MLflow guarda metadatos/digest del dataset; los viajes crudos no son artefactos.
    dataset = mlflow.data.from_pandas(train[['estacion', *cfg.VARIABLES, *cfg.FLOWS]],
        source=cfg.FEATURES.as_uri(), name=f'p1-train-{manifest["data_version"]}',
        digest=manifest['features_sha256'][:8])
    args = (train, validation, test, peaks, manifest, lineage)
    cfg.ADVANCED_REPORTS.mkdir(parents=True, exist_ok=True)
    if model:
        if model not in cfg.METHODS:
            raise ValueError(f'P1 soporta: {cfg.METHODS}')
        row = _log_candidate(model, model, (tree_params or {}) if model == 'arbol' else {},
                             *args, dataset=dataset)
        return pd.DataFrame([row])
    baseline = _log_candidate('baseline_semanal_avanzado', 'semanal', {}, *args, dataset=dataset)
    rows = [baseline, _log_candidate('historico_avanzado', 'historico', {}, *args,
                                     baseline=baseline, dataset=dataset)]
    grid = cfg.TREE_GRID if tree_params is None else {
        key: [tree_params.get(key, cfg.TREE_PARAMS[key])] for key in cfg.TREE_GRID}
    with mlflow.start_run(run_name='p1_busqueda_arboles_inventario') as parent:
        mlflow.set_tags({**lineage, 'question': 'p1', 'model_family': 'arbol', 'stage_of_work': 'search'})
        mlflow.log_dict(grid, 'config/grid.json')
        for depth, leaf in product(grid['max_depth'], grid['min_samples_leaf']):
            params = {'max_depth': depth, 'min_samples_leaf': leaf}
            rows.append(_log_candidate(f'arbol_d{depth}_l{leaf}', 'arbol', params, *args,
                                       baseline=baseline, nested=True, dataset=dataset))
        winner = select_candidate(rows)
        summary = {'selected_run_id': winner['run_id'], 'candidate': winner['candidate'],
                   'model_uri': winner['model_uri'], 'git_commit': lineage['git_commit'],
                   'git_dirty': lineage['git_dirty'], 'data_version': manifest['data_version'],
                   'search_run_id': parent.info.run_id,
                   'selection_metric': 'validation_inventario_mae',
                   'selection_uses_test': False, 'baseline_run_id': baseline['run_id']}
        mlflow.set_tag('selected_run_id', winner['run_id'])
        mlflow.log_dict(summary, 'reports/seleccion.json')
        comparison = pd.DataFrame(rows)
        comparison.to_csv(cfg.ADVANCED_REPORTS / 'comparacion.csv', index=False)
        (cfg.ADVANCED_REPORTS / 'seleccion.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        mlflow.log_artifact(str(cfg.ADVANCED_REPORTS / 'comparacion.csv'), artifact_path='reports')
    print(comparison[['candidate', 'validation_inventario_mae', 'test_inventario_mae',
                      'test_mejora_inventario_pct', 'run_id']].to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return comparison


# Permite ejecutar directamente el barrido completo de experimentos de P1 desde la terminal
# (ej. python -m src.models.p1_experiments) además de ser invocado por scripts/train.py.
if __name__ == '__main__':
    run_experiments()
