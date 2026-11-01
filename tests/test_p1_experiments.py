"""Regresiones: causalidad temporal, balance, selección y modelo portable de P1."""
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

from src import p1_config as cfg
from src.models.p1_estimator import FlowRegressor, model_input
from src.models.p1_experiments import (
    _log_candidate,
    evaluate,
    select_candidate,
    temporal_split,
)
from src.models.p1_train import evaluate_stock


def sample():
    table = pd.DataFrame({'estacion': [1, 1, 1, 2, 2, 2],
        'fecha': pd.to_datetime(['2026-01-01'] * 6), 'hora': [0, 1, 2, 0, 1, 2],
        'dia_semana': [3] * 6, 'mes': [1] * 6,
        'salidas_ayer': [10., 6., 3., 1., 2., 1.],
        'llegadas_ayer': [2., 4., 7., 0., 1., 0.],
        'salidas_semana': [8., 6., 3., 1., 1., 1.],
        'llegadas_semana': [2., 3., 5., 0., 1., 0.],
        'salidas': [10., 6., 3., 1., 2., 1.], 'llegadas': [2., 4., 7., 0., 1., 0.]})
    return table


def test_temporal_boundaries_exclude_gaps_and_never_overlap():
    dates = ['2025-01-31', cfg.TRAIN_START, cfg.TRAIN_END,
             cfg.VALIDATION_START, cfg.VALIDATION_END, cfg.TEST_START, cfg.END, '2026-10-01']
    train, val, test = temporal_split(pd.DataFrame({'fecha': pd.to_datetime(dates)}))
    assert [len(t) for t in (train, val, test)] == [2, 2, 2]
    assert train.fecha.max() < val.fecha.min() <= val.fecha.max() < test.fecha.min()


def test_selection_ignores_test_and_baseline_even_if_test_is_better():
    common = {'stage_of_work': 'candidate', 'validation_inventario_mae_peor_mes': 3.}
    rows = [{**common, 'candidate': 'a', 'validation_inventario_mae': 2., 'test_inventario_mae': 1.},
            {**common, 'candidate': 'b', 'validation_inventario_mae': 1., 'test_inventario_mae': 99.},
            {**common, 'candidate': 'base', 'validation_inventario_mae': 0., 'stage_of_work': 'baseline'}]
    assert select_candidate(rows)['candidate'] == 'b'


@pytest.mark.parametrize('method', cfg.METHODS)
def test_model_roundtrip_has_signature_and_requires_no_targets(tmp_path, method):
    data = sample()
    inputs = model_input(data)
    model = FlowRegressor(method=method, max_depth=2, min_samples_leaf=1).fit(inputs, data[cfg.FLOWS])
    predicted = model.predict(inputs)
    assert not set(cfg.FLOWS).intersection(inputs.columns)
    # serialization_format='cloudpickle' previene fallos por UntrustedTypesFoundException de skops
    # al persistir el custom estimator FlowRegressor y las instancias internas de DecisionTree.
    mlflow.sklearn.save_model(model, str(tmp_path / method), input_example=inputs.head(2),
        signature=infer_signature(inputs, predicted), pip_requirements=[],
        serialization_format="cloudpickle")
    restored = mlflow.pyfunc.load_model(str(tmp_path / method))
    pd.testing.assert_frame_equal(restored.predict(inputs), predicted)
    assert (predicted >= 0).all().all()


def test_inventory_metrics_match_previous_delivery_and_use_validation_margin():
    val = sample()
    test = sample().assign(fecha=pd.Timestamp('2026-07-02'))
    peaks = pd.DataFrame({'inicio_pico': [0, 0], 'hora_fin': [3, 3]},
                          index=['Laborable', 'Fin de semana'])
    model = FlowRegressor(method='semanal').fit(model_input(val), val[cfg.FLOWS])
    val_pred, test_pred = model.predict(model_input(val)), model.predict(model_input(test))
    _, margin, inventory, _ = evaluate(val, val_pred, peaks)
    metrics, _, _, _ = evaluate(test, test_pred, peaks, margin)
    for table, predicted in [(val, val_pred), (test, test_pred)]:
        for flow in cfg.FLOWS:
            table[f'{flow}_semanal'] = predicted[f'{flow}_previstas']
    _, old_margin, old_metrics = evaluate_stock(val, test, peaks, 'semanal')
    assert inventory.loc[(1, pd.Timestamp('2026-01-01')), 'observada'] == 14
    assert margin == pytest.approx(old_margin)
    assert metrics['inventario_mae'] == pytest.approx(old_metrics['inventario_mae'])
    assert metrics['cobertura_conteos_sin_limite_capacidad'] == pytest.approx(
        old_metrics['cobertura_conteos_sin_limite_capacidad'])


def test_historical_fallback_for_station_absent_from_training():
    data = sample()
    model = FlowRegressor(method='historico').fit(model_input(data), data[cfg.FLOWS])
    unseen = model_input(data.head(1)).assign(estacion=999)
    np.testing.assert_allclose(model.predict(unseen).iloc[0], data[cfg.FLOWS].mean())


def test_candidate_logging_restores_model_with_signature_and_target_free_example(tmp_path, monkeypatch):
    """Ejercita el registro real que fallaba por usar `example` sin definir."""
    import json

    import mlflow
    from mlflow import MlflowClient

    train = sample().assign(fecha=pd.Timestamp('2025-06-01'))
    validation = sample()
    test = sample().assign(fecha=pd.Timestamp('2026-07-01'))
    peaks = pd.DataFrame({'inicio_pico': [0, 0], 'hora_fin': [3, 3]},
                         index=['Laborable', 'Fin de semana'])
    manifest = {'data_version': 'fixture-p1', 'features_sha256': 'fixture-features'}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cfg, 'MANIFEST', manifest_path)
    monkeypatch.setattr(cfg, 'ADVANCED_REPORTS', tmp_path / 'reports')
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path / "tracking.db"}')
    try:
        mlflow.set_experiment('p1-candidate-regression')
        dataset = mlflow.data.from_pandas(model_input(train), name='fixture_train')
        result = _log_candidate(
            'regression', 'arbol', {'max_depth': 2, 'min_samples_leaf': 1},
            train, validation, test, peaks, manifest,
            {'git_commit': 'fixture-commit', 'git_dirty': 'False'}, dataset=dataset)
        client = MlflowClient()
        run = client.get_run(result['run_id'])
        assert run.info.status == 'FINISHED'
        assert run.data.tags['stage_of_work'] == 'candidate'
        assert np.isfinite(run.data.metrics['validation_inventario_mae'])

        model = mlflow.pyfunc.load_model(result['model_uri'])
        assert model.metadata.signature is not None
        assert model.metadata.saved_input_example_info is not None
        artifact = Path(mlflow.artifacts.download_artifacts(result['model_uri']))
        example = mlflow.models.Model.load(str(artifact / 'MLmodel')).load_input_example(str(artifact))
        assert len(example) == 5
        assert not set(cfg.FLOWS).intersection(example.columns)
        predictions = model.predict(example)
        assert list(predictions.columns) == ['salidas_previstas', 'llegadas_previstas']
        assert np.isfinite(predictions.to_numpy()).all()

        from src.models.p1_register import register_candidate

        monkeypatch.setattr(cfg, 'TRACKING_URI', mlflow.get_tracking_uri())
        monkeypatch.setattr(cfg, 'EXPERIMENT', 'p1-candidate-regression')
        monkeypatch.setattr(cfg, 'REGISTRY_NAME', 'p1-regression-demanda')
        evidence = register_candidate(result['run_id'], 'Prueba de registro desde LoggedModel')
        assert evidence['source_run_id'] == result['run_id']
        assert evidence['load_and_predict_verified'] is True
        champion = mlflow.pyfunc.load_model(evidence['model_uri'])
        pd.testing.assert_frame_equal(champion.predict(example), predictions)
    finally:
        mlflow.set_tracking_uri(previous_uri)
