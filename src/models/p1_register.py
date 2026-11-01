"""Registro explícito del candidato P1, promoción por alias y prueba de carga."""
import argparse
import json

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src import p1_config as cfg


def register_candidate(run_id, reason, alias=cfg.PRODUCTION_ALIAS):
    if not reason.strip():
        raise ValueError('Documenta la justificación de selección antes de promover.')
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    if (experiment.name != cfg.EXPERIMENT or run.info.status != 'FINISHED'
            or run.data.tags.get('question') != 'p1'
            or run.data.tags.get('stage_of_work') != 'candidate'):
        raise ValueError('Se requiere un candidato P1 terminado del experimento del equipo.')
    lineage = {'experiment_id': run.info.experiment_id, 'experiment': experiment.name,
               'source_run_id': run_id, 'git_commit': run.data.tags.get('git_commit', ''),
               'data_version': run.data.params.get('data_version', ''),
               'git_dirty': run.data.tags.get('git_dirty', '')}
    if not all(lineage[key] for key in ('git_commit', 'data_version')):
        raise ValueError('La corrida no conserva commit y versión de datos.')
    # Validar modelo y predicciones ANTES de asignar el alias de producción.
    # MLflow 3 almacena log_model como LoggedModel, fuera de artifacts/model.
    # Resolver desde la corrida conserva el linaje y no depende de un JSON local.
    outputs = run.outputs.model_outputs if run.outputs else []
    models = [client.get_logged_model(output.model_id) for output in outputs]
    models = [model for model in models if model.name == 'model']
    if len(models) > 1:
        raise ValueError('La corrida contiene varios modelos llamados model; selecciona una corrida inequívoca.')
    uri = models[0].model_uri if models else f'runs:/{run_id}/model'
    source_model = mlflow.pyfunc.load_model(uri)
    directory = mlflow.artifacts.download_artifacts(uri)
    example = source_model.metadata.load_input_example(directory)
    if example is None:
        raise ValueError('El modelo no conserva un ejemplo de entrada para verificar la inferencia.')
    expected = source_model.predict(example)
    if expected.shape != (len(example), 2) or not pd.DataFrame(expected).ge(0).all().all():
        raise ValueError('La prueba de inferencia de flujos falló.')
    version = mlflow.register_model(uri, cfg.REGISTRY_NAME)
    client.update_registered_model(cfg.REGISTRY_NAME, description=(
        'P1: pronóstico horario de salidas y llegadas de MiBici. El balance conservador '
        'convierte ambos flujos a inventario inicial; ver config/ejecucion.json. '
        'champion identifica la versión de producción para el ejercicio académico.'))
    client.update_model_version(cfg.REGISTRY_NAME, version.version, description=reason)
    for key, value in lineage.items():
        client.set_model_version_tag(cfg.REGISTRY_NAME, version.version, key, value)
    client.set_model_version_tag(cfg.REGISTRY_NAME, version.version, 'selection_reason', reason)
    # Una carga por versión permite detectar fallos antes de mover champion.
    restored = mlflow.pyfunc.load_model(f'models:/{cfg.REGISTRY_NAME}/{version.version}')
    pd.testing.assert_frame_equal(pd.DataFrame(expected), pd.DataFrame(restored.predict(example)))
    client.set_registered_model_alias(cfg.REGISTRY_NAME, alias, version.version)
    alias_uri = f'models:/{cfg.REGISTRY_NAME}@{alias}'
    champion = mlflow.pyfunc.load_model(alias_uri)
    pd.testing.assert_frame_equal(pd.DataFrame(expected), pd.DataFrame(champion.predict(example)))
    evidence = {'name': cfg.REGISTRY_NAME, 'version': version.version, 'alias': alias,
                'model_uri': alias_uri, **lineage, 'reason': reason,
                'load_and_predict_verified': True, 'example_rows': len(example)}
    cfg.ADVANCED_REPORTS.mkdir(parents=True, exist_ok=True)
    (cfg.ADVANCED_REPORTS / 'registry.json').write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    register_candidate(args.run_id, args.reason)


if __name__ == '__main__':
    main()
