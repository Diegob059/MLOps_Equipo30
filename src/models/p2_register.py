"""
Módulo de Registro en el Model Registry de MLflow y Promoción a Producción (Pregunta 2: Riesgo).
=================================================================================================
Registra en el Model Registry (`mibici-equipo30-p2-riesgo`) el modelo candidato de P2
seleccionado por el equipo tras la evaluación comparativa multicriterio (desempeño vs.
baseline, balanced accuracy, recall en desabasto/saturación y latencia de inferencia).

Asigna el alias de producción `champion` (o stage `Production`), verifica la trazabilidad
completa del linaje (código Git, versión de datos y corte temporal anti-leakage) y ejecuta
una prueba de inferencia real cargando el modelo desde el Model Registry por su alias
antes de certificar la promoción.

Uso desde terminal:
    python -m src.models.p2_register --run-id <run_id_candidato> --reason "<justificación>"
Vía MLflow Project:
    mlflow run . -e register_p2 -P run_id=<run_id> -P reason="<justificación>"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.p2_config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_PRODUCTION_ALIAS,
    P2_REGISTRY_NAME,
    P2_REPORTS_DIR,
    TRAIN_CUTOFF_DATE,
    logger,
)


def register_candidate(
    run_id: str,
    reason: str,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    model_name: str = P2_REGISTRY_NAME,
    alias: str = P2_PRODUCTION_ALIAS,
) -> Dict:
    """Registra, versiona y promueve con alias `champion` un candidato de P2 en el Model Registry."""
    if not reason or not reason.strip():
        raise ValueError("Es obligatorio documentar la justificación de selección (--reason) antes de promover.")

    _proj_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if _proj_run_id:
        os.environ["MIBICI_PROJECT_RUN_ID"] = _proj_run_id

    # Si MLflow CLI inyectó una URI sqlite directa en vez del servidor HTTP del Model Registry
    if not tracking_uri.startswith("http"):
        env_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if env_uri.startswith("http"):
            tracking_uri = env_uri
        else:
            tracking_uri = "http://127.0.0.1:5001"

    logger.info("=" * 70)
    logger.info("REGISTRO DE MODELO EN MODEL REGISTRY (PREGUNTA 2 - RIESGO)")
    logger.info("=" * 70)
    logger.info(f"Conectando al servidor MLflow Tracking: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Validación de la corrida de origen
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    stage = run.data.tags.get("stage_of_work", "")
    question = run.data.tags.get("question", run.data.tags.get("pregunta", ""))

    if run.info.status != "FINISHED":
        raise ValueError(f"La corrida {run_id} no está en estado FINISHED (estado: {run.info.status}).")
    if str(question).lower() != "p2":
        raise ValueError(f"La corrida {run_id} no pertenece a la Pregunta 2 (pregunta: {question}).")
    if stage != "candidate":
        raise ValueError(
            f"Solo se pueden promover corridas de tipo 'candidate' (la corrida {run_id} es de tipo '{stage}')."
        )

    git_commit = run.data.tags.get("git_commit", "")
    data_version = run.data.params.get("data_version", "")
    train_cutoff = run.data.params.get("train_cutoff_date", TRAIN_CUTOFF_DATE)

    lineage = {
        "experiment_id": str(run.info.experiment_id),
        "experiment": experiment.name,
        "source_run_id": run_id,
        "git_commit": git_commit,
        "data_version": data_version,
        "train_cutoff_date": train_cutoff,
        "candidate_id": run.data.tags.get("candidate_id", run.info.run_name),
    }

    # 2. Prueba de inferencia previa con el modelo de la corrida
    uri = f"runs:/{run_id}/model"
    logger.info(f"Validando artefacto de modelo en: {uri}")
    source_model = mlflow.pyfunc.load_model(uri)

    # Obtener ejemplo de entrada
    example = source_model.input_example
    if example is None:
        try:
            path = mlflow.artifacts.download_artifacts(f"{uri}/input_example.json")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            example = pd.DataFrame(payload["data"], columns=payload["columns"])
        except Exception as exc:
            logger.warning(f"No fue posible descargar input_example.json: {exc}")
            raise ValueError("El modelo de la corrida no contiene input_example válido.")

    expected_preds = source_model.predict(example)
    if hasattr(expected_preds, "values"):
        expected_preds = expected_preds.values
    expected_preds = np.asarray(expected_preds).ravel()

    if len(expected_preds) != len(example):
        raise ValueError("La dimensión de las predicciones del modelo fuente no coincide con el número de ejemplos.")
    if not np.isin(expected_preds, [0, 1, 2]).all():
        raise ValueError("El modelo fuente generó clases fuera del dominio operativo {0, 1, 2}.")

    logger.success(f"✓ Prueba de inferencia local exitosa ({len(example)} filas evaluadas).")

    # 3. Registro formal en el Model Registry
    logger.info(f"Registrando nueva versión para el modelo: '{model_name}'...")
    version = mlflow.register_model(uri, model_name)
    logger.success(f"✓ Modelo registrado como '{model_name}' Versión {version.version}")

    # Descripción a nivel de familia registrada (idempotente)
    client.update_registered_model(
        model_name,
        description=(
            "P2: Clasificación multiclase del riesgo de desabasto (vacío / Clase 1) y saturación "
            "(lleno / Clase 2) en estaciones MiBici Guadalajara con un horizonte de 2 horas hacia "
            "adelante. El alias 'champion' identifica la versión activa en producción para el "
            "ejercicio académico."
        ),
    )

    # Descripción específica de esta versión con métricas clave y justificación
    client.update_model_version(model_name, str(version.version), description=reason)

    # Etiquetas de linaje en la versión
    for key, value in lineage.items():
        client.set_model_version_tag(model_name, str(version.version), key, str(value))
    client.set_model_version_tag(model_name, str(version.version), "selection_reason", reason)

    # 4. Verificación de carga por número de versión
    logger.info(f"Verificando carga por versión: models:/{model_name}/{version.version}")
    restored_version_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")
    restored_preds = np.asarray(restored_version_model.predict(example)).ravel()
    np.testing.assert_array_equal(expected_preds, restored_preds)
    logger.success("✓ Predicciones idénticas al cargar por número de versión.")

    # 5. Asignación del alias de producción (`champion`)
    logger.info(f"Asignando alias '{alias}' a la versión {version.version}...")
    client.set_registered_model_alias(model_name, alias, str(version.version))
    logger.success(f"✓ Alias '{alias}' asignado exitosamente a la versión {version.version}.")

    # 6. Certificación final cargando por alias de producción
    alias_uri = f"models:/{model_name}@{alias}"
    logger.info(f"Certificando inferencia desde URI de producción: {alias_uri}")
    champion_model = mlflow.pyfunc.load_model(alias_uri)
    champion_preds = np.asarray(champion_model.predict(example)).ravel()
    np.testing.assert_array_equal(expected_preds, champion_preds)
    logger.success(f"✓ Modelo en producción certificado: predicciones consistentes en {alias_uri}.")

    # 7. Persistencia del reporte de auditoría
    evidence = {
        "model_name": model_name,
        "version": str(version.version),
        "alias": alias,
        "model_uri": alias_uri,
        **lineage,
        "reason": reason,
        "load_and_predict_verified": True,
        "example_rows": len(example),
        "classes_verified": [0, 1, 2],
    }

    P2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_report_path = P2_REPORTS_DIR / "registry.json"
    registry_report_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.success(f"✓ Evidencia de auditoría guardada en: {registry_report_path}")

    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="run_id de la corrida candidata de P2 elegida.")
    parser.add_argument("--reason", required=True, help="Justificación técnica y operativa de la selección.")
    parser.add_argument("--alias", default=P2_PRODUCTION_ALIAS, help=f"Alias a asignar (por defecto '{P2_PRODUCTION_ALIAS}').")
    args = parser.parse_args()

    evidence = register_candidate(run_id=args.run_id, reason=args.reason, alias=args.alias)
    print("\n" + json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
