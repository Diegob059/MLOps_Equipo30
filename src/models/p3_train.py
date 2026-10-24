"""
Módulo de Modelado, Entrenamiento y Registro en MLflow (Pregunta 3: Clustering).
================================================================================
Lee los dos conjuntos de features de estación producidos por
``src/features/features.py`` (perfil horario completo y perfil por franjas),
escala cada uno de forma independiente, y recorre modelos K-Means y Gaussian
Mixture para k = 2..10 en cada conjunto (36 combinaciones). Esa búsqueda se
registra como *nested runs* de MLflow (un run padre por búsqueda, un run hijo
por combinación), y se registran además en MLflow:

- *Baseline* sencillo: K-Means con k=4 fijo sobre el perfil horario completo,
  sin comparar hiperparámetros (análogo al primer acercamiento ingenuo).
- *Candidatos*: las 3 combinaciones (conjunto de features, algoritmo, k) más
  diversas y con mayor coeficiente de silueta del barrido (una por cada
  combinación de algoritmo/conjunto de features, ver `seleccionar_candidatos`),
  cada una registrada como un run completo e independiente.

Ejecución desde terminal:
    python -m src.models.p3_train
    python -m src.models.p3_train --algoritmo kmeans --k 4   # un solo run, sin el barrido completo
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import (
    END_TIMESTAMP,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    START_TIMESTAMP,
)
from src.data.dataset import get_current_data_version
from src.features.features import METADATA_COLS
from src.visualization.p3_plots import (
    graficar_mapa_calor_perfil_clusters,
    graficar_mapa_clusters,
    graficar_seleccion_k,
)

app = typer.Typer(help="CLI de entrenamiento y registro en MLflow para la Pregunta 3 (clustering de estaciones).")

CONJUNTOS_CARACTERISTICAS = ("hourly", "bucketed")
ALGORITMOS = ("kmeans", "gmm")
K_MINIMO, K_MAXIMO = 2, 10
K_BASELINE = 4

P3_MODEL_PATH = MODELS_DIR / "p3_cluster_model.pkl"
P3_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "p3_estaciones_clusters.parquet"
P3_MAP_FIGURE = FIGURES_DIR / "p3_clusters_estaciones_mapa.png"
P3_PROFILE_FIGURE = FIGURES_DIR / "p3_perfil_clusters.png"
P3_K_SELECTION_FIGURE = FIGURES_DIR / "p3_seleccion_k.png"
P3_REPORT_PATH = REPORTS_DIR / "p3_clustering_report.txt"


@dataclass
class ResultadoModelo:
    """Todo lo necesario para registrar en MLflow, guardar o graficar un modelo ajustado."""

    modelo: object
    escalador: StandardScaler
    conjunto_caracteristicas: str
    algoritmo: str
    k: int
    columnas_caracteristicas: list
    etiquetas: np.ndarray
    df: pd.DataFrame


def get_git_commit() -> str:
    """Obtiene el hash del commit actual de Git (para la etiqueta `git_commit` en MLflow)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_feature_set(name: str, processed_dir: Path = PROCESSED_DATA_DIR) -> tuple[pd.DataFrame, list]:
    """Carga un conjunto de features de estación escrito por `src/features/features.py`."""
    path = processed_dir / f"estaciones_caracteristicas_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero `make data_p3` "
            f"(o `python -m src.data.dataset && python -m src.features.features`)."
        )
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    logger.info(f"Conjunto de features '{name}': {df.shape[0]} estaciones x {len(feature_cols)} features ({path.name})")
    return df, feature_cols


def fit_and_evaluate(X: np.ndarray, algoritmo: str, k: int, random_state: int = RANDOM_SEED):
    """Ajusta un modelo de clustering y calcula sus métricas de calidad."""
    if algoritmo == "kmeans":
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = modelo.fit_predict(X)
        extra = {"inercia": float(modelo.inertia_)}
    elif algoritmo == "gmm":
        modelo = GaussianMixture(n_components=k, random_state=random_state)
        etiquetas = modelo.fit_predict(X)
        extra = {"bic": float(modelo.bic(X)), "aic": float(modelo.aic(X))}
    else:
        raise ValueError(f"algoritmo debe ser 'kmeans' o 'gmm', recibido: {algoritmo!r}")

    metricas = {
        "test_silhouette_score": float(silhouette_score(X, etiquetas)),
        "test_davies_bouldin_index": float(davies_bouldin_score(X, etiquetas)),
        **extra,
    }
    return modelo, etiquetas, metricas


def ejecutar_barrido(
    conjuntos_caracteristicas: tuple = CONJUNTOS_CARACTERISTICAS,
    algoritmos: tuple = ALGORITMOS,
    rango_k: range = range(K_MINIMO, K_MAXIMO + 1),
    processed_dir: Path = PROCESSED_DATA_DIR,
    data_version: str = "",
    git_commit: str = "",
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    parent_run_name: str = "p3_busqueda_hiperparametros_clustering",
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], ResultadoModelo], str]:
    """Recorre (conjunto_caracteristicas x algoritmo x k) y registra cada combinación como
    un *run anidado* de MLflow dentro de un run padre que representa la búsqueda completa
    (ver rúbrica: "al menos una búsqueda de hiperparámetros organizada como nested runs").

    Los runs hijos solo registran parámetros/métricas/etiquetas (sin artefactos pesados);
    los candidatos que se promueven a partir de este barrido (ver `seleccionar_candidatos`)
    son los que reciben el registro completo con modelo y gráficas.

    Regresa `(resultados, ajustados, parent_run_id)`: `resultados`/`ajustados` se usan para
    elegir el baseline y los candidatos finales; `parent_run_id` permite enlazar cada
    candidato promovido con la búsqueda de la que salió.
    """
    filas = []
    ajustados: dict[tuple[str, str, int], ResultadoModelo] = {}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags(
            {
                "equipo": "Equipo 30",
                "pregunta": "P3",
                "question": "p3",
                "git_commit": git_commit,
                "stage_of_work": "search",
                "problem_type": "unsupervised_clustering",
                "system": "MiBici Guadalajara",
            }
        )
        mlflow.log_params(
            {
                "conjuntos_caracteristicas": ",".join(conjuntos_caracteristicas),
                "algoritmos": ",".join(algoritmos),
                "k_min": rango_k.start,
                "k_max": rango_k.stop - 1,
                "data_version": data_version,
                "random_seed": RANDOM_SEED,
            }
        )

        for conjunto_caracteristicas in conjuntos_caracteristicas:
            df, columnas_caracteristicas = load_feature_set(conjunto_caracteristicas, processed_dir)
            escalador = StandardScaler()
            X = escalador.fit_transform(df[columnas_caracteristicas])

            for algoritmo in algoritmos:
                for k in rango_k:
                    modelo, etiquetas, metricas = fit_and_evaluate(X, algoritmo, k)
                    filas.append({"conjunto_caracteristicas": conjunto_caracteristicas, "algoritmo": algoritmo, "k": k, **metricas})
                    ajustados[(conjunto_caracteristicas, algoritmo, k)] = ResultadoModelo(
                        modelo=modelo,
                        escalador=escalador,
                        conjunto_caracteristicas=conjunto_caracteristicas,
                        algoritmo=algoritmo,
                        k=k,
                        columnas_caracteristicas=columnas_caracteristicas,
                        etiquetas=etiquetas,
                        df=df,
                    )
                    with mlflow.start_run(run_name=f"trial_{conjunto_caracteristicas}_{algoritmo}_k{k}", nested=True):
                        mlflow.log_params(
                            {
                                "algorithm": algoritmo,
                                "k": k,
                                "feature_set": conjunto_caracteristicas,
                                "scaling": "StandardScaler",
                                "data_version": data_version,
                            }
                        )
                        mlflow.log_metrics(metricas)
                        mlflow.set_tags(
                            {
                                "equipo": "Equipo 30",
                                "pregunta": "P3",
                                "question": "p3",
                                "git_commit": git_commit,
                                "model_family": algoritmo,
                                "stage_of_work": "search",
                                "problem_type": "unsupervised_clustering",
                            }
                        )

        resultados = pd.DataFrame(filas).rename(columns={"test_silhouette_score": "silueta"})

        # Figura resumen de toda la búsqueda: se registra en el run padre (no en cada hijo ni
        # en cada candidato, ya que compara las 36 combinaciones, no una sola).
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_k = graficar_seleccion_k(resultados)
        fig_k.savefig(P3_K_SELECTION_FIGURE, dpi=150)
        mlflow.log_artifact(str(P3_K_SELECTION_FIGURE), artifact_path="plots")

        parent_run_id = parent_run.info.run_id

    return resultados, ajustados, parent_run_id


def construir_baseline(processed_dir: Path = PROCESSED_DATA_DIR, k: int = K_BASELINE) -> ResultadoModelo:
    """K-Means con k fijo (sin ajuste de hiperparámetros) sobre el perfil horario: el *baseline* sencillo de P3."""
    df, columnas_caracteristicas = load_feature_set("hourly", processed_dir)
    escalador = StandardScaler()
    X = escalador.fit_transform(df[columnas_caracteristicas])
    modelo, etiquetas, metricas = fit_and_evaluate(X, algoritmo="kmeans", k=k)
    return ResultadoModelo(
        modelo=modelo,
        escalador=escalador,
        conjunto_caracteristicas="hourly",
        algoritmo="kmeans",
        k=k,
        columnas_caracteristicas=columnas_caracteristicas,
        etiquetas=etiquetas,
        df=df,
    ), metricas


def seleccionar_mejor(resultados: pd.DataFrame, ajustados: dict[tuple[str, str, int], ResultadoModelo]) -> tuple[ResultadoModelo, dict]:
    """Elige la combinación con mayor coeficiente de silueta del barrido."""
    mejor_fila = resultados.sort_values("silueta", ascending=False).iloc[0]
    clave = (mejor_fila["conjunto_caracteristicas"], mejor_fila["algoritmo"], int(mejor_fila["k"]))
    resultado = ajustados[clave]
    metricas = {
        "test_silhouette_score": float(mejor_fila["silueta"]),
        "test_davies_bouldin_index": float(mejor_fila["test_davies_bouldin_index"]),
    }
    return resultado, metricas


def seleccionar_candidatos(
    resultados: pd.DataFrame, ajustados: dict[tuple[str, str, int], ResultadoModelo], n: int = 3
) -> list[tuple[ResultadoModelo, dict]]:
    """Elige hasta `n` candidatos diversos del barrido para promover a runs completos de MLflow.

    A diferencia de `seleccionar_mejor` (un solo ganador global), primero toma el mejor `k`
    dentro de cada combinación (conjunto_caracteristicas, algoritmo) -por defecto, 4 grupos:
    hourly/kmeans, hourly/gmm, bucketed/kmeans, bucketed/gmm- y luego ordena esos "mejores
    por grupo" por silueta descendente, quedándose con los `n` más altos. Esto garantiza que
    los candidatos promovidos varíen de verdad en algoritmo y conjunto de features (no solo
    en k dentro de la misma combinación), tal como pide la rúbrica.
    """
    mejores_por_grupo = (
        resultados.sort_values("silueta", ascending=False)
        .groupby(["conjunto_caracteristicas", "algoritmo"], as_index=False)
        .first()
    )
    top_n = mejores_por_grupo.sort_values("silueta", ascending=False).head(n)

    candidatos = []
    for _, fila in top_n.iterrows():
        clave = (fila["conjunto_caracteristicas"], fila["algoritmo"], int(fila["k"]))
        resultado = ajustados[clave]
        metricas = {
            "test_silhouette_score": float(fila["silueta"]),
            "test_davies_bouldin_index": float(fila["test_davies_bouldin_index"]),
        }
        candidatos.append((resultado, metricas))
    return candidatos


def _build_params_tags(resultado: ResultadoModelo, data_version: str, git_commit: str, stage_of_work: str) -> tuple[dict, dict]:
    """Arma los parámetros y etiquetas estándar de un run de P3 para MLflow.

    `stage_of_work` identifica el rol del run dentro del flujo de experimentación
    ("baseline", "search" para los hijos anidados del barrido, o "candidate" para
    los modelos promovidos), y junto con `question`/`model_family` permite filtrar
    y comparar runs en la interfaz de MLflow.
    """
    params = {
        "pregunta": "P3",
        "algorithm": resultado.algoritmo,
        "k": resultado.k,
        "feature_set": resultado.conjunto_caracteristicas,
        "scaling": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "date_range": f"{START_TIMESTAMP[:10]} to {END_TIMESTAMP[:10]}",
        "data_version": data_version,
        "n_stations": len(resultado.df),
        "n_features": len(resultado.columnas_caracteristicas),
    }
    tags = {
        "equipo": "Equipo 30",
        "pregunta": "P3",
        "question": "p3",
        "git_commit": git_commit,
        "problem_type": "unsupervised_clustering",
        "model_family": resultado.algoritmo,
        "stage_of_work": stage_of_work,
        "model_role": stage_of_work,
        "system": "MiBici Guadalajara",
    }
    return params, tags


def generar_artefactos(resultado_candidato: ResultadoModelo, metricas_candidato: dict) -> list:
    """Genera las figuras y el reporte de texto de un candidato; regresa la lista de artefactos (ruta, subcarpeta).

    La figura comparativa de selección de k (que resume las 36 combinaciones del
    barrido) ya no se genera aquí: se registra una sola vez en el run padre de la
    búsqueda (ver `ejecutar_barrido`), no repetida en cada candidato.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig_mapa = graficar_mapa_clusters(resultado_candidato.df, resultado_candidato.etiquetas)
    fig_mapa.savefig(P3_MAP_FIGURE, dpi=150)

    fig_perfil = graficar_mapa_calor_perfil_clusters(
        resultado_candidato.df, resultado_candidato.columnas_caracteristicas, resultado_candidato.etiquetas
    )
    fig_perfil.savefig(P3_PROFILE_FIGURE, dpi=150)

    tamanios = pd.Series(resultado_candidato.etiquetas).value_counts().sort_index()
    P3_REPORT_PATH.write_text(
        "PREGUNTA 3: CLUSTERING Y PERFILAMIENTO DE ESTACIONES (MIBICI)\n"
        "============================================================\n"
        f"Algoritmo               : {resultado_candidato.algoritmo} (k={resultado_candidato.k})\n"
        f"Conjunto de features    : {resultado_candidato.conjunto_caracteristicas}\n"
        f"Escalador               : StandardScaler\n"
        f"Total Estaciones        : {len(resultado_candidato.df)}\n"
        f"Silhouette Score        : {metricas_candidato['test_silhouette_score']:.4f}\n"
        f"Davies-Bouldin Score    : {metricas_candidato['test_davies_bouldin_index']:.4f}\n"
        f"Distribución Estaciones : {tamanios.to_dict()}\n",
        encoding="utf-8",
    )
    return [(P3_MAP_FIGURE, "figures"), (P3_PROFILE_FIGURE, "figures"), (P3_REPORT_PATH, "reports")]


def guardar_mejor_modelo(resultado_candidato: ResultadoModelo, output_path: Path = P3_MODEL_PATH) -> Path:
    """Persiste el modelo candidato (modelo + escalador + metadatos) para `src/models/p3_predict.py`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": resultado_candidato.modelo,
            "escalador": resultado_candidato.escalador,
            "conjunto_caracteristicas": resultado_candidato.conjunto_caracteristicas,
            "algoritmo": resultado_candidato.algoritmo,
            "k": resultado_candidato.k,
            "columnas_caracteristicas": resultado_candidato.columnas_caracteristicas,
        },
        output_path,
    )
    logger.success(f"Modelo guardado: {output_path}")
    return output_path


def guardar_asignaciones_cluster(resultado_candidato: ResultadoModelo, output_path: Path = P3_CLUSTERS_PARQUET) -> Path:
    """Escribe las asignaciones de cluster por estación del modelo candidato en `data/processed/`."""
    df = resultado_candidato.df.copy()
    df["cluster"] = resultado_candidato.etiquetas
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.success(f"Asignaciones de cluster guardadas: {output_path}")
    return output_path


def preparar_runs(processed_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """Ejecuta la búsqueda (con nested runs) + el baseline y arma todo lo necesario para
    registrar en MLflow: 1 baseline + 1 run padre de búsqueda (con sus hijos anidados) +
    >= 3 candidatos diversos (distintos algoritmo/conjunto de features).

    Regresa un dict con las llaves "baseline" (resultado/params/metrics/tags), "candidatos"
    (lista de esos mismos 4 campos, uno por candidato promovido), "artefactos" (figuras +
    reporte del mejor candidato, usado para los demás también) y "search_run_id" (el run
    padre de la búsqueda, para enlazar el linaje de cada candidato).
    """
    data_version = get_current_data_version()
    git_commit = get_git_commit()

    resultados_barrido, ajustados, parent_run_id = ejecutar_barrido(
        processed_dir=processed_dir, data_version=data_version, git_commit=git_commit
    )
    candidatos_crudos = seleccionar_candidatos(resultados_barrido, ajustados, n=3)
    resultado_baseline, metricas_baseline = construir_baseline(processed_dir=processed_dir)

    logger.success(
        f"{len(candidatos_crudos)} candidatos seleccionados de la búsqueda (run padre {parent_run_id}): "
        + "; ".join(
            f"{r.conjunto_caracteristicas}/{r.algoritmo}/k={r.k} (silhouette={m['test_silhouette_score']:.3f})"
            for r, m in candidatos_crudos
        )
    )

    params_baseline, tags_baseline = _build_params_tags(resultado_baseline, data_version, git_commit, stage_of_work="baseline")

    runs_candidatos = []
    for resultado, metricas in candidatos_crudos:
        params, tags = _build_params_tags(resultado, data_version, git_commit, stage_of_work="candidate")
        tags["search_run_id"] = parent_run_id
        runs_candidatos.append(
            {
                "run_name": f"p3_candidato_{resultado.conjunto_caracteristicas}_{resultado.algoritmo}_k{resultado.k}",
                "resultado": resultado,
                "params": params,
                "metrics": metricas,
                "tags": tags,
            }
        )

    # Artefactos ligeros compartidos por ahora (figuras + reporte del mejor candidato); el
    # siguiente paso de esta implementación amplía esto por candidato con log_model (firma +
    # ejemplo de entrada), log_input, manifiesto de datos y subcarpetas plots/data/config.
    mejor_resultado, mejor_metricas = candidatos_crudos[0]
    artefactos = generar_artefactos(mejor_resultado, mejor_metricas)

    return {
        "baseline": {
            "run_name": "p3_baseline_kmeans_clustering_estaciones",
            "resultado": resultado_baseline,
            "params": params_baseline,
            "metrics": metricas_baseline,
            "tags": tags_baseline,
        },
        "candidatos": runs_candidatos,
        "artefactos": artefactos,
        "search_run_id": parent_run_id,
    }


def registrar_run_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    artifacts: list,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Registra un run en MLflow (un solo backend; usado por el CLI independiente de este módulo)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        for file_path, artifact_subpath in artifacts:
            if Path(file_path).exists():
                mlflow.log_artifact(str(file_path), artifact_path=artifact_subpath)
    logger.info(f"Run '{run_name}' registrado en {tracking_uri} (experimento '{experiment_name}')")


def entrenar_y_registrar() -> dict:
    """Ejecuta el pipeline completo de P3 (búsqueda con nested runs + baseline + >= 3
    candidatos diversos) y los registra en MLflow."""
    runs = preparar_runs()
    registrar_run_mlflow(
        runs["baseline"]["run_name"], runs["baseline"]["params"], runs["baseline"]["metrics"],
        runs["baseline"]["tags"], runs["artefactos"],
    )
    for candidato in runs["candidatos"]:
        registrar_run_mlflow(candidato["run_name"], candidato["params"], candidato["metrics"], candidato["tags"], runs["artefactos"])

    mejor_resultado = runs["candidatos"][0]["resultado"]
    guardar_mejor_modelo(mejor_resultado)
    guardar_asignaciones_cluster(mejor_resultado)
    return runs


def entrenar_run_individual(
    algoritmo: str,
    k: int,
    conjunto_caracteristicas: str = "hourly",
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict:
    """Ajusta y registra un único run de P3 con `algoritmo`/`k`/`conjunto_caracteristicas` explícitos.

    Usado por el entry point `train` del `MLproject` (ver `scripts/train.py`),
    p. ej. `mlflow run . -e train -P question=p3 -P model=kmeans -P k=4`.
    A diferencia de `entrenar_y_registrar`, no recorre el barrido completo ni
    sobreescribe el modelo candidato serializado que usa `p3_predict.py`.
    """
    data_version = get_current_data_version()
    git_commit = get_git_commit()

    df, columnas_caracteristicas = load_feature_set(conjunto_caracteristicas, processed_dir)
    escalador = StandardScaler()
    X = escalador.fit_transform(df[columnas_caracteristicas])
    modelo, etiquetas, metricas = fit_and_evaluate(X, algoritmo, k)
    resultado = ResultadoModelo(
        modelo=modelo,
        escalador=escalador,
        conjunto_caracteristicas=conjunto_caracteristicas,
        algoritmo=algoritmo,
        k=k,
        columnas_caracteristicas=columnas_caracteristicas,
        etiquetas=etiquetas,
        df=df,
    )
    params, tags = _build_params_tags(resultado, data_version, git_commit, stage_of_work="candidate")
    tags["model_role"] = "single_run_override"
    run_name = f"p3_run_{conjunto_caracteristicas}_{algoritmo}_k{k}"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_mapa = graficar_mapa_clusters(resultado.df, resultado.etiquetas)
    fig_mapa.savefig(P3_MAP_FIGURE, dpi=150)
    fig_perfil = graficar_mapa_calor_perfil_clusters(
        resultado.df, resultado.columnas_caracteristicas, resultado.etiquetas
    )
    fig_perfil.savefig(P3_PROFILE_FIGURE, dpi=150)
    artefactos = [(P3_MAP_FIGURE, "figures"), (P3_PROFILE_FIGURE, "figures")]

    registrar_run_mlflow(run_name, params, metricas, tags, artefactos)
    return {"resultado": resultado, "params": params, "metrics": metricas, "tags": tags, "run_name": run_name}


@app.command()
def main(
    algoritmo: str = typer.Option(None, "--algoritmo", help="kmeans|gmm. Junto con --k, ejecuta un solo run en vez del barrido completo."),
    k: int = typer.Option(None, "--k", help="Número de clusters para el run individual (requerido junto con --algoritmo)."),
    conjunto_caracteristicas: str = typer.Option("hourly", "--conjunto-caracteristicas", help="hourly|bucketed (solo aplica al run individual)."),
):
    """Ejecuta el pipeline completo de P3 (barrido + baseline + candidato), o un solo run si se dan --algoritmo/--k."""
    if algoritmo is not None or k is not None:
        if algoritmo is None or k is None:
            raise typer.BadParameter("--algoritmo y --k deben especificarse juntos para un run individual.")
        entrenar_run_individual(algoritmo=algoritmo, k=k, conjunto_caracteristicas=conjunto_caracteristicas)
    else:
        entrenar_y_registrar()


if __name__ == "__main__":
    app()
