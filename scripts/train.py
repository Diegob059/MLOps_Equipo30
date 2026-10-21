"""Entry point `train` del MLflow Project (ver `MLproject` en la raíz).

Despacha al módulo de entrenamiento de la pregunta indicada, pasando el
modelo/algoritmo y los hiperparámetros recibidos por CLI. Los valores por
defecto de rutas, fechas y semilla NO se repiten aquí: cada módulo de
entrenamiento ya los toma de su `config.py` correspondiente
(`src/config.py`, `src/p1_config.py` o `src/p2_config.py`); este script solo
traduce overrides explícitos de hiperparámetros cuando el usuario los da.

Uso:
    python scripts/train.py --question p1 --max-depth 8
    python scripts/train.py --question p2 --model lgbm --n-estimators 300
    python scripts/train.py --question p3 --model kmeans --k 4

Vía MLflow Project:
    mlflow run . -e train -P question=p1 -P max_depth=8
    mlflow run . -e train -P question=p2 -P model=lgbm -P n_estimators=300
    mlflow run . -e train -P question=p3 -P model=kmeans -P k=4
"""
import argparse
import os

# `mlflow run` crea su propio run "envoltorio" por invocación e inyecta
# MLFLOW_RUN_ID para que cualquier mlflow.start_run() anidado lo reutilice.
# Nuestros módulos de entrenamiento ya registran varios runs nombrados por
# ejecución (baseline + candidato, según la convención de la rúbrica), lo
# cual choca con ese run único. Se limpia la variable para que cada módulo
# siga creando sus propios runs con nombre, como lo hace fuera de `mlflow run`.
os.environ.pop("MLFLOW_RUN_ID", None)


def _clean(value):
    """Normaliza un valor de CLI recibido de `mlflow run`.

    En Windows, `mlflow run` sustituye los parámetros vacíos del `MLproject`
    envolviéndolos en comillas literales (p. ej. `--n-estimators ""`), y esas
    comillas no siempre se eliminan antes de llegar a argparse. Esta función
    quita un nivel de comillas envolventes para que un valor "vacío" siempre
    se traduzca a la cadena vacía real, sin importar la plataforma.
    """
    if value is None:
        return ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    return value


def _none_if_empty(value):
    cleaned = _clean(value)
    return cleaned or None


def train_p1(model: str, overrides: dict) -> None:
    if model:
        raise ValueError(
            f"P1 no acepta --model (compara internamente los métodos semanal/histórico/árbol); "
            f"recibido: {model!r}. Usa --max-depth/--min-samples-leaf para ajustar el árbol."
        )
    from src import p1_config as cfg
    from src.models.p1_train import run_pipeline

    tree_params = dict(cfg.TREE_PARAMS)
    if overrides.get("max_depth") is not None:
        tree_params["max_depth"] = int(overrides["max_depth"])
    if overrides.get("min_samples_leaf") is not None:
        tree_params["min_samples_leaf"] = int(overrides["min_samples_leaf"])
    run_pipeline(tree_params=tree_params)


def train_p2(model: str, overrides: dict) -> None:
    if model not in ("", "lgbm"):
        raise ValueError(f"P2 solo soporta --model lgbm (recibido: {model!r}).")
    from src.models.p2_train import train_and_evaluate
    from src.p2_config import LGBM_PARAMS

    lgbm_params = dict(LGBM_PARAMS)
    if overrides.get("n_estimators") is not None:
        lgbm_params["n_estimators"] = int(overrides["n_estimators"])
    if overrides.get("max_depth") is not None:
        lgbm_params["max_depth"] = int(overrides["max_depth"])
    if overrides.get("learning_rate") is not None:
        lgbm_params["learning_rate"] = float(overrides["learning_rate"])
    if overrides.get("num_leaves") is not None:
        lgbm_params["num_leaves"] = int(overrides["num_leaves"])
    train_and_evaluate(lgbm_params=lgbm_params)


def train_p3(model: str, overrides: dict) -> None:
    from src.models.p3_train import entrenar_run_individual, entrenar_y_registrar

    if not model:
        entrenar_y_registrar()
        return
    if overrides.get("k") is None:
        raise ValueError("P3 requiere -P k=<entero> cuando se especifica --model (algoritmo de clustering).")
    entrenar_run_individual(
        algoritmo=model,
        k=int(overrides["k"]),
        conjunto_caracteristicas=overrides.get("feature_set") or "hourly",
    )


DISPATCH = {"p1": train_p1, "p2": train_p2, "p3": train_p3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True, choices=list(DISPATCH), help="p1, p2 o p3.")
    parser.add_argument("--model", default="", help="Algoritmo/modelo (según la pregunta; vacío = pipeline por defecto).")
    parser.add_argument("--n-estimators", dest="n_estimators", default="", help="P2 (LightGBM).")
    parser.add_argument("--max-depth", dest="max_depth", default="", help="P1 (árbol) o P2 (LightGBM).")
    parser.add_argument("--min-samples-leaf", dest="min_samples_leaf", default="", help="P1 (árbol).")
    parser.add_argument("--learning-rate", dest="learning_rate", default="", help="P2 (LightGBM).")
    parser.add_argument("--num-leaves", dest="num_leaves", default="", help="P2 (LightGBM).")
    parser.add_argument("--k", default="", help="P3 (número de clusters para --model).")
    parser.add_argument("--feature-set", dest="feature_set", default="", help="P3: hourly|bucketed.")
    args = parser.parse_args()

    overrides = {
        "n_estimators": _none_if_empty(args.n_estimators),
        "max_depth": _none_if_empty(args.max_depth),
        "min_samples_leaf": _none_if_empty(args.min_samples_leaf),
        "learning_rate": _none_if_empty(args.learning_rate),
        "num_leaves": _none_if_empty(args.num_leaves),
        "k": _none_if_empty(args.k),
        "feature_set": _none_if_empty(args.feature_set),
    }
    DISPATCH[args.question](_clean(args.model), overrides)


if __name__ == "__main__":
    main()
