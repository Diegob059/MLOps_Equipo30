"""
Validación de integridad y calidad de datos con DeepChecks.
=============================================================
Chequeos simples y genéricos (sin reglas de negocio por pregunta):

- `data_integrity()`: valores faltantes, duplicados, tipos de datos mixtos,
  columnas de un solo valor, correlación entre features, etc.
- `train_test_validation()`: *drift* entre el conjunto de entrenamiento y el
  de prueba (solo para P1 y P2, que tienen partición temporal real; P3 es
  clustering no supervisado sobre todas las estaciones, sin train/test).

Reutiliza la misma partición temporal que ya usa cada módulo de entrenamiento
(`p1_train.py`, `p2_features.get_train_test_data`, `p3_train.load_feature_set`)
en vez de redefinirla aquí.

Ejecución desde terminal:
    python -m src.data.validate_data --question p1
    python -m src.data.validate_data --question p2
    python -m src.data.validate_data --question p3
"""

import argparse

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
from loguru import logger

from src.config import RANDOM_SEED, REPORTS_DIR


def _cargar_p1():
    from src import p1_config as cfg

    data = pd.read_parquet(cfg.FEATURES)
    train = data[data["fecha"].between(cfg.TRAIN_START, "2025-12-31")]
    test = data[data["fecha"].between(cfg.TEST_START, cfg.END)]
    return train, test, None


def _cargar_p2():
    from src.features.p2_features import get_train_test_data

    X_train, y_train, X_test, y_test, _, _ = get_train_test_data()
    train = X_train.assign(target=y_train)
    test = X_test.assign(target=y_test)
    return train, test, "target"


def _cargar_p3():
    from src.models.p3_train import load_feature_set

    df, _ = load_feature_set("hourly")
    return df, None, None  # Sin partición train/test: clustering no supervisado.


CARGADORES = {"p1": _cargar_p1, "p2": _cargar_p2, "p3": _cargar_p3}


def validar(question: str, muestra: int = 20_000) -> None:
    """Corre los chequeos de DeepChecks para `question` y guarda reportes HTML en `reports/`.

    `muestra` limita cuántas filas se validan como máximo: algunos datasets (p. ej. P2) tienen
    millones de filas, y correr DeepChecks sobre todas haría la validación extremadamente lenta
    sin aportar mucho más valor que una muestra aleatoria representativa.
    """
    train, test, label = CARGADORES[question]()
    if len(train) > muestra:
        train = train.sample(n=muestra, random_state=RANDOM_SEED)
    if test is not None and len(test) > muestra:
        test = test.sample(n=muestra, random_state=RANDOM_SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ds_train = Dataset(train, label=label) if label else Dataset(train)
    resultado_integridad = data_integrity().run(ds_train)
    salida_integridad = REPORTS_DIR / f"{question}_deepchecks_integridad.html"
    resultado_integridad.save_as_html(str(salida_integridad))
    logger.success(f"[{question}] Reporte de integridad: {salida_integridad}")

    if test is not None:
        ds_test = Dataset(test, label=label) if label else Dataset(test)
        resultado_drift = train_test_validation().run(ds_train, ds_test)
        salida_drift = REPORTS_DIR / f"{question}_deepchecks_train_test.html"
        resultado_drift.save_as_html(str(salida_drift))
        logger.success(f"[{question}] Reporte de train/test (drift): {salida_drift}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True, choices=list(CARGADORES), help="p1, p2 o p3.")
    parser.add_argument(
        "--muestra", type=int, default=20_000, help="Máximo de filas a validar por conjunto (default: 20000)."
    )
    args = parser.parse_args()
    validar(args.question, muestra=args.muestra)


if __name__ == "__main__":
    main()
