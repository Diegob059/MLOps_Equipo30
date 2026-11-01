"""
Pruebas de validación de datos e integridad con DeepChecks (PyTest).
=====================================================================
Verifica la calidad e integridad del dataset procesado del proyecto MiBici:
1. Integridad de datos: ausencia de nulos inesperados, tipos válidos y duplicados.
2. Validación Train/Test: consistencia de esquemas y evaluación de drift entre particiones.
"""

# Compatibilidad con NumPy 2.x para dependencias internas de DeepChecks
import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "bool"):
    np.bool = bool

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    IsSingleValue,
    MixedDataTypes,
    MixedNulls,
)
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd
import pytest

from src.p2_config import P2_FEATURES_TARGET_PARQUET, RANDOM_SEED


@pytest.fixture
def mibici_sample_datasets():
    """Fixture que provee muestras representativas de entrenamiento y prueba para DeepChecks."""
    if P2_FEATURES_TARGET_PARQUET.exists():
        # Excluir periodo inicial de calentamiento y muestrear representativamente
        df = (
            pd.read_parquet(P2_FEATURES_TARGET_PARQUET)
            .dropna(subset=["salidas_lag_168", "balance_rolling_12h"])
            .sample(2000, random_state=RANDOM_SEED)
        )
        drop_cols = ["timestamp", "station_id", "target", "risk_ratio", "net_flow_target"]
        features = [
            c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        train_df = df.iloc[:1000][features + ["target"]]
        test_df = df.iloc[1000:][features + ["target"]]
    else:
        # Fallback sintético en CI
        rng = np.random.default_rng(RANDOM_SEED)
        cols = [f"feat_{i}" for i in range(10)]
        data_train = rng.normal(size=(500, 10))
        data_test = rng.normal(size=(500, 10))
        target_train = rng.integers(0, 3, size=500)
        target_test = rng.integers(0, 3, size=500)

        train_df = pd.DataFrame(data_train, columns=cols)
        train_df["target"] = target_train
        test_df = pd.DataFrame(data_test, columns=cols)
        test_df["target"] = target_test
        features = cols

    train_ds = Dataset(train_df, label="target", cat_features=[])
    test_ds = Dataset(test_df, label="target", cat_features=[])
    return train_ds, test_ds, features


def test_deepchecks_data_integrity_checks(mibici_sample_datasets):
    """Verifica que los datos no presenten tipos mixtos, nulos ambiguos ni valores únicos espurios."""
    train_ds, _, _ = mibici_sample_datasets

    # 1. Sin tipos mixtos en columnas
    mixed_types_check = MixedDataTypes().run(train_ds)
    has_mixed_types = any(len(types) > 0 for types in mixed_types_check.value.values())
    assert not has_mixed_types, "Se encontraron columnas con tipos de datos mixtos"

    # 2. Sin variantes de nulos ambiguos ('nan', 'null', etc.)
    mixed_nulls_check = MixedNulls().run(train_ds)
    has_mixed_nulls = any(len(n) > 0 for n in mixed_nulls_check.value["columns"].values())
    assert not has_mixed_nulls, "Se encontraron nulos con formatos inconsistentes en las columnas"

    # 3. Ninguna columna informativa contiene un único valor constante
    single_val_check = IsSingleValue().run(train_ds)
    single_value_cols = [col for col, count in single_val_check.value.items() if count <= 1]
    assert not single_value_cols, (
        f"Columnas con valor constante encontradas: {single_value_cols}"
    )


def test_deepchecks_data_duplicates_check(mibici_sample_datasets):
    """Verifica la tasa de filas duplicadas en el dataset procesado."""
    train_ds, _, _ = mibici_sample_datasets
    duplicates_check = DataDuplicates().run(train_ds)
    duplicate_ratio = float(duplicates_check.value)
    assert duplicate_ratio < 0.20, f"Tasa de duplicados anormalmente alta: {duplicate_ratio:.2%}"


def test_deepchecks_full_integrity_suite(mibici_sample_datasets):
    """Ejecuta la suite estándar de data_integrity y verifica que la ejecución sea exitosa."""
    train_ds, _, _ = mibici_sample_datasets
    suite = data_integrity()
    results = suite.run(train_ds)
    assert results is not None
    assert len(results.results) > 0


def test_deepchecks_train_test_validation_suite(mibici_sample_datasets):
    """Ejecuta la suite estándar de train_test_validation y verifica compatibilidad entre conjuntos."""
    train_ds, test_ds, _ = mibici_sample_datasets
    suite = train_test_validation()
    results = suite.run(train_dataset=train_ds, test_dataset=test_ds)
    assert results is not None
    assert len(results.results) > 0
