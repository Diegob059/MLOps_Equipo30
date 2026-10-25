"""
Pruebas unitarias para las transformaciones de ingeniería de características y target (Pregunta 2).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.p2_features import (
    add_cyclical_calendar_features,
    add_demand_lags,
    add_rolling_balance_features,
    compute_target_variable,
    get_train_test_data,
)


@pytest.fixture
def sample_station_grid():
    """Genera una serie temporal sintética para 1 estación durante 10 horas."""
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "station_id": [1] * 10,
            "timestamp": timestamps,
            "capacity": [20] * 10,
            "salidas": [2, 3, 5, 1, 0, 8, 2, 1, 0, 0],
            "llegadas": [1, 1, 2, 4, 10, 1, 0, 1, 0, 0],
        }
    )
    df["balance_neto"] = (df["llegadas"] - df["salidas"]).astype(np.int16)
    return df


def test_p2_compute_target_variable(sample_station_grid):
    """Verifica que el target a 2 horas compute correctamente las clases."""
    df_res = compute_target_variable(sample_station_grid)
    assert "delta_net" in df_res.columns
    assert "ratio" in df_res.columns
    assert "target" in df_res.columns

    # Las últimas 2 horas no tienen horizonte futuro observable y deben ser -1
    assert df_res.iloc[-1]["target"] == -1
    assert df_res.iloc[-2]["target"] == -1

    # Las clases válidas deben estar en {0, 1, 2}
    valid_targets = set(df_res.iloc[:-2]["target"].unique())
    assert valid_targets.issubset({0, 1, 2})


def test_p2_add_demand_lags(sample_station_grid):
    """Verifica el cálculo de retardos temporales y ausencia de fugas."""
    df_res, lag_cols = add_demand_lags(sample_station_grid, lags=[1, 2])
    assert "salidas_lag_1" in df_res.columns
    assert "llegadas_lag_2" in df_res.columns
    # El primer registro debe ser NaN para lag 1
    assert np.isnan(df_res.iloc[0]["salidas_lag_1"])
    # El valor del lag 1 en t=1 debe coincidir con el valor de t=0
    assert df_res.iloc[1]["salidas_lag_1"] == sample_station_grid.iloc[0]["salidas"]


def test_p2_add_rolling_balance_features(sample_station_grid):
    """Verifica que el balance rodante aplique shift(1) para evitar leakage."""
    df_res, roll_cols = add_rolling_balance_features(sample_station_grid, windows=[3])
    assert "balance_rolling_3h" in df_res.columns
    # En t=0 debe ser NaN porque shift(1) desplaza la primera observación
    assert np.isnan(df_res.iloc[0]["balance_rolling_3h"])


def test_p2_add_cyclical_calendar_features(sample_station_grid):
    """Verifica la generación de componentes cíclicas seno y coseno."""
    df_res, cyc_cols = add_cyclical_calendar_features(sample_station_grid)
    for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
        assert col in df_res.columns
        # Las funciones trigonométricas deben estar acotadas entre -1 y 1
        assert df_res[col].min() >= -1.0
        assert df_res[col].max() <= 1.0


def test_p2_get_train_test_data_anti_leakage(tmp_path):
    """
    Verifica que la partición temporal excluya de Train las filas cuya ventana
    futura cruce hacia el periodo de Test (evitando fuga de datos de enero 2026).
    """
    # Rango horario que cruza fin de año 2025 e inicio de 2026
    timestamps = pd.date_range("2025-12-31 18:00:00", "2026-01-01 06:00:00", freq="h")
    n = len(timestamps)
    df = pd.DataFrame(
        {
            "station_id": [1] * n,
            "timestamp": timestamps,
            "capacity": [20] * n,
            "salidas": [1] * n,
            "llegadas": [1] * n,
            "balance_neto": [0] * n,
            "delta_net": [0] * n,
            "ratio": [0.0] * n,
            "target": [0] * n,
            "salidas_lag_168": [1.0] * n,
            "feature_demo": np.arange(n),
        }
    )

    parquet_file = tmp_path / "test_features.parquet"
    df.to_parquet(parquet_file, index=False)

    X_train, y_train, X_test, y_test, feature_cols, df_clean = get_train_test_data(
        features_parquet=parquet_file,
        split_date="2026-01-01 00:00:00",
        horizon_hours=2,
        warmup_lag_col="salidas_lag_168",
    )

    # El corte de train debe ser < 2025-12-31 22:00:00 (horizon_hours=2)
    train_timestamps = df_clean.loc[X_train.index, "timestamp"]
    test_timestamps = df_clean.loc[X_test.index, "timestamp"]

    # 1. No debe existir ninguna fila en Train a partir de las 22:00 del 31 de diciembre
    assert (train_timestamps < pd.Timestamp("2025-12-31 22:00:00")).all()
    assert train_timestamps.max() == pd.Timestamp("2025-12-31 21:00:00")

    # 2. Las filas de las 22:00 y 23:00 del 31 de dic no deben estar ni en Train ni en Test
    excluded_mask = df_clean["timestamp"].isin(
        [pd.Timestamp("2025-12-31 22:00:00"), pd.Timestamp("2025-12-31 23:00:00")]
    )
    excluded_indices = df_clean[excluded_mask].index
    assert not any(idx in X_train.index for idx in excluded_indices)
    assert not any(idx in X_test.index for idx in excluded_indices)

    # 3. Test debe comenzar estrictamente en 2026-01-01 00:00:00
    assert (test_timestamps >= pd.Timestamp("2026-01-01 00:00:00")).all()
    assert test_timestamps.min() == pd.Timestamp("2026-01-01 00:00:00")

