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
)


@pytest.fixture
def sample_station_grid():
    """Genera una serie temporal sintética para 1 estación durante 10 horas."""
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=10, freq="h")
    df = pd.DataFrame({
        "station_id": [1] * 10,
        "timestamp": timestamps,
        "capacity": [20] * 10,
        "salidas": [2, 3, 5, 1, 0, 8, 2, 1, 0, 0],
        "llegadas": [1, 1, 2, 4, 10, 1, 0, 1, 0, 0],
    })
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
