import pandas as pd

from src.data.dataset import limpiar_viajes, obtener_estaciones_comunes


def _viaje(viaje_id, archivo, origen, destino, genero="M", anio=1990, inicio="2025-01-01 08:00:00", minutos=10.0):
    """Construye una fila de viaje sintética con las mismas columnas que usa `dataset.py`."""
    inicio_ts = pd.Timestamp(inicio)
    return {
        "viaje_id": viaje_id,
        "usuario_id": 1,
        "genero": genero,
        "anio_nacimiento": anio,
        "inicio": inicio_ts,
        "fin": inicio_ts + pd.Timedelta(minutes=minutos),
        "origen_id": origen,
        "destino_id": destino,
        "archivo": archivo,
    }


def test_obtener_estaciones_comunes_excluye_estaciones_nuevas():
    viajes = pd.DataFrame(
        [
            _viaje(1, "2025_01", 1, 2),
            _viaje(2, "2025_02", 1, 2),
            _viaje(3, "2025_02", 1, 3),  # la estación 3 solo aparece a partir del mes 2
        ]
    )
    assert obtener_estaciones_comunes(viajes) == {1, 2}


def test_limpiar_viajes_elimina_duplicados_exactos():
    fila = _viaje(1, "2025_01", 1, 2)
    viajes = pd.DataFrame([fila, dict(fila)])
    limpios = limpiar_viajes(viajes)
    assert len(limpios) == 1


def test_limpiar_viajes_elimina_filas_con_genero_o_anio_nulo():
    viajes = pd.DataFrame(
        [
            _viaje(1, "2025_01", 1, 2, genero=None),
            _viaje(2, "2025_01", 1, 2, anio=None),
            _viaje(3, "2025_01", 1, 2),
        ]
    )
    limpios = limpiar_viajes(viajes)
    assert limpios["viaje_id"].tolist() == [3]


def test_limpiar_viajes_filtra_por_duracion():
    viajes = pd.DataFrame(
        [
            _viaje(1, "2025_01", 1, 2, minutos=0.5),  # muy corto
            _viaje(2, "2025_01", 1, 2, minutos=200),  # muy largo
            _viaje(3, "2025_01", 1, 2, minutos=10),  # dentro del rango
        ]
    )
    limpios = limpiar_viajes(viajes, duracion_minima_min=1, duracion_maxima_min=180)
    assert limpios["viaje_id"].tolist() == [3]
    assert limpios.loc[0, "duracion_min"] == 10


def test_limpiar_viajes_filtra_por_estaciones_comunes():
    viajes = pd.DataFrame(
        [
            _viaje(1, "2025_01", 1, 2),
            _viaje(2, "2025_01", 1, 99),  # la estación 99 no es común a todos los meses
        ]
    )
    limpios = limpiar_viajes(viajes, estaciones_comunes={1, 2})
    assert limpios["viaje_id"].tolist() == [1]
