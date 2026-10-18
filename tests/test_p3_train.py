import numpy as np
import pandas as pd
import pytest

from src.models.p3_train import fit_and_evaluate, seleccionar_mejor


def _generar_grupos_sinteticos():
    """Dos grupos sintéticos bien separados, fáciles de recuperar para cualquier algoritmo de clustering."""
    rng = np.random.default_rng(42)
    grupo_a = rng.normal(loc=0.0, scale=0.2, size=(20, 3))
    grupo_b = rng.normal(loc=5.0, scale=0.2, size=(20, 3))
    return np.vstack([grupo_a, grupo_b])


@pytest.mark.parametrize("algoritmo", ["kmeans", "gmm"])
def test_fit_and_evaluate_recupera_dos_clusters_separados(algoritmo):
    X = _generar_grupos_sinteticos()
    _modelo, etiquetas, metricas = fit_and_evaluate(X, algoritmo=algoritmo, k=2)

    assert len(etiquetas) == len(X)
    assert len(set(etiquetas)) == 2
    # Grupos bien separados deberían obtener un coeficiente de silueta cercano al máximo posible.
    assert metricas["test_silhouette_score"] > 0.9
    if algoritmo == "kmeans":
        assert "inercia" in metricas
    else:
        assert "bic" in metricas and "aic" in metricas


def test_fit_and_evaluate_rechaza_algoritmo_desconocido():
    X = _generar_grupos_sinteticos()
    with pytest.raises(ValueError):
        fit_and_evaluate(X, algoritmo="dbscan", k=2)


def test_seleccionar_mejor_elige_mayor_silueta():
    resultados = pd.DataFrame(
        [
            {
                "conjunto_caracteristicas": "hourly",
                "algoritmo": "kmeans",
                "k": 3,
                "silueta": 0.40,
                "test_davies_bouldin_index": 1.0,
            },
            {
                "conjunto_caracteristicas": "hourly",
                "algoritmo": "kmeans",
                "k": 4,
                "silueta": 0.55,
                "test_davies_bouldin_index": 0.9,
            },
            {
                "conjunto_caracteristicas": "bucketed",
                "algoritmo": "gmm",
                "k": 5,
                "silueta": 0.20,
                "test_davies_bouldin_index": 1.5,
            },
        ]
    )
    ajustados = {
        ("hourly", "kmeans", 3): "modelo_a",
        ("hourly", "kmeans", 4): "modelo_b",
        ("bucketed", "gmm", 5): "modelo_c",
    }
    resultado, metricas = seleccionar_mejor(resultados, ajustados)
    assert resultado == "modelo_b"
    assert metricas["test_silhouette_score"] == pytest.approx(0.55)
