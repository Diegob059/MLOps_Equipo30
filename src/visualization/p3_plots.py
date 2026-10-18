"""
Visualizaciones para la Pregunta 3: Clustering de Estaciones (MiBici).
=======================================================================
Grafica los resultados del *pipeline* de clustering de
``src/models/p3_train.py``: comparación de modelos candidatos, perfil
promedio de cada cluster y mapa geográfico de clusters.

Ejecución desde terminal (grafica las asignaciones ya guardadas):
    python -m src.visualization.p3_plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="CLI de visualización de clusters de estaciones (Pregunta 3).")


def graficar_seleccion_k(resultados: pd.DataFrame) -> plt.Figure:
    """Coeficiente de silueta vs. k, una línea por combinación (conjunto_caracteristicas, algoritmo).

    `resultados` es la tabla del barrido producida por
    `src.models.p3_train.ejecutar_barrido`, con al menos las columnas
    conjunto_caracteristicas, algoritmo, k, silueta.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for (conjunto_caracteristicas, algoritmo), grupo in resultados.groupby(["conjunto_caracteristicas", "algoritmo"]):
        grupo = grupo.sort_values("k")
        ax.plot(grupo["k"], grupo["silueta"], marker="o", label=f"{conjunto_caracteristicas} / {algoritmo}")
    ax.set_xlabel("k (número de clusters)")
    ax.set_ylabel("Coeficiente de silueta")
    ax.set_title("P3 - Selección de k por conjunto de features y algoritmo")
    ax.legend()
    fig.tight_layout()
    return fig


def graficar_mapa_calor_perfil_clusters(df: pd.DataFrame, columnas_caracteristicas: list[str], etiquetas: np.ndarray) -> plt.Figure:
    """Promedio de cada feature (sin escalar) por cluster, como mapa de calor.

    Esto es lo que hace interpretables los clusters: cada fila es un
    cluster, cada columna una feature, y el color muestra cómo se compara
    ese cluster con los demás en esa feature.
    """
    perfil = df[columnas_caracteristicas].groupby(etiquetas).mean()
    perfil.index.name = "cluster"
    fig, ax = plt.subplots(figsize=(max(8, len(columnas_caracteristicas) * 0.35), max(3, perfil.shape[0] * 0.6)))
    sns.heatmap(perfil, cmap="vlag", center=0 if (perfil.min().min() < 0) else None, annot=False, ax=ax)
    ax.set_title("P3 - Perfil promedio de cada cluster (features sin escalar)")
    fig.tight_layout()
    return fig


def graficar_mapa_clusters(df: pd.DataFrame, etiquetas: np.ndarray) -> plt.Figure:
    """Dispersa las estaciones en el mapa, coloreadas por cluster, para interpretar dónde está cada patrón."""
    fig, ax = plt.subplots(figsize=(9, 7))
    paleta = sns.color_palette("tab10", n_colors=len(set(etiquetas)))
    for id_cluster in sorted(set(etiquetas)):
        mascara = etiquetas == id_cluster
        ax.scatter(
            df.loc[mascara, "longitude"],
            df.loc[mascara, "latitude"],
            s=30,
            alpha=0.85,
            color=paleta[id_cluster % len(paleta)],
            label=f"Cluster {id_cluster} (n={mascara.sum()})",
        )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect(1 / np.cos(np.deg2rad(20.67)))
    ax.set_title("P3 - Estaciones coloreadas por cluster de patrón de uso")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "p3_estaciones_clusters.parquet",
    output_path: Path = FIGURES_DIR / "p3_clusters_estaciones_mapa.png",
):
    """CLI independiente: grafica las asignaciones de cluster ya guardadas en el mapa."""
    df = pd.read_parquet(input_path)
    fig = graficar_mapa_clusters(df, df["cluster"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.success(f"Guardado: {output_path}")


if __name__ == "__main__":
    app()
