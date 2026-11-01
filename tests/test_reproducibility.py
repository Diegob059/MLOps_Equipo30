"""
Prueba de reproducibilidad del entrenamiento usando una semilla fija.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification

from src.p2_config import RANDOM_SEED


def test_lgbm_reproducibility():
    """
    Verifica que dos modelos entrenados con la misma semilla
    produzcan exactamente las mismas predicciones.
    """

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_SEED,
    )

    model1 = LGBMClassifier(
        random_state=RANDOM_SEED,
        n_estimators=20,
        verbose=-1,
    )

    model2 = LGBMClassifier(
        random_state=RANDOM_SEED,
        n_estimators=20,
        verbose=-1,
    )

    model1.fit(X, y)
    model2.fit(X, y)

    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    assert np.array_equal(pred1, pred2)