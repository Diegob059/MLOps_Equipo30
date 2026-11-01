"""
Pruebas de reproducibilidad estricta para modelos y pipelines del proyecto MiBici.
===================================================================================
Verifica que las ejecuciones de modelado con semillas fijas produzcan
predicciones, probabilidades y métricas de evaluación idénticas:
1. Modelo base sintético (LightGBM).
2. Pipeline de riesgo operativo (Pregunta 2: LightGBM multiclase con datos reales).
3. Pipeline de segmentación de estaciones (Pregunta 3: GMM / clustering con datos reales).
"""

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED
from src.features.features import METADATA_COLS
from src.p2_config import LGBM_PARAMS, P2_FEATURES_TARGET_PARQUET


def test_lgbm_reproducibility():
    """Verifica que dos modelos LightGBM entrenados con la misma semilla

    produzcan exactamente las mismas predicciones en datos de clasificación.
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


def test_p2_pipeline_reproducibility_on_real_data():
    """Verifica la reproducibilidad del pipeline de la Pregunta 2 (Riesgo en estaciones).

    Dos entrenamientos independientes con los hiperparámetros oficiales y semilla fija
    sobre una muestra del dataset real deben generar predicciones, probabilidades
    y métricas idénticas (F1 macro y Balanced Accuracy).
    """
    if P2_FEATURES_TARGET_PARQUET.exists():
        df = pd.read_parquet(P2_FEATURES_TARGET_PARQUET).head(1500)
        drop_cols = [
            "timestamp",
            "station_id",
            "target",
            "risk_ratio",
            "net_flow_target",
        ]
        feature_cols = [
            c
            for c in df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feature_cols]
        y = df["target"].astype(int)
    else:
        # Fallback en caso de ejecución en entorno de integración continua sin datos crudos
        X, y = make_classification(
            n_samples=1000,
            n_features=12,
            n_informative=8,
            n_classes=3,
            random_state=RANDOM_SEED,
        )

    # Hiperparámetros representativos de P2 con semilla fija
    params = {
        **LGBM_PARAMS,
        "n_estimators": 30,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    }

    clf1 = LGBMClassifier(**params)
    clf2 = LGBMClassifier(**params)

    clf1.fit(X, y)
    clf2.fit(X, y)

    # Predicciones de clase y probabilidades
    preds1 = clf1.predict(X)
    preds2 = clf2.predict(X)
    probs1 = clf1.predict_proba(X)
    probs2 = clf2.predict_proba(X)

    assert np.array_equal(preds1, preds2), (
        "Las predicciones de clase difieren entre ejecuciones"
    )
    assert np.allclose(probs1, probs2, atol=1e-8), (
        "Las probabilidades difieren entre ejecuciones"
    )

    # Consistencia exacta de métricas clave de evaluación
    f1_m1 = f1_score(y, preds1, average="macro")
    f1_m2 = f1_score(y, preds2, average="macro")
    bal_acc1 = balanced_accuracy_score(y, preds1)
    bal_acc2 = balanced_accuracy_score(y, preds2)

    assert f1_m1 == f1_m2, f"Discrepancia en F1 macro: {f1_m1} != {f1_m2}"
    assert bal_acc1 == bal_acc2, (
        f"Discrepancia en Balanced Accuracy: {bal_acc1} != {bal_acc2}"
    )


def test_p3_clustering_pipeline_reproducibility_on_real_data():
    """Verifica la reproducibilidad del pipeline de la Pregunta 3 (Clustering de estaciones).

    Dos ajustes del pipeline Champion (StandardScaler + GaussianMixture) sobre el perfil
    horario real deben generar la misma asignación de clusters, silhouette score
    e índice Davies-Bouldin.
    """
    hourly_parquet = PROCESSED_DATA_DIR / "estaciones_caracteristicas_hourly.parquet"

    if hourly_parquet.exists():
        df = pd.read_parquet(hourly_parquet)
        feature_cols = [
            c
            for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feature_cols]
    else:
        # Fallback sintético representativo de perfiles horarios (352 estaciones x 52 horas)
        rng = np.random.default_rng(RANDOM_SEED)
        X = pd.DataFrame(rng.normal(size=(352, 52)))

    # Pipeline idéntico al Champion de P3
    pipeline1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gmm", GaussianMixture(n_components=2, random_state=RANDOM_SEED)),
        ]
    )
    pipeline2 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gmm", GaussianMixture(n_components=2, random_state=RANDOM_SEED)),
        ]
    )

    labels1 = pipeline1.fit_predict(X)
    labels2 = pipeline2.fit_predict(X)

    assert np.array_equal(labels1, labels2), (
        "La asignación de clusters difiere con la misma semilla"
    )

    X_scaled1 = pipeline1.named_steps["scaler"].transform(X)
    X_scaled2 = pipeline2.named_steps["scaler"].transform(X)

    sil1 = float(silhouette_score(X_scaled1, labels1))
    sil2 = float(silhouette_score(X_scaled2, labels2))
    db1 = float(davies_bouldin_score(X_scaled1, labels1))
    db2 = float(davies_bouldin_score(X_scaled2, labels2))

    assert sil1 == sil2, f"Discrepancia en Silhouette Score: {sil1} != {sil2}"
    assert db1 == db2, f"Discrepancia en Davies-Bouldin: {db1} != {db2}"
