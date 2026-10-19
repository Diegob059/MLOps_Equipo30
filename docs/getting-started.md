# Guía de Inicio Rápido (Getting Started) - MiBici Equipo 30

Esta guía describe los pasos necesarios para configurar el entorno de trabajo, reproducir la ingesta y preparación de datos, ejecutar los experimentos base en **MLflow Tracking** para las tres preguntas del proyecto (**P1, P2 y P3**), e interactuar con la interfaz gráfica de usuario.

---

## 1. Configuración del Entorno de Desarrollo

Se recomienda utilizar un entorno virtual con Python 3.10 o superior (ejecutado y validado en Python 3.13):

```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias del proyecto y paquete local
pip install -r requirements.txt
pip install -e .
```

---

## 2. Ingesta y Procesamiento de Datos (2025 - 2026)

Para descargar los datos abiertos de MiBici Guadalajara (años 2025 y 2026) y construir la base procesada:

```bash
# 1. Ingesta y validación con fallback
python -m src.data.p2_download_raw_data

# 2. Agregación temporal horaria y cruce con nomenclatura
python -m src.data.p2_dataset

# 3. Construcción de características y target sin fuga temporal
python -m src.features.p2_features
```

---

## 3. Ejecución de Experimentos en MLflow Tracking

El proyecto incluye un orquestador unificado que ejecuta los baselines y modelos de las tres preguntas y los registra bajo el experimento oficial **`mibici-equipo30`**:

```bash
# Ejecutar y registrar todos los experimentos (P1, P2 Baseline, P2 Candidato, P3)
python -m src.models.run_all_experiments
```

También es posible ejecutar únicamente la modelación de la Pregunta 2:

```bash
python -m src.models.p2_train
```

### Runs Registrados en `mibici-equipo30`:
1. **P1 — Inventario inicial:** `p1_baseline_semanal_balance`, `p1_baseline_historico_balance`, `p1_arboles_balance` y `p1_inventario_inicial`. Comparan salidas y llegadas sobre la misma rejilla completa. Métricas: `test_salidas_mae`, `test_llegadas_mae`, `test_inventario_mae` y cobertura de conteos horarios. La [guía de P1](p1-inventario-inicial.md) describe su entorno y evaluación independiente.
2. **`p2_baseline_dummy_stratified`** (P2 Baseline): Heurística estratificada para riesgo de desabasto y saturación. Métricas: `test_macro_f1: 0.3330`, `test_balanced_accuracy: 0.3331`.
3. **`p2_lgbm_risk_classifier_2025_2026`** (P2 Candidato): Clasificador LightGBM con `class_weight='balanced'`. Métricas: `test_macro_f1: 0.3667`, `test_balanced_accuracy: 87.51%`, `recall_desabasto: 87.80%`, `recall_saturacion: 90.73%`. Artefactos: Matriz de confusión, importancia de variables, reporte y modelo `.joblib`.
4. **`p3_baseline_kmeans_clustering_estaciones`** (P3): Clustering no supervisado de estaciones (K-Means k=4). Métricas: `test_silhouette_score: 0.2735`, `test_davies_bouldin_index: 1.3819`. Artefactos: Mapa de clusters y perfiles horarios.

---

## 4. Visualización en la Interfaz Gráfica de MLflow

Para inspeccionar el dashboard interactivo de MLflow con todos los parámetros, métricas prefijadas, etiquetas y artefactos:

```bash
# Iniciar servidor local de MLflow apuntando a la base SQLite
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5005
```

> **Nota importante sobre el acceso web:**
> - En **macOS**, el puerto por defecto `5000` está ocupado por el receptor de Apple AirPlay (`ControlCenter`). Por tal motivo se utiliza el puerto **`5005`**.
> - En **MLflow 3.16+**, la interfaz web abre por defecto en la vista de *GenAI / Traces* (la cual se muestra vacía al no haber llamadas LLM).
> - Para ver inmediatamente la tabla completa de corridas de Machine Learning, abra la siguiente URL directa o seleccione la pestaña superior **"Model training"**:
>
> 👉 **[http://127.0.0.1:5005/#/experiments/2/runs?workflowType=machine_learning](http://127.0.0.1:5005/#/experiments/2/runs?workflowType=machine_learning)**

---

## 5. Ejecución de Pruebas Unitarias

Para verificar la integridad de las funciones de datos y características:

```bash
pytest
```
