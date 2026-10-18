# TC5061 — MLOps Equipo 30: Plataforma MiBici Guadalajara

Repositorio del **Equipo 30** para la materia de Operaciones de Inteligencia Artificial (MLOps) en la Maestría de Inteligencia Artificial Aplicada (Tecnológico de Monterrey).

Este repositorio contiene la plataforma integral de MLOps para la optimización operativa de **MiBici** (Área Metropolitana de Guadalajara), resolviendo la **Pregunta 2: Predicción de riesgo de desabasto y saturación en estaciones a 2 horas hacia adelante**.

## 🚀 Acceso Rápido a la Documentación y Código

* **Proyecto Principal:** [`MLOps_Equipo30/`](MLOps_Equipo30/)
* **Documentación Técnica & Quickstart:** Consulta el [`MLOps_Equipo30/README.md`](MLOps_Equipo30/README.md) para instrucciones completas de instalación, ejecución del pipeline reproducible, comandos de MLflow y flujo de Git.
* **ML Canvas de la Pregunta 2:** [`MLOps_Equipo30/references/ml_canvas_pregunta2.md`](MLOps_Equipo30/references/ml_canvas_pregunta2.md)
* **Diagrama de Arquitectura End-to-End:** [`MLOps_Equipo30/references/architecture_diagram.md`](MLOps_Equipo30/references/architecture_diagram.md)
* **Notebook de Exploración y Blueprint:** [`MLOps_Equipo30/notebooks/1.0-rb-p2-riesgo-estaciones.ipynb`](MLOps_Equipo30/notebooks/1.0-rb-p2-riesgo-estaciones.ipynb)
* **Propuesta Entregable DOCX:** [`docs/TC5061_Propuesta_Proyecto_Equipo30.docx`](docs/TC5061_Propuesta_Proyecto_Equipo30.docx)
* **ML Canvas Oficial (v1.2):** [`MLOps_Equipo30/docs/Machine Learning Canvas (v1.2).docx`](MLOps_Equipo30/docs/Machine%20Learning%20Canvas%20(v1.2).docx) y [`MLOps_Equipo30/references/p2_machine_learning_canvas.docx`](MLOps_Equipo30/references/p2_machine_learning_canvas.docx)

---

## 🛠️ Comandos Principales de Ejecución (Pregunta 2)

Desde la carpeta `MLOps_Equipo30/`:

```bash
# 1. Ingesta y versionado Parquet + SHA-256 (Datos 2025-2026)
python -m MiBici.p2_dataset

# 2. Generación del Feature Store sin Data Leakage (7.19M filas)
python -m MiBici.p2_features

# 3. Entrenamiento (LightGBM vs Baseline) y Tracking en MLflow
python -m MiBici.modeling.p2_train

# 4. Inferencia en batch y diagnóstico de 5 casos
python -m MiBici.modeling.p2_predict --sample-size 5

# 5. Visualizar Experimentos y Métricas en MLflow Localmente
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Abrir en el navegador: http://127.0.0.1:5000

# 6. Pruebas Unitarias Automatizadas
PYTHONPATH=. pytest
```
