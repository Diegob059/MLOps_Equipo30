# TC5061 — MLOps Equipo 30: Plataforma MiBici Guadalajara

Repositorio del **Equipo 30** para la materia de Operaciones de Inteligencia Artificial (MLOps) en la Maestría de Inteligencia Artificial Aplicada (Tecnológico de Monterrey).

Este repositorio contiene la plataforma integral de MLOps para la optimización operativa de **MiBici** (Área Metropolitana de Guadalajara), resolviendo la **Pregunta 2: Predicción de riesgo de desabasto y saturación en estaciones a 2 horas hacia adelante**.

## 🚀 Acceso Rápido a la Documentación y Código

* **Proyecto Principal:** [`MLOps_Equipo30/`](MLOps_Equipo30/)
* **Documentación Técnica & Quickstart:** Consulta el [`MLOps_Equipo30/README.md`](MLOps_Equipo30/README.md) para instrucciones completas de instalación, ejecución del pipeline reproducible, comandos de MLflow y flujo de Git.
* **ML Canvas de la Pregunta 2:** [`MLOps_Equipo30/references/ml_canvas_pregunta2.md`](MLOps_Equipo30/references/ml_canvas_pregunta2.md)
* **Diagrama de Arquitectura End-to-End:** [`MLOps_Equipo30/references/architecture_diagram.md`](MLOps_Equipo30/references/architecture_diagram.md)
* **Notebook de Exploración y Blueprint:** [`MLOps_Equipo30/notebooks/1.0-pregunta-2-riesgo-estaciones.ipynb`](MLOps_Equipo30/notebooks/1.0-pregunta-2-riesgo-estaciones.ipynb)

---

## 🛠️ Comandos Principales de Ejecución

Desde la carpeta `MLOps_Equipo30/`:

```bash
# 1. Ingesta y versionado Parquet + SHA-256
python -m MiBici.dataset

# 2. Generación del Feature Store sin Data Leakage
python -m MiBici.features

# 3. Entrenamiento (LightGBM) y Registro en MLflow
python -m MiBici.modeling.train

# 4. Inferencia en batch y diagnóstico de 5 casos
python -m MiBici.modeling.predict --sample-size 5
```
