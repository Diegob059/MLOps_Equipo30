# Documentación Oficial — MiBici MLOps (Equipo 30)

Plataforma de Machine Learning Operations (MLOps) para análisis de demanda, predicción de desabasto/saturación y agrupamiento de estaciones del sistema **MiBici** en Guadalajara, alineado 100% con la estructura canónica **Cookiecutter Data Science v1 (CCDS v1)**.

---

## 🎯 Preguntas del Proyecto

1. **Pregunta 1:** Demanda agregada horaria y diaria de viajes por estación y zona geográfica.
2. **Pregunta 2 (Principal):** Predicción anticipada (2 horas) de riesgo operativo de desabasto o saturación por estación mediante LightGBM (`class_weight='balanced'`).
3. **Pregunta 3:** Perfilamiento y agrupamiento no supervisado de estaciones (*K-Means Clustering*, $k=4$) para optimización logística de rebalanceo.

---

## 📋 Documentos Clave

- [Guía de Inicio Rápido (Getting Started)](getting-started.md)
- [Diagrama de arquitectura general de P1, P2 y P3](../references/architecture_diagram.md)
- [Machine Learning Canvas completo de P1, P2 y P3](../references/ml_canvas_equipo30.docx)
- [Manifiesto de Versionado de Datos SHA-256](../references/p2_data_manifest.json)

---

## 🛠️ Comandos Principales (`Makefile`)

| Comando | Descripción |
|---|---|
| `make requirements` | Instala dependencias y el paquete `src` en modo editable (`pip install -e .`). |
| `make clean` | Elimina archivos compilados, cachés de pytest/ruff y artefactos de build. |
| `make lint` | Ejecuta validación estática de código con `ruff check` y `ruff format --check`. |
| `make format` | Aplica formateo automático de código con `ruff format`. |
| `make test` | Ejecuta la suite de pruebas unitarias automatizadas con `pytest`. |
| `make download_data` | Descarga de forma desatendida los datos abiertos oficiales (2025-2026). |
| `make data` | Ejecuta la normalización Parquet y generación del Feature Store. |
| `make train` | Entrena y registra todos los experimentos en **MLflow Tracking** (`mibici-equipo30`). |
| `make train_p2` | Entrena específicamente el clasificador de riesgo de la Pregunta 2. |
| `make predict_p2` | Ejecuta inferencia por lotes y muestra 5 ejemplos cualitativos en consola. |
| `make mlflow` | Inicia el servidor de interfaz web de MLflow en el puerto local `5005`. |
