# Diagrama de Arquitectura del Sistema MLOps (MiBici)

Este documento detalla la arquitectura modular end-to-end implementada para responder a la **Pregunta 2 de MiBici**: Predicción anticipada (2 horas) de riesgo de desabasto y saturación en estaciones.

---

## 1. Diagrama de Arquitectura End-to-End

```mermaid
flowchart TD
    subgraph S1["1. Ingesta y Versionado de Datos (src/data/ y MiBici.dataset)"]
        A["Portal MiBici (Datos Abiertos)"] -->|"Scraping Resiliente / Fallback"| B["data/raw/Viajes/ (21 CSVs: 2025-2026)<br/>data/raw/Nomenclatura/ (1 CSV)"]
        B -->|"Normalización, Chunks (latin1 -> UTF-8)"| C["MiBici.dataset"]
        C -->|"Serialización Eficiente"| D["data/interim/estaciones.parquet<br/>data/interim/viajes_agregados_hora.parquet"]
        C -->|"Cálculo de Hashes SHA-256"| E["references/data_manifest.json<br/>(Versionado de Datos en Git)"]
    end

    subgraph S2["2. Feature Store & Target (MiBici.features)"]
        D --> F["Rejilla Cartesiana Espacio-Temporal<br/>(Estaciones Activas x 15,312 Horas)"]
        F -->|"Imputación Física"| G["Zero-Filling (Ausencia = 0 real)"]
        G -->|"shift(-1, -2) / Capacidad"| H["Cálculo del Target Operativo (2h)<br/>{0: Normal, 1: Desabasto, 2: Saturación}"]
        G -->|"shift(lags, rollings) [t <= 0]"| I["Feature Engineering Histórico<br/>- Lags (1, 2, 24, 168)<br/>- Rollings 3h, 12h<br/>- Cíclicas sin/cos"]
        H --> J["data/processed/mibici_features_target.parquet"]
        I --> J
    end

    subgraph S3["3. Entrenamiento, Validación y MLflow (MiBici.modeling.train)"]
        J -->|"División Cronológica Estricta<br/>(Train < 2026-01-01 <= Test)"| K["Partición Train / Test"]
        K --> L["Baseline Heurístico<br/>(DummyClassifier Estratificado)"]
        K --> M["Modelo Candidato<br/>(LGBMClassifier con class_weight='balanced')"]
        L --> N["Evaluación Cuantitativa<br/>(Macro F1, Balanced Acc, Recall por clase)"]
        M --> N
        N --> O["Serialización de Modelo<br/>models/lgbm_risk_classifier.joblib"]
        N --> P["MiBici.plots<br/>reports/figures/confusion_matrix.png<br/>reports/figures/feature_importance.png"]
        N --> Q["Servidor / Base de Datos MLflow<br/>(sqlite:///mlflow.db)<br/>- Parámetros e Hiperparámetros<br/>- Métricas & Data Version Manifest<br/>- Artefactos y Figuras"]
        E -.->|"Tag de Versión de Datos"| Q
    end

    subgraph S4["4. Inferencia Operativa y Diagnóstico (MiBici.modeling.predict)"]
        O --> R["MiBici.modeling.predict"]
        J --> R
        D --> R
        R --> S["data/processed/test_predictions.parquet"]
        R --> T["Inspección Cualitativa en Terminal<br/>(5 Ejemplos: Esperado vs Predicho + Probabilidades)"]
    end
```

---

## 2. Descripción de Componentes y Módulos

### Módulo 1: Ingesta y Versionado (`MiBici.dataset`)
* **Propósito:** Automatizar la descarga desatendida, normalizar nombres de variables, codificación y tipos temporales, consolidando los datos en formato Parquet (`data/interim/`).
* **Mecanismo de Versionado:** Genera un manifiesto criptográfico en `references/data_manifest.json` que registra el hash SHA-256 de cada archivo individual y calcula un hash global para la versión del dataset (ej. `mibici-v2025_2026-d6201b6995`), consumible directamente por MLflow.

### Módulo 2: Ingeniería de Características (`MiBici.features`)
* **Propósito:** Generar la matriz cartesiana continua y construir variables explicativas utilizando **exclusivamente datos del pasado ($t \le 0$)**.
* **Zero-Filling:** Convierte horas inactivas en ceros físicos reales para mantener la continuidad matemática de las series temporales.
* **Target Sin Leakage:** Formula el balance futuro de las 2 horas subsecuentes ($\Delta_{\text{net}} = \text{Llegadas}_{[t+1, t+2]} - \text{Salidas}_{[t+1, t+2]}$), discretizado en 3 clases según el ratio sobre la capacidad de la estación ($\le -0.50$, $\ge 0.50$ o intermedio).
* **Almacenamiento:** Serializa el dataset limpio en `data/processed/mibici_features_target.parquet`.

### Módulo 3: Entrenamiento y Trazabilidad (`MiBici.modeling.train`)
* **Propósito:** Ajustar el modelo LightGBM con penalización asimétrica de clases para mitigar el desbalance severo (clases críticas $< 1\%$).
* **MLflow Tracking:** Registra en la base de datos SQLite (`mlflow.db`) todos los parámetros de entrenamiento, versión del dataset, métricas de detección (Recall de desabasto y saturación) y almacena como artefactos las figuras diagnósticas generadas en `reports/figures/`.
* **Persistencia:** Guarda el clasificador entrenado en `models/lgbm_risk_classifier.joblib`.

### Módulo 4: Inferencia y Diagnóstico (`MiBici.modeling.predict`)
* **Propósito:** Proveer un punto de entrada desacoplado para inferencia en lotes (Batch Inference).
* **Diagnóstico Operativo:** Muestra en consola ejemplos con el valor esperado real contra el predicho por el modelo, incluyendo la distribución de probabilidades y diagnóstico cualitativo (`✓ Acierto`, `⚠️ Alerta Preventiva`).

### Módulo de Configuración Central (`MiBici.config`)
* Centraliza rutas absolutas, umbrales de decisión, semillas aleatorias (`RANDOM_SEED = 42`), hiperparámetros y URIs de tracking.
