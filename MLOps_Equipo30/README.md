# TC5061 — MLOps Equipo 30: Plataforma de Inteligencia Operativa MiBici

Sistema integral de Machine Learning Operations (MLOps) diseñado para la anticipación de riesgos operativos, balanceo logístico y análisis de demanda del sistema de bicicletas públicas compartidas **MiBici** en el Área Metropolitana de Guadalajara (AMG).

---

## 🎯 1. Propósito del Proyecto y Preguntas de Negocio

El objetivo general es transformar los datos históricos abiertos de MiBici en capacidades analíticas y predictivas reproducibles que optimicen la disponibilidad del servicio y reduzcan los tiempos de espera y fricción de los usuarios.

El proyecto aborda tres preguntas estratégicas:

1. **Pregunta 1 (Demanda Espacio-Temporal):** ¿Cómo se distribuye y evoluciona la demanda horaria y diaria de viajes entre las distintas zonas del AMG?
2. **Pregunta 2 (Riesgo Operativo en Estaciones — Foco Principal de este Módulo):**  
   *¿Una estación estará en riesgo de quedarse sin bicicletas (Desabasto) o sin anclajes libres (Saturación) en un horizonte futuro de 2 horas?*  
   Permite a los despachadores del CCO planificar rutas proactivas de rebalanceo antes de que el usuario experimente la falta de servicio.
3. **Pregunta 3 (Optimización Logística):** ¿Cuáles son las rutas y secuencias de rebalanceo más eficientes para las furgonetas operativas de MiBici?

---

## 📊 2. Datos Utilizados

* **Fuente Oficial:** Portal de Datos Abiertos de MiBici ([https://www.mibici.net/es/datos-abiertos/](https://www.mibici.net/es/datos-abiertos/)).
* **Periodo Histórico:** Años completos 2023 y 2024 (24 archivos mensuales de viajes).
* **Volumen:** Más de **8.93 millones de viajes** registrados y agregados en series de tiempo horarias para 370 estaciones activas.
* **Catálogo de Nomenclatura:** Catálogo oficial con ID de estación, nombre, geolocalización (latitud, longitud) y capacidad física de anclajes instalados (`capacity`).
* **Formato Intermedio:** **Apache Parquet** comprimido para alta velocidad de I/O y bajo consumo de disco.

---

## 🗂️ 3. Estructura del Repositorio (Cookiecutter Data Science)

```
MLOps_Equipo30/
├── data/
│   ├── raw/                 <- Datos crudos inmutables (CSVs de Viajes y Nomenclatura).
│   ├── interim/             <- Datos limpios en Parquet (estaciones.parquet, viajes_agregados_hora.parquet).
│   └── processed/           <- Datasets finales para modelado (mibici_features_target.parquet).
├── notebooks/
│   └── 1.0-pregunta-2-riesgo-estaciones.ipynb <- Exploración interactiva y blueprint analítico.
├── MiBici/                  <- Código fuente modular del paquete Python del proyecto.
│   ├── __init__.py
│   ├── config.py            <- Centralización de rutas, fechas, hiperparámetros y MLflow.
│   ├── dataset.py           <- Ingesta desatendida, normalización Parquet y versionado con hashes.
│   ├── features.py          <- Feature store: rejilla espacio-temporal, zero-filling, lags y rollings.
│   ├── plots.py             <- Generación de diagnósticos visuales (matriz de confusión e importancias).
│   └── modeling/
│       ├── __init__.py
│       ├── train.py         <- Entrenamiento (Baseline vs LightGBM), evaluación y registro en MLflow.
│       └── predict.py       <- Inferencia en batch y diagnóstico cualitativo de 5 ejemplos de test.
├── models/                  <- Modelos serializados entrenados (lgbm_risk_classifier.joblib).
├── reports/
│   └── figures/             <- Figuras diagnósticas generadas (confusion_matrix.png, feature_importance.png).
├── references/              <- Documentación de arquitectura, ML Canvas y manifiesto de datos.
│   ├── data_manifest.json   <- Manifiesto criptográfico (SHA-256) de versionado de datos en Git.
│   ├── ml_canvas_pregunta2.md <- Machine Learning Canvas detallado para la Pregunta 2.
│   └── architecture_diagram.md<- Diagramas de arquitectura del flujo end-to-end.
├── pyproject.toml           <- Configuración de empaquetado del módulo MiBici.
├── requirements.txt         <- Dependencias con versiones fijadas para reproducibilidad.
├── Makefile                 <- Comandos estándar de conveniencia.
└── README.md                <- Documentación principal del proyecto.
```

---

## ⚙️ 4. Instalación y Configuración

Clonar el repositorio y configurar el entorno virtual:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Diegob059/TC5061-Proyecto-Equipo30.git
cd TC5061-Proyecto-Equipo30/MLOps_Equipo30

# 2. Instalar las dependencias fijadas del proyecto
pip install -r requirements.txt
```

---

## 🚀 5. Ejecución del Pipeline End-to-End (Sin Pasos Manuales)

Cada fase del pipeline es ejecutable desde la línea de comandos mediante interfaces CLI construidas con `typer`:

### Paso 1: Ingesta, Normalización y Versionado de Datos
Descarga automáticamente los 24 meses de viajes y nomenclatura, normaliza tipos y genera el manifiesto SHA-256 en Parquet:
```bash
python -m MiBici.dataset
```
*Salidas:* `data/interim/estaciones.parquet`, `data/interim/viajes_agregados_hora.parquet`, `references/data_manifest.json`.

### Paso 2: Generación del Feature Store y Target
Construye la rejilla cartesiana con Zero-Filling, calcula el target sin leakage a 2 horas y genera lags y rolling balances:
```bash
python -m MiBici.features
```
*Salidas:* `data/processed/mibici_features_target.parquet`.

### Paso 3: Entrenamiento del Modelo y Registro en MLflow
Ajusta el Baseline Estratificado y el modelo candidato LightGBM (`class_weight='balanced'`), evalúa sobre Test (julio - diciembre 2024), guarda el modelo y registra parámetros y artefactos en MLflow:
```bash
python -m MiBici.modeling.train
```
*Salidas:* `models/lgbm_risk_classifier.joblib`, `reports/figures/confusion_matrix.png`, `reports/figures/feature_importance.png`, base de datos MLflow `mlflow.db`.

### Paso 4: Inferencia en Batch e Inspección de Casos de Test
Carga el modelo serializado, ejecuta inferencia sobre las 1.63 millones de observaciones de prueba y presenta una tabla de 5 ejemplos cualitativos (valor predicho vs esperado con probabilidades):
```bash
python -m MiBici.modeling.predict --sample-size 5
```
*Salidas:* `data/processed/test_predictions.parquet` y resumen tabular en consola.

### Paso 5: Explorador de Experimentos en MLflow UI
Para inspeccionar gráficamente las métricas, parámetros y artefactos registrados:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Acceder en el navegador a `http://127.0.0.1:5000`.

---

## 🌿 6. Flujo de Trabajo con Git y GitHub (GitHub Flow)

Para asegurar la calidad, reproducibilidad y trazabilidad del código, el equipo opera bajo las siguientes políticas:

### 6.1 Rama Principal Protegida (`main`)
* La rama `main` representa código en estado de producción y está protegida contra *direct pushes*.
* Cualquier cambio debe integrarse obligatoriamente a través de un **Pull Request (PR)** verificado.

### 6.2 Convención de Nombres de Ramas
Las ramas de trabajo se crean a partir de `main` siguiendo la nomenclatura semántica:
* `feature/<descripción-corta>`: Nuevas funcionalidades o pipelines (ej. `feature/ingesta-parquet`, `feature/eda-prediccion-desabasto`).
* `fix/<descripción-corta>`: Correcciones de bugs o errores en transformaciones.
* `refactor/<descripción-corta>`: Mejoras estructurales sin alterar la lógica de negocio.
* `docs/<descripción-corta>`: Cambios en documentación o artefactos en `references/`.

### 6.3 Convención de Mensajes de Commit (Conventional Commits)
Se adopta el estándar de [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/):
```
<tipo>(<alcance>): <descripción concisa en imperativo>

[cuerpo explicativo opcional detallando el motivo y decisiones de diseño]
```
*Tipos aceptados:* `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`.

### 6.4 Política de Revisión entre Pares (Peer Code Review)
* **Apertura de PRs:** Cada integrante del equipo es responsable de abrir al menos un Pull Request describiendo los cambios, comandos de prueba y evidencia de ejecución.
* **Revisión Obligatoria:** Cada integrante debe realizar al menos una revisión formal (*Code Review*) con comentarios técnicos y aprobación explícita (*Approved*) a otro compañero antes de realizar el merge hacia `main`.
* **Fusión:** Se utiliza *Squash and Merge* o *Merge Commit* manteniendo un historial limpio.

---

## 👥 7. Integrantes del Equipo y Roles (Equipo 30)

| Integrante | Rol en el Proyecto | Responsabilidades Principales |
|---|---|---|
| **Ricardo Bahamón** | **MLOps & Pipeline Engineer** | Arquitectura del pipeline reproducible, orquestación de ingesta Parquet, configuración de MLflow, tracking de artefactos y control de versiones de datos. |
| **Diego Botero** | **Lead Data Scientist & Modeling** | Formulación matemática del target de riesgo, ingeniería de variables de series de tiempo (lags/rollings), entrenamiento y calibración de modelos LightGBM. |
| **Compañero Equipo 30** | **Data Engineer & Quality QA** | Scraping dinámico resiliente, saneamiento de catálogos geoespaciales, validación de integridad criptográfica (SHA-256) y pruebas unitarias. |
| **Compañero Equipo 30** | **Product & Deployment Lead** | Elaboración del ML Canvas, especificación de requisitos operativos del CCO, diseño de diagnósticos visuales y documentación técnica. |

---

## 📄 8. Licencia
Este proyecto está desarrollado bajo fines académicos para el curso TC5061 (Operaciones de Inteligencia Artificial) en el Tecnológico de Monterrey. Licencia MIT.
