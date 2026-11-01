# TC5061 — MLOps Equipo 30: Plataforma de Inteligencia Operativa MiBici

Sistema integral de Machine Learning Operations (MLOps) diseñado para la anticipación de riesgos operativos, balanceo logístico y análisis de demanda del sistema de bicicletas públicas compartidas **MiBici** en el Área Metropolitana de Guadalajara (AMG).

---

## 🎯 1. Propósito del Proyecto y Preguntas de Negocio

El objetivo general es transformar los datos históricos abiertos de MiBici en capacidades analíticas y predictivas reproducibles que optimicen la disponibilidad del servicio y reduzcan los tiempos de espera y fricción de los usuarios.

El proyecto aborda tres preguntas estratégicas:

1. **Pregunta 1 (Demanda e Inventario Inicial):** ¿Cuántos viajes se iniciarán por estación hasta terminar el primer pico de la mañana y con cuántas bicicletas debería iniciar cada estación, considerando las llegadas previstas? Regresión de salidas y llegadas horarias, seguida de un balance conservador de inventario.
2. **Pregunta 2 (Riesgo Operativo en Estaciones — Foco Principal de este Módulo):**  
   *¿Una estación estará en riesgo de quedarse sin bicicletas (Desabasto) o sin anclajes libres (Saturación) en un horizonte futuro de 2 horas?*  
   Permite a los despachadores del CCO planificar rutas proactivas de rebalanceo antes de que el usuario experimente la falta de servicio.
3. **Pregunta 3 (Segmentación de Estaciones):** ¿Qué estaciones comparten patrones de uso similares (perfiles horarios de salidas/llegadas) y cómo se distribuyen geográficamente estos grupos en el AMG? Clustering no supervisado (K-Means y Gaussian Mixture) sobre perfiles de actividad por estación, útil para diseñar estrategias de rebalanceo y mantenimiento diferenciadas por segmento.

---

## 📊 2. Datos Utilizados

* **Periodo Histórico:** Años 2025 y 2026 (21 archivos mensuales de viajes: 2025 completo y 2026 hasta septiembre).
* **Volumen:** Más de **7.89 millones de viajes** registrados y agregados en series de tiempo horarias para 470 estaciones activas.
* **Catálogo de Nomenclatura:** Catálogo oficial con ID de estación, nombre, geolocalización (latitud, longitud) y capacidad física de anclajes instalados (`capacity`).
* **Formato Intermedio:** **Apache Parquet** comprimido para alta velocidad de I/O y bajo consumo de disco.

---

## 🗂️ 3. Estructura del Repositorio (Cookiecutter Data Science)

```
TC5061-Proyecto-Equipo30/
├── MLproject                 <- Define los entry points `ingest`/`train` para `mlflow run` (ver sección 6).
├── python_env.yaml           <- Entorno declarado para el MLproject (Python 3.12, referencia a requirements.txt).
├── Makefile                  <- Comandos estándar de conveniencia y automatización (make mlflow, data_p1/p2/p3, train_p1/p2/p3...).
├── requirements.txt          <- Dependencias con versiones fijadas para reproducibilidad.
├── setup.py                  <- Setup clásico de empaquetado pip editable (CCDS v1).
├── scripts/
│   ├── ingest.py             <- Entry point `ingest` del MLproject (despacha a p1/p2/p3/all).
│   ├── train.py              <- Entry point `train` del MLproject (pregunta + modelo + hiperparámetros).
│   └── ejecutar_notebook_p1.py <- Helper para re-ejecutar el notebook de P1 y dejar sus gráficas actualizadas.
├── data/
│   ├── external/             <- Datos externos complementarios (metadatos de clima o API).
│   ├── raw/                  <- Datos crudos inmutables (CSVs de Viajes y Nomenclatura 2025-2026).
│   ├── interim/               <- Datos limpios en Parquet (viajes_limpios.parquet, estaciones.parquet, flujos horarios P1).
│   └── processed/            <- Datasets finales para modelado: features y predicciones de P1, P2 y P3.
├── docs/                     <- Documentación del proyecto, Getting Started, ML Canvas y guías por pregunta.
│   ├── getting-started.md    <- Guía de configuración y reproducción paso a paso.
│   ├── p1-inventario-inicial.md <- Guía detallada de P1 (entorno, particiones temporales, corridas de MLflow).
│   └── Machine Learning Canvas (v1.2).docx <- Canvas oficial del equipo.
├── notebooks/                <- Convención `1.x-iniciales-descripcion.ipynb`.
│   ├── 1.0-rb-p2-riesgo-estaciones.ipynb       <- Exploración y blueprint analítico de P2.
│   ├── 1.0-db-clustering-estaciones.ipynb      <- EDA de clustering de estaciones de P3.
│   └── 1.3-ge-p1-balance-salidas-llegadas.ipynb <- Resultados y gráficas de P1.
├── src/                      <- Código fuente canónico del paquete Python (CCDS v1).
│   ├── config.py             <- Configuración global compartida (rutas, fechas 2025-2026, semilla, MLflow).
│   ├── p1_config.py          <- Configuración específica de P1.
│   ├── p2_config.py          <- Configuración específica de P2 (rutas, hiperparámetros, MLflow).
│   ├── data/
│   │   ├── dataset.py        <- Ingesta y limpieza compartida (usada por P3; catálogo de estaciones común a P1/P2).
│   │   ├── p1_dataset.py     <- Ingesta y preparación de datos de P1.
│   │   ├── p2_dataset.py     <- Ingesta desatendida, normalización Parquet y versionado SHA-256 de P2.
│   │   └── p2_download_raw_data.py <- Descargador automatizado de datos abiertos 2025-2026.
│   ├── features/
│   │   ├── features.py       <- Ingeniería de características compartida (perfiles de estación de P3).
│   │   ├── p1_features.py    <- Features horarias de demanda/inventario de P1.
│   │   └── p2_features.py    <- Feature store: rejilla continua (7.19M filas), zero-filling, lags y target de P2.
│   ├── models/
│   │   ├── p1_train.py / p1_predict.py / p1_balance.py   <- Entrenamiento, inferencia y balance de inventario de P1.
│   │   ├── p2_train.py / p2_predict.py                   <- Entrenamiento (Baseline vs LightGBM) e inferencia de P2.
│   │   ├── p3_train.py / p3_predict.py / p3_register.py  <- Búsqueda de clustering con *nested runs*, inferencia y registro en el Model Registry de P3.
│   │   └── run_all_experiments.py <- Orquestador compartido: ejecuta y registra P1, P2 y P3 en MLflow.
│   └── visualization/
│       └── p1_plots.py / p2_plots.py / p3_plots.py <- Diagnósticos visuales por pregunta.
├── models/                   <- Modelos serializados entrenados (`*.joblib`, `*.pkl`); no se versionan en Git.
├── reports/
│   ├── figures/               <- Figuras diagnósticas generadas por pregunta (p1_*, p2_*, p3_*).
│   └── p1/                   <- Métricas, reportes y recomendaciones de ejemplo de P1.
├── references/                <- Documentación de arquitectura, ML Canvas y manifiestos de datos.
│   ├── data_manifest.json / p1_data_manifest.json / p2_data_manifest.json <- Manifiestos criptográficos (SHA-256) de versionado de datos.
│   └── *_ml_canvas.md / architecture_diagram.md <- Machine Learning Canvas y diagramas de arquitectura.
├── tests/
│   ├── test_data.py / test_dataset.py / test_features.py <- Pruebas unitarias compartidas.
│   ├── test_p1_experiments.py                            <- Pruebas unitarias de P1.
│   ├── test_p2_data.py / test_p2_features.py / test_p2_train.py <- Pruebas unitarias de P2.
│   └── test_p3_train.py / test_p3_register.py            <- Pruebas unitarias de P3 (clustering, registro y auditoría).
└── README.md                <- Documentación principal del proyecto.
```

---

## ⚙️ 4. Instalación y Configuración

Se requiere Python 3.12. En macOS, instalar primero el runtime OpenMP que utiliza
el paquete de LightGBM: `brew install libomp` (con Homebrew disponible). Sin él,
la importación puede fallar con `Library not loaded: ...libomp.dylib`. Esta es una
dependencia del sistema y no se instala mediante `requirements.txt`.
Referencia: [instalación del paquete Python de LightGBM](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst).

Clonar el repositorio y configurar el entorno virtual:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Diegob059/TC5061-Proyecto-Equipo30.git
cd TC5061-Proyecto-Equipo30

# 2. Crear y activar el entorno
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 3. Instalar las dependencias fijadas del proyecto
python -m pip install -r requirements.txt

# 4. Verificar la dependencia nativa y ejecutar la suite completa
python -c "import lightgbm; print(lightgbm.__version__)"
python -m pytest -q
```

---

## 🚀 5. Ejecución del Pipeline End-to-End de la Pregunta 2 (Prefijo `P2`)

Cada fase del pipeline de la Pregunta 2 cuenta con el prefijo `p2_` para mantenerse 100% desacoplado de las implementaciones de los demás compañeros del equipo:

### Paso 0: Levantar el servidor local de MLflow Tracking (requisito para todo lo demás)
Todas las preguntas (P1, P2 y P3) registran sus corridas contra un servidor de *tracking* respaldado por base de datos (requisito para que el Model Registry funcione). **Debe estar corriendo antes de ejecutar cualquier comando de ingesta o entrenamiento:**
```bash
make mlflow
# O directamente:
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
En macOS, si AirPlay ocupa el puerto 5000, usa `--port 5001` y exporta `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` en las terminales de ejecución.

Esto expone la UI y la API en `http://127.0.0.1:5000` y crea/actualiza `mlflow.db` en la raíz del repo (archivo ignorado por Git, ver `.gitignore`). Déjalo corriendo en una terminal aparte mientras ejecutas los pasos siguientes. Por defecto, `MLFLOW_TRACKING_URI` apunta a `http://127.0.0.1:5000`; puedes sobreescribirlo con una variable de entorno si usas otro host/puerto.

### Paso 1: Ingesta, Normalización y Versionado de Datos (2025 y 2026)
Descarga automáticamente los 21 meses de viajes y nomenclatura más reciente, normaliza tipos y genera el manifiesto SHA-256 en Parquet:
```bash
python -m src.data.p2_dataset
```
*Salidas:* `data/interim/p2_estaciones.parquet`, `data/interim/p2_viajes_agregados_hora.parquet`, `references/p2_data_manifest.json`.

### Paso 2: Generación del Feature Store y Target
Construye la rejilla cartesiana continua con Zero-Filling (7,196,640 filas), calcula el target sin leakage a 2 horas y genera lags y rolling balances:
```bash
python -m src.features.p2_features
```
*Salidas:* `data/processed/p2_mibici_features_target.parquet`.

### Paso 3: Entrenamiento, Comparación de Candidatos y Registro en MLflow
Ajusta el Baseline Estratificado y evalúa al menos 3 candidatos diversos de LightGBM (`lgbm_balanceado`, `lgbm_profundo` y `lgbm_compacto`) con partición temporal estricta anti-leakage (< 2025-12-31 22:00:00). Registra parámetros, métricas, modelos firmados con `input_example`, datos y diagnósticos en MLflow:
```bash
python -m src.models.p2_train
```
O vía MLflow Project:
```bash
mlflow run . -e train -P question=p2 --experiment-name mibici-equipo30 --env-manager local
```
*Salidas:* `reports/p2/comparacion.csv`, `reports/p2/seleccion.json`, `models/p2_lgbm_risk_classifier.joblib`, `reports/figures/p2_confusion_matrix.png`, `reports/figures/p2_feature_importance.png`.

### Paso 4: Registro del Champion en Model Registry y Transición a Producción
Registra el modelo seleccionado en el Model Registry bajo `mibici-equipo30-p2-riesgo`, copia el linaje completo y le asigna el alias `champion`:
```bash
python -m src.models.p2_register --run-id <run_id_elegido> --reason "Candidato champion con mejor balance F1 y recall"
```
O vía MLflow Project:
```bash
mlflow run . -e register_p2 -P run_id=<run_id_elegido> -P reason="Candidato champion con mejor balance F1 y recall" --experiment-name mibici-equipo30 --env-manager local
```
*Salidas:* `reports/p2/registry.json`.

### Paso 5: Inferencia en Batch desde el Model Registry (Producción)
Ejecuta inferencia sobre las 3,078,500 observaciones de prueba de 2026 cargando directamente el modelo Champion desde el Model Registry:
```bash
python -m src.models.p2_predict --model-uri models:/mibici-equipo30-p2-riesgo@champion --sample-size 5
```
*Salidas:* `data/processed/p2_test_predictions.parquet` y resumen tabular en consola.

### Paso 6: Ejecución de Pruebas Unitarias
```bash
pytest tests
```

Para más detalles sobre la metodología, partición anti-leakage y comparativas de P2, consultar la [Guía de MLflow Avanzado de P2](docs/p2-mlflow-avanzado.md).

---

## P1: bicicletas al inicio del día

La implementación de P1 combina regresión horaria de salidas y llegadas con un balance de inventario hasta el final del primer pico matutino. Compara baseline semanal, promedio histórico y árboles sobre las mismas estaciones, fechas y horas, incluidas las horas sin actividad.

Para esta entrega, con el entorno activado y el servidor de tracking en ejecución:

```bash
mlflow run . -e ingest -P question=p1 --experiment-name mibici-equipo30 --env-manager local
mlflow run . -e train -P question=p1 --experiment-name mibici-equipo30 --env-manager local
```

La ingesta descarga automáticamente los archivos faltantes del periodo fijado y verifica sus hashes contra el manifiesto versionado. El entrenamiento compara el baseline semanal, el promedio histórico y cuatro árboles (profundidades 6/10 × hojas mínimas 50/100). La búsqueda usa un padre y cuatro corridas hijas; cada modelo conserva firma, ejemplo, manifiesto, configuración, metadatos del dataset y diagnósticos. Se mantienen exactamente la cohorte, las fechas y las métricas de P1 de la propuesta. La selección utiliza **MAE de inventario en validación**; prueba se reserva para reportar desempeño.

La [guía de tracking avanzado de P1](docs/p1-mlflow-avanzado.md) contiene los comandos para registrar y cargar `models:/mibici-equipo30-p1-demanda@champion`, la inferencia diaria, las tablas y los pendientes de integración del equipo. La [guía original](docs/p1-inventario-inicial.md) documenta los supuestos del balance y el flujo anterior. El experimento compartido es `mibici-equipo30`.

---

## P3: segmentación de estaciones por patrón de uso

La implementación de P3 agrupa las 352 estaciones comunes a los 21 meses según su perfil horario de salidas/llegadas (clustering no supervisado), para identificar qué estaciones se comportan de forma similar y dónde se ubican esos grupos en el AMG.

```bash
# 1. Ingesta + features compartidos (también usados por P1/P2 para el catálogo de estaciones)
python -m src.data.dataset
python -m src.features.features

# 2. Entrenamiento: búsqueda de hiperparámetros + baseline + candidatos, todo registrado en MLflow
python -m src.models.p3_train

# 3. Inferencia: asigna cluster a cada estación usando el mejor candidato guardado localmente
python -m src.models.p3_predict
```
O con los atajos del `Makefile`:
```bash
make data_p3
make train_p3
```

El paso 2 compara K-Means y Gaussian Mixture en un barrido de k=2 a 10 sobre dos conjuntos de features (perfil horario completo y perfil agrupado por franjas; 36 combinaciones en total), organizado como una **búsqueda con *nested runs*** de MLflow: un run padre (`p3_busqueda_hiperparametros_clustering`) con las 36 combinaciones como runs hijos anidados. De ahí se promueven **al menos 3 candidatos diversos** (el mejor k de cada combinación algoritmo/conjunto de features), además de un baseline sencillo (K-Means, k=4 fijo). Cada candidato registra en MLflow, en subcarpetas separadas:
- `model/`: el modelo (escalador + clustering, como un `Pipeline` de sklearn) vía `mlflow.sklearn.log_model`, con firma e input de ejemplo.
- `data/`: el manifiesto de versión de datos.
- `plots/`: mapa de clusters y perfil por cluster.
- `config/`: un JSON con los hiperparámetros exactos de la corrida.

Todas las corridas llevan etiquetas `question`, `model_family` y `stage_of_work` (`baseline`/`search`/`candidate`) para filtrar y comparar en la UI, además de `equipo`/`git_commit`.

Para entrenar un solo modelo puntual (un algoritmo/k específico) en vez del barrido completo:
```bash
python -m src.models.p3_train --algoritmo kmeans --k 4 --conjunto-caracteristicas hourly
```
Esto es exactamente lo que invoca el entry point `train` del `MLproject` para P3 (ver sección 6.3): `mlflow run . -e train -P question=p3 -P model=kmeans -P k=4`.

*Salidas:* `data/interim/viajes_limpios.parquet`/`estaciones.parquet`, `data/processed/estaciones_caracteristicas_{hourly,bucketed}.parquet`, `models/p3_cluster_model.pkl`, `data/processed/p3_estaciones_clusters.parquet`, figuras en `reports/figures/p3_*.png` y reporte en `reports/p3_clustering_report.txt`. Los runs se registran en el experimento compartido `mibici-equipo30`. El [notebook de EDA](notebooks/1.0-db-clustering-estaciones.ipynb) presenta el análisis exploratorio previo al modelado.

### Registrar el modelo elegido en el Model Registry y promoverlo a producción
Una vez que el equipo compara los candidatos en la UI de MLflow y decide cuál pasa a producción, se registra explícitamente dando el `run_id` de esa corrida (esto **no** ocurre automáticamente en cada entrenamiento):
```bash
python -m src.models.p3_register --run-id <run_id_de_la_corrida_candidata_elegida> --reason "<justificación>"
```
O vía MLflow Project:
```bash
mlflow run . -e register_p3 -P run_id=<run_id_elegido> -P reason="Champion P3: clustering horario GMM k=2" --experiment-name mibici-equipo30 --env-manager local
```
Esto registra la versión en `mibici-equipo30-p3-grupos`, copia el linaje completo (experimento, commit de Git, versión de datos y `search_run_id`), asigna el alias de producción `champion` y certifica la inferencia en producción cargando desde `models:/mibici-equipo30-p3-grupos@champion`. Las evidencias auditables se guardan en `reports/p3/comparacion.csv`, `reports/p3/seleccion.json` y `reports/p3/registry.json`.

### Inferencia en Batch desde el Model Registry (Producción)
Asigna el cluster operacional a las estaciones directamente desde el modelo registrado en producción:
```bash
python -m src.models.p3_predict --model-uri models:/mibici-equipo30-p3-grupos@champion
```
*Salidas:* `data/processed/p3_estaciones_clusters.parquet` y resumen de distribución de clusters en consola.

Para más detalles sobre la metodología, partición, comparativas y justificación del Champion de P3, consultar la [Guía de MLflow Avanzado de P3](docs/p3-mlflow-avanzado.md).

---

## 🧪 6. Reproducibilidad con MLflow Projects (`MLproject`)

Todas las corridas de este entregable se lanzan desde el `MLproject` de la raíz del repositorio (no desde notebooks), usando `mlflow run`. El entorno (`python_env.yaml`) y los valores por defecto de rutas/fechas/semilla viven en `requirements.txt` y en el `config.py` de cada pregunta respectivamente; el `MLproject` **no los repite**, solo expone los hiperparámetros que tiene sentido variar por corrida.

### 6.1 Requisitos previos (desde un clon limpio)
```bash
git clone https://github.com/Diegob059/TC5061-Proyecto-Equipo30.git
cd TC5061-Proyecto-Equipo30
python3.12 -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```
Levanta el servidor de tracking en otra terminal (ver sección 5, Paso 0) y, en la terminal donde ejecutarás `mlflow run`, **exporta la misma URI** (el `MLproject` crea su propio run "envoltorio" por invocación antes de llamar a tu código, y necesita saber dónde registrarlo):
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"   # PowerShell
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000    # macOS/Linux
```

### 6.2 Entry point `ingest`
Ejecuta la ingesta + feature engineering de una pregunta (o las tres):
```bash
mlflow run . -e ingest -P question=p3 --env-manager local
mlflow run . -e ingest --env-manager local              # question=all (P1+P2+P3) por defecto
```

### 6.3 Entry point `train`
Recibe la pregunta, el modelo/algoritmo (opcional) y sus hiperparámetros (opcionales; si se omiten, cada módulo corre su pipeline por defecto tal como en la sección 5):
```bash
# P1: baseline + histórico + búsqueda anidada de cuatro árboles
mlflow run . -e train -P question=p1 --experiment-name mibici-equipo30 --env-manager local

# P2: baseline + candidato LightGBM
mlflow run . -e train -P question=p2 -P model=lgbm -P n_estimators=300 --experiment-name mibici-equipo30 --env-manager local

# P3: un solo run de clustering (algoritmo + k explícitos), en vez del barrido completo
mlflow run . -e train -P question=p3 -P model=kmeans -P k=4 --experiment-name mibici-equipo30 --env-manager local
```
`--env-manager local` reutiliza el entorno activo (recomendado para este proyecto, dado que ya instalamos `requirements.txt`/`python_env.yaml` manualmente); `--experiment-name mibici-equipo30` evita que el run "envoltorio" que crea `mlflow run` caiga en el experimento `Default`.

**Nota:** `--env-manager local` requiere que el virtualenv esté **activado** en la misma terminal (no basta con invocar `python.exe` por ruta completa), de lo contrario el subproceso que lanza `mlflow run` no encontrará el paquete `src` instalado en modo editable.

### Resultados ejecutados de P1

P1 tiene baseline + cinco candidatas, cuatro combinaciones en nested runs y `mibici-equipo30-p1-demanda@champion` verificado en el tracking local. El elegido (árbol profundidad 10/hoja mínima 50) mejora el MAE de inventario 11.34% en validación y 9.27% en prueba. Consulta el [reporte corregido del equipo en PDF](docs/TC5061_Reporte_MLFlow_Equipo30.pdf), su [versión editable](docs/TC5061_Reporte_MLFlow_Equipo30.docx) y la [evidencia de P1](docs/p1-mlflow-avanzado.md#resultados-ejecutados-de-p1-25-de-octubre-de-2026).

La reparación de P1 restaura el ejemplo de entrada y resuelve el `LoggedModel` de
MLflow 3 antes de promoverlo. La suite completa aprueba 36 pruebas. En una ejecución
nueva de `train` y `register_p1` mediante MLflow Project, 118 métricas comparables
coinciden exactamente con las históricas y se verifica carga/predicción por alias.
Se reutilizaron el entorno fijado y los datos procesados: no es una instalación
nueva ni una repetición de ingesta o de P2/P3. Los identificadores están en
[verificacion_reparacion_mlflow.json](reports/verificacion_reparacion_mlflow.json).
El reporte conserva las métricas y el linaje históricos y documenta pendientes
de evaluación, capturas de los tres modelos y colaboración individual.

### 6.4 Qué no se entrega
Por diseño, el repositorio **no incluye** (ver `.gitignore`): los archivos de datos crudos de MiBici, `mlflow.db`, las carpetas `mlruns/`/`mlartifacts/`, ni modelos serializados (`models/*.joblib`, `*.pkl`). Todo eso se reconstruye con los comandos de `ingest`/`train` anteriores desde un clon limpio.

---

## 🌿 7. Flujo de Trabajo con Git y GitHub (GitHub Flow)

Para asegurar la calidad, reproducibilidad y trazabilidad del código, el equipo opera bajo las siguientes políticas:

### 7.1 Rama Principal Protegida (`main`)
* La rama `main` representa código en estado de producción y está protegida contra *direct pushes*.
* Cualquier cambio debe integrarse obligatoriamente a través de un **Pull Request (PR)** verificado.

### 7.2 Convención de Nombres de Ramas
Las ramas de trabajo se crean a partir de `main` siguiendo la nomenclatura semántica:
* `feature/<descripción-corta>`: Nuevas funcionalidades o pipelines (ej. `feature/ingesta-parquet`, `feature/eda-prediccion-desabasto`).
* `fix/<descripción-corta>`: Correcciones de bugs o errores en transformaciones.
* `refactor/<descripción-corta>`: Mejoras estructurales sin alterar la lógica de negocio.
* `docs/<descripción-corta>`: Cambios en documentación o artefactos en `references/`.

### 7.3 Convención de Mensajes de Commit (Conventional Commits)
Se adopta el estándar de [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/):
```
<tipo>(<alcance>): <descripción concisa en imperativo>

[cuerpo explicativo opcional detallando el motivo y decisiones de diseño]
```
*Tipos aceptados:* `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`.

### 7.4 Política de Revisión entre Pares (Peer Code Review)
* **Apertura de PRs:** Cada integrante del equipo es responsable de abrir al menos un Pull Request describiendo los cambios, comandos de prueba y evidencia de ejecución.
* **Revisión Obligatoria:** Cada integrante debe realizar al menos una revisión formal (*Code Review*) con comentarios técnicos y aprobación explícita (*Approved*) a otro compañero antes de realizar el merge hacia `main`.
* **Fusión:** Se utiliza *Squash and Merge* o *Merge Commit* manteniendo un historial limpio.

---

## 👥 8. Integrantes del Equipo y Roles (Equipo 30)

| Integrante | Rol en el Proyecto | Responsabilidades Principales |
|---|---|---|
| **Ricardo Bahamón** | **MLOps & Pipeline Engineer** | Arquitectura del pipeline reproducible, orquestación de ingesta Parquet, configuración de MLflow, tracking de artefactos y control de versiones de datos. |
| **Diego Becerril** | **Lead Data Scientist & Modeling** | Formulación matemática del target de riesgo, ingeniería de variables de series de tiempo (lags/rollings), entrenamiento y calibración de modelos LightGBM. |
| **Gregorio Escobedo** | **Data Engineer & Quality QA** | Scraping dinámico resiliente, saneamiento de catálogos geoespaciales, validación de integridad criptográfica (SHA-256) y pruebas unitarias. |
| **Freddin Alcantara** | **Product & Deployment Lead** | Elaboración del ML Canvas, especificación de requisitos operativos del CCO, diseño de diagnósticos visuales y documentación técnica. |

---

## 📄 9. Licencia
Este proyecto está desarrollado bajo fines académicos para el curso TC5061 (Operaciones de Inteligencia Artificial) en el Tecnológico de Monterrey. Licencia MIT.
