# P3: MLflow Project, Clustering con Búsqueda Anidada y Model Registry

## 1. Alcance de la contribución

Esta contribución implementa la experimentación avanzada, el empaquetado como MLflow Project, la búsqueda de hiperparámetros con corridas anidadas (*nested runs*) y el ciclo de vida en el Model Registry para la **Pregunta 3** (Clustering no supervisado de estaciones en MiBici Guadalajara).

Principales componentes completados:
1. **Búsqueda exhaustiva con *Nested Runs***:
   - 36 combinaciones evaluadas bajo un run padre (`p3_busqueda_hiperparametros_clustering`) y 36 runs hijos anidados (`nested=True`), explorando:
     - Algoritmos: K-Means y Gaussian Mixture Models (GMM).
     - Rango de grupos: $k \in [2, 10]$.
     - Conjuntos de características: `hourly` (perfil horario completo de 24 horas, 52 features) y `bucketed` (perfil agregado en 7 franjas operativas, 14 features).
2. **Evaluación comparativa multicriterio (Baseline vs. 3 Candidatos diversos)**:
   - **Baseline ingenuo**: K-Means con $k=4$ fijo sobre el perfil horario (`hourly`), análogo al primer acercamiento sin ajuste de hiperparámetros.
   - **Candidato 1 (Champion)**: `hourly_gmm_k2` (GMM sobre perfil horario con $k=2$, silueta 0.2918).
   - **Candidato 2**: `bucketed_kmeans_k2` (K-Means sobre franjas operativas con $k=2$, silueta 0.1573).
   - **Candidato 3**: `bucketed_gmm_k2` (GMM sobre franjas operativas con $k=2$, silueta 0.1536).
3. **Registro completo de artefactos en cada candidato**:
   - Modelo serializado con `mlflow.sklearn.log_model` empaquetado como `Pipeline([("scaler", StandardScaler()), ("cluster", model)])`, recibiendo características directamente en crudo, con firma inferida (`signature`) y 5 ejemplos de entrada (`input_example`).
   - Conjunto de entrenamiento registrado en el linaje mediante `mlflow.log_input`.
   - Manifiesto de datos (`references/data_manifest.json` en `data/`).
   - Configuración reproducible (`config/`), mapa espacial de clusters (`figures/`) y reporte de segmentación (`reports/`).
4. **Etiquetado y gobernanza**:
   - `question: "p3"`, `stage_of_work: "candidate"` (o `"baseline"` o `"search"`), `model_family: "gmm"` / `"kmeans"`, `git_commit`, `data_version`, `search_run_id`.
5. **Model Registry y Producción**:
   - Módulo formal de registro `src/models/p3_register.py`.
   - Registro bajo el nombre oficial `mibici-equipo30-p3-grupos`.
   - Promoción con alias de producción `champion`.
   - Certificación de inferencia cargando el modelo desde el Model Registry (`models:/mibici-equipo30-p3-grupos@champion`).
   - Reportes auditables persistidos en `reports/p3/comparacion.csv`, `reports/p3/seleccion.json` y `reports/p3/registry.json`.
6. **Inferencia por lotes modular**:
   - `src/models/p3_predict.py` con opción `--model-uri` para inferir directamente desde el Model Registry o desde archivo local.

---

## 2. Reproducción desde un clon limpio

Python 3.12 o 3.13. En macOS/Linux:

```bash
git clone https://github.com/Diegob059/TC5061-Proyecto-Equipo30.git
cd TC5061-Proyecto-Equipo30
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

En una terminal independiente, levantar el servidor local de MLflow con SQLite:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001 --host 127.0.0.1
```

*(Se utiliza el puerto 5001 para evitar conflictos con AirPlay Receiver en macOS).*

En la terminal de trabajo, exportar la URI del servidor:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

### Ejecución vía MLflow Project

1. **Ingesta y Feature Engineering de P3**:
```bash
mlflow run . -e ingest -P question=p3 --experiment-name mibici-equipo30 --env-manager local
```

2. **Entrenamiento y búsqueda anidada de P3**:
```bash
# Ejecución de la búsqueda completa (36 runs anidados + Baseline + 3 Candidatos):
mlflow run . -e train -P question=p3 --experiment-name mibici-equipo30 --env-manager local

# O ejecución de un run individual específico:
mlflow run . -e train -P question=p3 -P model=gmm -P k=2 -P feature_set=hourly --experiment-name mibici-equipo30 --env-manager local
```

3. **Registro y promoción al Model Registry**:
```bash
mlflow run . -e register_p3 -P run_id=f732c016be2d4e80b05415d005826f4e -P reason="Champion P3: clustering horario GMM k=2" --experiment-name mibici-equipo30 --env-manager local
```

4. **Inferencia en batch desde el Model Registry**:
```bash
python -m src.models.p3_predict --model-uri models:/mibici-equipo30-p3-grupos@champion
```

---

## 3. Búsqueda de Hiperparámetros con Runs Anidados (*Nested Runs*)

El diseño de experimentación de P3 en `src/models/p3_train.py` organiza la exploración de forma jerárquica:
- **Run Padre (`p3_busqueda_hiperparametros_clustering`)**: Actúa como contenedor de la búsqueda sistemática de hiperparámetros.
- **36 Runs Hijos (`nested=True`)**:
  - $2 \text{ algoritmos} \times 9 \text{ valores de } k \ (2..10) \times 2 \text{ conjuntos de features} = 36 \text{ combinaciones}$.
  - Cada hijo registra sus métricas no supervisadas (`test_silhouette_score`, `test_davies_bouldin_index`, `inercia` o `bic`/`aic`).
- **Selección de Candidatos (`seleccionar_candidatos`)**:
  - Filtra el mejor $k$ por cada combinación de algoritmo y conjunto de features (garantizando diversidad algorítmica y representacional).
  - Selecciona los 3 candidatos con mayor coeficiente de silueta y los registra como **corridas candidatas completas e independientes** con todos sus artefactos y linaje vinculado mediante `search_run_id`.

---

## 4. Artefactos registrados por corrida

| Ubicación en MLflow | Descripción |
| :--- | :--- |
| `model` (Logged Model) | `Pipeline` de scikit-learn con `StandardScaler` + modelo (`GaussianMixture` o `KMeans`), firma (`signature`) y 5 ejemplos de entrada (`input_example`). |
| `data/` | Manifiesto de datos `references/data_manifest.json` con hashes SHA-256 de las fuentes crudas. |
| `Inputs` (`mlflow.log_input`) | Dataset tabular de características de estación registrado formalmente en el linaje del run. |
| `config/` | Archivo JSON con algoritmo, $k$, conjunto de features, semilla aleatoria y métricas alcanzadas. |
| `figures/` | Visualizaciones diagnósticas: mapa geográfico de clusters (`p3_clusters_estaciones_mapa.png`), curva de silueta/Davies-Bouldin vs $k$ (`p3_seleccion_k.png`) y perfil temporal horario (`p3_perfil_clusters.png`). |
| `reports/` | Reporte textual de clustering con resumen de métricas, distribución de estaciones y centroides. |

---

## 5. Tabla comparativa multicriterio (Baseline vs. Candidatos)

Evaluación sobre las 352 estaciones activas del sistema MiBici Guadalajara:

| Modelo / Candidato | Run ID | Algoritmo | $k$ | Conjunto de Features | Features ($p$) | Coeficiente de Silueta ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Inercia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Heurístico** | `3f45a08b84d3` | KMeans | 4 | `hourly` | 52 | 0.0911 | 2.3217 | 14,019.47 |
| **bucketed_gmm_k2** | `3d51e921ec7e` | GMM | 2 | `bucketed` | 14 | 0.1536 | 2.4225 | - |
| **bucketed_kmeans_k2** | `5ad7a5ae9827` | KMeans | 2 | `bucketed` | 14 | 0.1573 | 2.1235 | - |
| **hourly_gmm_k2 (Champion)** | `f732c016be2d` | GMM | 2 | `hourly` | 52 | **0.2918** | **1.9792** | - |

> Archivo de auditoría generado: [`reports/p3/comparacion.csv`](file:///Users/ricardobahamon/Documents/Maestria/Trimestre_4/Operaciones_AI/Repo_semana_6/reports/p3/comparacion.csv).

---

## 6. Selección y Justificación del Modelo Champion

El candidato **`hourly_gmm_k2`** fue seleccionado como el **Champion de P3**:

1. **Mayor calidad geométrica de agrupamiento**:
   - Alcanza el **mayor coeficiente de silueta del proyecto (0.2918)**, superando al baseline ingenuo (0.0911) en más de **+220% relativo**.
   - Registra el **menor índice de Davies-Bouldin (1.9792 frente a 2.3217 del baseline)**, indicando que los clusters generados poseen una dispersión interna significativamente menor y una separación inter-cluster nítida.
2. **Interpretabilidad y relevancia operativa de negocio**:
   - La resolución de 24 horas (`hourly`) con $k=2$ segmenta con claridad operativa los dos grandes perfiles de la red de transporte:
     - **Cluster 0 (329 estaciones, 93.5%)**: **Origen neto residencial / periférico**. Presenta su pico máximo de salidas matutino entre las 07:00 y las 09:00 hrs y demanda de arribo vespertino, requiriendo abastecimiento preventivo de bicicletas en la madrugada.
     - **Cluster 1 (23 estaciones, 6.5%)**: **Destino neto laboral / comercial / céntrico**. Presenta una concentración masiva de llegadas entre las 08:00 y las 10:00 hrs y picos de salida al cierre de la jornada (18:00 a 20:00 hrs), requiriendo desahogo prioritario de anclajes saturados al mediodía.
3. **Flexibilidad probabilística**:
   - A diferencia de K-Means, que asume agrupamientos esféricos y de igual varianza mediante fronteras rígidas, Gaussian Mixture Model permite capturar dispersiones elipsoidales y probabilidades de pertenencia, modelando con mayor realismo la heterogeneidad de viajes en estaciones de transición.

---

## 7. Model Registry y Transición a Producción

El modelo seleccionado se registró en el Model Registry bajo el nombre:
```
mibici-equipo30-p3-grupos
```
- **Alias de Producción**: `champion`.
- **URI de Inferencia en Producción**: `models:/mibici-equipo30-p3-grupos@champion`.
- **Validación de Carga e Inferencia**: Certificada con `100%` de consistencia predictiva respecto a la corrida de origen y a través del alias.
- **Evidencia auditable en repositorio**:
  - Selección: [`reports/p3/seleccion.json`](file:///Users/ricardobahamon/Documents/Maestria/Trimestre_4/Operaciones_AI/Repo_semana_6/reports/p3/seleccion.json)
  - Comparación: [`reports/p3/comparacion.csv`](file:///Users/ricardobahamon/Documents/Maestria/Trimestre_4/Operaciones_AI/Repo_semana_6/reports/p3/comparacion.csv)
  - Auditoría de registro: [`reports/p3/registry.json`](file:///Users/ricardobahamon/Documents/Maestria/Trimestre_4/Operaciones_AI/Repo_semana_6/reports/p3/registry.json)
