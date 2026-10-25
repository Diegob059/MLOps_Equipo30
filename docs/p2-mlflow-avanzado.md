# P2: MLflow Project, Experimentación Avanzada y Model Registry

## 1. Alcance de la contribución

Esta contribución implementa la experimentación avanzada, el control de versiones, el empaquetado como MLflow Project y el ciclo de vida en el Model Registry para la **Pregunta 2** (Riesgo de Desabasto o Saturación de Estaciones en MiBici Guadalajara).

Principales componentes completados:
1. **Partición temporal estricta sin fuga (Anti-Leakage)**: Exclusión de las dos últimas horas de 2025 (`>= 2025-12-31 22:00:00`) del conjunto de entrenamiento, evitando que las ventanas futuras a 2 horas ($t+1, t+2$) utilicen viajes de enero de 2026 pertenecientes a la prueba.
2. **Evaluación comparativa de $\ge 3$ candidatos diversos**:
   - Baseline heurístico estratificado (`DummyClassifier`).
   - Candidato 1: `lgbm_balanceado` (profundidad 7, 120 árboles, lr 0.08).
   - Candidato 2: `lgbm_profundo` (profundidad 9, 180 árboles, lr 0.04).
   - Candidato 3: `lgbm_compacto` (profundidad 5, 80 árboles, lr 0.10).
3. **Registro completo de artefactos en cada candidato**:
   - Modelo serializado con `mlflow.sklearn.log_model` (`cloudpickle`), con firma inferida (`signature`) y ejemplo de entrada (`input_example`).
   - Dataset de entrenamiento registrado formalmente vía `mlflow.log_input`.
   - Manifiesto de datos (`data/`), configuración reproducible (`config/`), reportes (`reports/`) y matrices diagnósticas (`figures/`).
4. **Etiquetado y gobernanza**:
   - `question: "p2"`, `stage_of_work: "candidate"` (o `"baseline"`), `model_family: "LGBMClassifier"`, `git_commit`, `git_dirty`, `data_version`, `target_horizon: "2_hours"`.
5. **Model Registry y Producción**:
   - Módulo formal de registro `src/models/p2_register.py`.
   - Registro bajo el nombre oficial `mibici-equipo30-p2-riesgo`.
   - Promoción con alias de producción `champion`.
   - Certificación de inferencia cargando el modelo desde el Model Registry (`models:/mibici-equipo30-p2-riesgo@champion`).
   - Registro de evidencia auditable en `reports/p2/registry.json`.

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

1. **Ingesta y Feature Engineering de P2**:
```bash
mlflow run . -e ingest -P question=p2 --experiment-name mibici-equipo30 --env-manager local
```

2. **Entrenamiento y búsqueda comparativa de P2**:
```bash
# Ejecución de la suite completa (Baseline + 3 Candidatos):
mlflow run . -e train -P question=p2 --experiment-name mibici-equipo30 --env-manager local

# O ejecución de un candidato con hiperparámetros personalizados:
mlflow run . -e train -P question=p2 -P max_depth=9 -P n_estimators=180 --experiment-name mibici-equipo30 --env-manager local
```

3. **Registro y promoción al Model Registry**:
```bash
mlflow run . -e register_p2 -P run_id=RUN_ID_ELEGIDO -P reason="Candidato champion con mejor balance F1 y recall" --experiment-name mibici-equipo30 --env-manager local
```

4. **Inferencia en batch desde el Model Registry**:
```bash
python -m src.models.p2_predict --model-uri models:/mibici-equipo30-p2-riesgo@champion --sample-size 5
```

---

## 3. Partición temporal y prevención de fuga (Anti-Leakage)

El target operativo de P2 evalúa el balance neto en una ventana de 2 horas hacia adelante ($t+1, t+2$):
- **Problema de fuga temporal previo**: Las observaciones del 31 de diciembre de 2025 a las 22:00 y 23:00 proyectaban su horizonte hacia las 23:00-00:00 y 00:00-01:00 del 1 de enero de 2026, utilizando viajes del período de prueba para construir el target de entrenamiento.
- **Solución implementada**:
  Se aplicó la regla estricta:
  $$\text{train\_cutoff} = \text{split\_date} - \text{horizon\_hours} = \text{2025-12-31 22:00:00}$$
  Excluyendo exactamente **940 observaciones** (470 estaciones $\times$ 2 horas).
- **Resultado del dataset limpio**:
  - **Entrenamiento**: 4,037,300 filas ($\le$ 2025-12-31 21:00:00, con target hasta las 23:00:00 de 2025).
  - **Prueba**: 3,078,500 filas ($\ge$ 2026-01-01 00:00:00).
  - Fuga temporal: **0% (100% limpia)**.

---

## 4. Artefactos registrados por corrida

| Ubicación en MLflow | Descripción |
| :--- | :--- |
| `model` (Logged Model) | Modelo LightGBM serializado (`cloudpickle`) con firma (`signature`) y 5 filas de `input_example`. |
| `data/` | Manifiesto de datos `references/p2_data_manifest.json` con metadatos de versión y hashes. |
| `Inputs` (`mlflow.log_input`) | Dataset estructurado de características registrado en el linaje del run. |
| `config/` | Archivo JSON con hiperparámetros exactos, cortes temporales, semilla y parámetros de hardware. |
| `figures/` | Matriz de confusión multiclase y gráfico de importancia de características (top 15). |
| `reports/` | Reporte textual de clasificación con métricas de soporte, precisión y recall por clase. |

---

## 5. Tabla comparativa de resultados sobre Test

Evaluación sobre los 3,078,500 registros del período de prueba (enero–septiembre de 2026):

| Modelo | Run ID | Hiperparámetros clave | Macro F1 | Balanced Acc | Recall Desabasto (1) | Recall Saturación (2) | Latencia (ms/1k) | Tiempo ajuste (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Estratificado** | `400e74a12e3d` | DummyClassifier (stratified) | 0.3333 | 33.36% | 0.78% | 0.62% | - | <0.5 s |
| **lgbm_compacto** | `c5529d63ab68` | depth=5, n=80, leaves=16, lr=0.10 | 0.3578 | 85.88% | 85.64% | 89.51% | 0.827 ms | 7.0 s |
| **lgbm_balanceado** | `4d481b24e658` | depth=7, n=120, leaves=31, lr=0.08 | 0.3662 | 87.46% | 87.90% | 90.58% | 1.698 ms | 11.9 s |
| **lgbm_profundo (Champion)** | `8350c8c6b05d` | depth=9, n=180, leaves=63, lr=0.04 | **0.3708** | **88.00%** | **88.41%** | **90.92%** | 3.840 ms | 23.0 s |

---

## 6. Selección y Justificación del Modelo Champion

El candidato **`lgbm_profundo`** fue seleccionado como el **Champion de P2**:
1. **Desempeño técnico**: Obtiene el mayor **Macro F1 (0.3708)** y la mayor **Balanced Accuracy (88.00%)**, superando ampliamente al baseline (+0.0375 puntos en Macro F1 y +54.6 puntos en Balanced Accuracy).
2. **Prioridad de negocio operativo**: En un sistema de bicicletas compartidas, el costo de no detectar una estación vacía (desabasto) o llena (saturación) es crítico para la experiencia del usuario y la operación de las cuadrillas de balanceo. `lgbm_profundo` alcanza los recalls más altos en ambas clases críticas: **88.41% en desabasto** y **90.92% en saturación**.
3. **Costo de inferencia y escalabilidad**: Con un tiempo de inferencia de apenas **3.84 ms por cada 1,000 registros**, predecir el estado de las 470 estaciones del sistema toma menos de **2 milisegundos por hora**, permitiendo su despliegue operativo en tiempo real sin requerir aceleración GPU.

---

## 7. Model Registry y Transición a Producción

El modelo seleccionado se registró en el Model Registry bajo el nombre:
```
mibici-equipo30-p2-riesgo
```
- **Versión**: 3 (asociada al run `8350c8c6b05d49f9a2f1b7746bd0463c`).
- **Alias de Producción**: `champion`.
- **URI de Inferencia en Producción**: `models:/mibici-equipo30-p2-riesgo@champion`.
- **Validación de Carga e Inferencia**: Certificada con `100%` de coincidencia en predicciones respecto a la corrida de origen.
- **Evidencia en repositorio**: `reports/p2/registry.json`.
