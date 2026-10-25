# P1: MLflow Project, experimentación y Model Registry

## Alcance de la contribución

Continuación de `origin/main` en `f967a70` (PR #17). En ese punto ya existían servidor SQLite, dependencias fijadas, `MLproject`, despachadores ingest/train y búsqueda/registro de P3. Esta contribución completa la experimentación y ciclo de vida de **P1**. No declara completada la entrega del equipo.

## Reproducción desde un clon limpio

Python 3.12. Activar el entorno en **cada terminal**. En macOS/Linux:

```bash
git clone https://github.com/Diegob059/TC5061-Proyecto-Equipo30.git
cd TC5061-Proyecto-Equipo30
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows: `.venv\Scripts\Activate.ps1`. En una terminal aparte:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001 --workers 1
```

Se muestra 5001 porque AirPlay puede ocupar 5000 en macOS. Es una configuración de ejecución, no un cambio al valor por defecto del equipo. En la terminal del proyecto:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
mlflow run . -e ingest -P question=p1 --experiment-name mibici-equipo30 --env-manager local
mlflow run . -e train -P question=p1 --experiment-name mibici-equipo30 --env-manager local
```

En PowerShell se exporta con `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"`. `--env-manager local` utiliza el entorno activado y las versiones declaradas en `requirements.txt`, referenciadas también en `python_env.yaml`.

La ingesta obtiene exclusivamente enero de 2025–septiembre de 2026 y el catálogo de septiembre de 2026 mediante el descargador existente. Verifica los archivos contra el manifiesto previo antes de regenerarlo. Una discrepancia de SHA-256 detiene la ejecución: requiere investigar la fuente y documentar una nueva versión; nunca se reemplaza silenciosamente el catálogo por el más reciente. Los CSV y Parquet permanecen ignorados por Git.

## Comparación válida contra la propuesta

Se conserva la cohorte de estaciones con salidas en enero de 2025, rejilla horaria con ceros y variables disponibles a medianoche. Entrenamiento: febrero–diciembre de 2025; validación: enero–junio de 2026; prueba: julio–septiembre de 2026. No se ajusta ningún árbol con validación ni prueba.

El baseline es la observación de la misma estación y hora de hace siete días. Se añade el promedio histórico y cuatro combinaciones de árboles independientes para salidas y llegadas: `max_depth ∈ {6,10}`, `min_samples_leaf ∈ {50,100}`, con semilla de `p1_config.py`. El padre `p1_busqueda_arboles_inventario` contiene una hija por combinación. Cada hija es también una candidata con todos sus artefactos. Etiquetas: `question=p1`, `model_family`, `stage_of_work`, `equipo`, `git_commit`, `git_dirty`, `project_run_id` y `data_version`.

Las métricas anteriores de flujos y de inventario se conservan. Ahora se reportan también las de inventario en **validación**, mejora porcentual respecto al baseline, peor mes, desviación mensual, tiempo de ajuste y latencia por mil filas. El MAE de inventario evalúa directamente el número inicial de bicicletas; la cobertura de conteos no mide disponibilidad física ni viajes rechazados.

La selección minimiza `validation_inventario_mae`, desempata con el peor mes de validación y menor profundidad. El conjunto de prueba no interviene en esta decisión. Se añade el percentil 90 del error de subestimación de validación, igual que antes; selección y margen comparten validación, limitación que se registra y conserva. La comparación anterior de la propuesta sigue en `reports/p1/metricas.csv`; la nueva se escribe aparte en `reports/p1/avanzado/comparacion.csv`.

## Artefactos por baseline/candidata

| Ubicación MLflow | Contenido |
|---|---|
| `model` (Logged Models en MLflow 3) | Dos flujos como un estimador sklearn, firma y ejemplo sin targets |
| `data/` | Manifiesto de datos con hashes de originales, agregado y features |
| Inputs | Dataset de entrenamiento agregado mediante `mlflow.log_input`; no se suben viajes crudos |
| `config/` | Parámetros efectivos, fechas, margen, franjas, entorno y MLproject |
| `plots/` | Real contra predicho, residuales de inventario y estabilidad por mes de prueba |
| `source/` | Código fuente ejecutado, incluyendo el estimador portable |

La firma requiere `estacion` y las siete variables de `p1_config.py`, sin identificadores de viajes o personas. El modelo devuelve `salidas_previstas` y `llegadas_previstas`. El inventario se obtiene aplicando el balance y el margen registrados; no se interpreta ninguna salida horaria como inventario físico.

## Registrar, promover y verificar

Después de comparar la tabla y la UI, tomar el `selected_run_id` de `reports/p1/avanzado/seleccion.json`. El registro es explícito y exige una justificación:

```bash
mlflow run . -e register_p1 -P run_id=RUN_ID_ELEGIDO -P reason="Justificacion basada en validacion, estabilidad y costo" --experiment-name mibici-equipo30 --env-manager local
```

El módulo rechaza corridas incompletas, baselines, otras preguntas y otros experimentos. Registra `mibici-equipo30-p1-demanda`, copia experimento/corrida/commit/versión de datos, añade descripción y verifica predicciones antes de mover el alias. Carga de nuevo por `models:/mibici-equipo30-p1-demanda@champion` y compara las predicciones contra la corrida fuente. La evidencia de carga se escribe en `reports/p1/avanzado/registry.json`.

Se usa **alias `champion`** para producción académica: permite cambiar la versión elegida sin modificar la URI del consumidor y conserva cada versión y su origen. No implica un servicio desplegado. El esquema coincide con el flujo vigente recomendado por MLflow y evita depender de las etapas clásicas obsoletas.

Reproducir un árbol elegido sin repetir la búsqueda:

```bash
mlflow run . -e train -P question=p1 -P model=arbol -P max_depth=6 -P min_samples_leaf=100 --experiment-name mibici-equipo30 --env-manager local
```

El comando es un ejemplo de combinación; usar los valores del candidato efectivamente elegido. Si gana el promedio histórico: `-P model=historico`. Inferencia desde el Registry:

```bash
python -m src.models.p1_predict --date 2026-10-01 --model-uri models:/mibici-equipo30-p1-demanda@champion --output reports/p1/avanzado/sugerencia_registry.csv
```

Requiere la historia hasta ayer y el catálogo con el mismo hash de la corrida. La demostración del 1 de octubre no equivale a una recomendación actual sin datos nuevos.

## Validación

```bash
python -m pytest tests -q
```

Las pruebas nuevas verifican separación temporal, selección independiente de prueba, equivalencia de métricas con la entrega anterior, orden conservador de salidas/llegadas, fallback de estaciones y serialización/carga con firma de los tres métodos. La ejecución real y las capturas de MLflow se documentan junto con los resultados.

## Pendientes de integración del equipo

- P2: completar tres candidatas adicionales al baseline, búsqueda/artefactos con firma, validación temporal documentada y modelo en producción.
- P3: el PR #17 implementa búsqueda, artefactos y registro; aún hay que demostrar ejecución, justificar selección, asignar alias/etapa y verificar carga.
- Reunir tablas y capturas de las tres preguntas para el PDF final `TC5061_Reporte_MLFlow_Equipo30.pdf`.
- Cada integrante debe abrir un PR y revisar uno en esta entrega. Este PR necesita revisión de un compañero; no sustituye la participación de los demás.
- El tracking es local: al compartir solo Git no se transfieren la base ni los artefactos. El equipo debe reproducir en su servidor acordado; no se versionan `mlflow.db`, `mlruns/`, datos ni modelos.

## Referencias (APA 7)

MLflow. (s. f.). *MLflow Projects*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/projects/

MLflow. (s. f.). *Model Registry workflows*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/model-registry/workflow/

MLflow. (s. f.). *mlflow.sklearn*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
