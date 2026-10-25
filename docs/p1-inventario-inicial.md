# P1: demanda e inventario inicial

## Pregunta y alcance

P1 estima los viajes que se iniciarán en cada estación hasta terminar el primer pico matutino y convierte las previsiones de salidas y llegadas en una referencia de bicicletas iniciales. No supone reabastecimiento durante ese intervalo.

- Datos: enero de 2025–septiembre de 2026; 21 archivos de viajes y catálogo de septiembre de 2026.
- Cohorte: estaciones con salidas en enero de 2025. Las estaciones nuevas no forman parte de esta evaluación.
- Pronóstico emitido a las 00:00 para la mañana de ese día. En evaluación, la historia se actualiza cada día; no se usan movimientos del día pronosticado como entrada.
- Franja: tres horas consecutivas de mayor salida entre 06:00 y 12:00, calculadas solo con 2025 para laborables y fines de semana. El balance incluye desde las 00:00 hasta el final de esa franja.
- Entrenamiento: febrero–diciembre de 2025. Validación: enero–junio de 2026. Prueba: julio–septiembre de 2026.
- Las llegadas se asignan al destino y a la fecha y hora de fin del viaje, incluso cuando cruzan medianoche.

## Entrega de tracking avanzado

Para la segunda entrega se usa [P1: MLflow Project, búsqueda y Registry](p1-mlflow-avanzado.md). Los comandos directos que siguen documentan la entrega anterior; las nuevas corridas se lanzan con `mlflow run`. La inferencia admite `--model-uri models:/mibici-equipo30-p1-demanda@champion`.

## Reproducción

El entorno de P1 usa Python 3.12 y dependencias compatibles con las restricciones generales del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Las dependencias del proyecto (incluidas las de P1) están consolidadas en el `requirements.txt` de la raíz, con versiones fijadas para Python 3.12 (ver también `python_env.yaml` para el entorno declarado del `MLflow Project`).

Los archivos se ubican en `data/raw/Viajes/` y el catálogo en `data/raw/Nomenclatura/nomenclatura_2026_09.csv`. El descargador existente del proyecto puede obtenerlos:

```bash
python -m src.data.p2_download_raw_data
```

Para ese descargador se necesitan además sus dependencias del entorno general del equipo. Si ya se dispone de los CSV, se pueden utilizar directamente:

```bash
python -m src.data.p1_dataset
python -m src.features.p1_features
python -m src.models.p1_train
```

La ingesta también acepta otra ubicación sin modificar los originales:

```bash
python -m src.data.p1_dataset --raw-dir /ruta/Viajes --catalog /ruta/nomenclatura_2026_09.csv
python -m src.features.p1_features
python -m src.models.p1_train --catalog /ruta/nomenclatura_2026_09.csv
```

El entrenamiento registra cuatro corridas en `mibici-equipo30`, utilizando `MLFLOW_TRACKING_URI` (por defecto `http://127.0.0.1:5000`, servidor con SQLite):

1. `p1_baseline_semanal_balance`: misma estación y hora de hace siete días.
2. `p1_baseline_historico_balance`: promedio de entrenamiento por estación, día de semana y hora.
3. `p1_arboles_balance`: un árbol para salidas y otro para llegadas.
4. `p1_inventario_inicial`: método elegido por flujo según MAE de validación.

Todas las alternativas usan las mismas filas, fechas y estaciones, incluidas las horas sin viajes. Cada alternativa recibe un margen calculado únicamente con validación. Se registran versión de datos, parámetros, métricas, commit, estado de cambios locales y una copia del código ejecutado. La corrida seleccionada incluye el modelo y las figuras.

El comando `python -m src.models.run_all_experiments` prepara y ejecuta esta implementación de P1 antes de los módulos existentes de P2 y P3. Requiere el entorno y los datos preparados de las otras preguntas, según la guía general. Para trabajar exclusivamente en P1 se usan los tres comandos anteriores.

Para guardar el notebook con sus tablas y gráficas después del entrenamiento:

```bash
python scripts/ejecutar_notebook_p1.py
```

## Inferencia de un día

```bash
python -m src.models.p1_predict --date 2026-10-01
```

Usa el modelo guardado y solo la historia anterior a la fecha solicitada. Con los archivos actuales puede emitir hasta el 1 de octubre de 2026. Es una demostración basada en datos históricos, no una recomendación vigente para una fecha posterior sin nueva historia.

Las fechas anteriores al fin de la calibración se rechazan. El catálogo de septiembre se mantiene como aproximación de capacidad; no se asume que su capacidad sea válida para todas las fechas históricas. No se reemplaza un catálogo o un conjunto de datos versionado sin volver a preparar y entrenar el modelo.

## Cálculo y métricas

En cada hora restamos primero las salidas y después sumamos las llegadas. La necesidad inicial es el mayor faltante acumulado. A esa cantidad se añade el percentil 90 del error de subestimación de validación y se redondea hacia arriba.

Con salidas `[10, 6, 3]` y llegadas `[2, 4, 7]`, el faltante conservador máximo es 14 bicicletas. El neto total al final sería 6, pero no representa el mayor faltante de la mañana.

- `salidas_mae`, `llegadas_mae` y sus RMSE: viajes por estación y hora.
- `inventario_mae` y `inventario_rmse`: bicicletas de requerimiento inicial conservador, antes del margen.
- `cobertura_conteos_sin_limite_capacidad`: proporción de pares estación/día donde la referencia alcanza el requerimiento horario observado. No mide satisfacción de usuarios ni disponibilidad real.

La ocupación máxima prevista se compara con la capacidad además de revisar el inventario inicial. Las referencias que no caben se marcan y no generan una sugerencia numérica. Las estaciones sin actividad en los 28 días previos se señalan para confirmar operación.

## Archivos y salidas

| Componente | Ubicación |
|---|---|
| Parámetros | `src/p1_config.py` |
| Conteos, calidad y hashes | `src/data/p1_dataset.py` |
| Rejilla completa y rezagos | `src/features/p1_features.py` |
| Modelos, evaluación y MLflow | `src/models/p1_train.py` |
| Balance y capacidad | `src/models/p1_balance.py` |
| Inferencia | `src/models/p1_predict.py` |
| Figuras | `src/visualization/p1_plots.py` |
| Notebook explicativo | `notebooks/1.3-ge-p1-balance-salidas-llegadas.ipynb` |
| Manifiesto versionado | `references/p1_data_manifest.json` |
| Métricas y recomendaciones | `reports/p1/` |
| Figuras exportadas | `reports/figures/p1/` |

El baseline anterior de demanda horaria usaba una tabla sin las horas totalmente inactivas y evaluaba desde enero de 2026. Sus métricas en `reports/p1_baseline_metrics.txt` son un antecedente; no son comparables directamente con esta evaluación. La nueva comparación recalcula el promedio histórico con la rejilla completa y las mismas particiones.

## Límites del análisis

Los viajes registrados no muestran solicitudes rechazadas ni movimientos de bicicletas realizados por operadores. Un cero no confirma que la estación estuviera abierta. El orden horario supuesto puede sobreestimar la necesidad frente a una secuencia exacta de eventos. El margen global no garantiza un nivel de servicio por estación; selección y margen comparten validación, con prueba independiente. La capacidad se revisa en el ejemplo del catálogo de septiembre, no como dato histórico de entrenamiento. No se estima el inventario físico actual.
