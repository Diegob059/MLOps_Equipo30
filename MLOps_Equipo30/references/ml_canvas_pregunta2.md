# Machine Learning Canvas: Predicción de Riesgo en Estaciones MiBici (Pregunta 2)

> **Pregunta de Negocio:** *¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro (2 horas)?*  
> **Sistema Objetivo:** MiBici (Sistema de Bicicletas Compartidas del Área Metropolitana de Guadalajara).

---

## 1. Propuesta de Valor (Value Proposition)
Anticipar con 2 horas de ventana operativa el riesgo crítico de **Desabasto** (estación vacía) o **Saturación** (estación sin anclajes libres) a nivel estación-hora. Esto permite a los despachadores y cuadrillas de furgonetas rebalancear bicicletas proactivamente antes de que ocurra la falla del servicio, mejorando la disponibilidad percibida y la fidelización del usuario.

## 2. Usuarios y Stakeholders (End Users)
* **Despachadores del Centro de Control de Operaciones (CCO):** Priorizan rutas de furgonetas de rebalanceo basadas en estaciones con probabilidad de riesgo $\ge 0.50$.
* **Conductores de Cuadrillas de Balanceo:** Reciben listas jerarquizadas de estaciones críticas a atender durante turnos pico.
* **Planificadores de Movilidad (AMIM / IMEPLAN):** Identifican cuellos de botella estructurales donde la capacidad física instalada es insuficiente ante la demanda sistemática.

## 3. Formulación de la Tarea de Machine Learning (ML Task)
* **Tipo de Tarea:** Clasificación Multiclase Supervisada a nivel estación-hora ($t$).
* **Horizonte de Predicción:** 2 horas hacia adelante ($t+1, t+2$).
* **Definición del Target ($\Delta_{\text{net}}$ y Ratio):**
  $$\Delta_{\text{net}} = (\text{Llegadas}_{t+1} + \text{Llegadas}_{t+2}) - (\text{Salidas}_{t+1} + \text{Salidas}_{t+2})$$
  $$\text{Ratio} = \frac{\Delta_{\text{net}}}{\text{Capacidad}}$$
* **Clases Operativas:**
  * **Clase 1 (Riesgo Desabasto / Vacío):** $\text{Ratio} \le -0.50$ (déficit neto $\ge 50\%$ de la capacidad de la estación).
  * **Clase 2 (Riesgo Saturación / Lleno):** $\text{Ratio} \ge 0.50$ (superávit neto $\ge 50\%$ de la capacidad de la estación).
  * **Clase 0 (Operación Normal):** $-0.50 < \text{Ratio} < 0.50$.

## 4. Fuentes de Datos e Ingesta (Data Sources)
* **Datos Abiertos MiBici (2025 - 2026):** 21 archivos CSV mensuales (12 meses de 2025 y 9 meses de 2026 hasta septiembre) con millones de transacciones de viajes (`Viaje_Id`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`).
* **Catálogo de Nomenclatura:** Catálogo geoespacial y físico de estaciones (`id`, `name`, `latitude`, `longitude`, `dpcapacity`).
* **Rejilla Cartesiana Espacio-Temporal:** Rejilla completa de estaciones activas $\times$ 15,312 horas continuas con **Zero-Filling** (la inactividad es un cero físico real).

## 5. Ingeniería de Características (Features - Estrictamente $t \le 0$)
1. **Lags de Demanda y Oferta:** Salidas y llegadas en $t-1$, $t-2$, $t-24$ (día anterior) y $t-168$ (semana anterior).
2. **Medias Rodantes de Balance Neto:** Ventanas de 3 horas y 12 horas del balance (`Llegadas - Salidas`), aplicando `.shift(1)` para evitar fuga de información (*zero data leakage*).
3. **Variables Cíclicas del Calendario:** Seno y coseno de la hora del día ($2\pi h / 24$) y día de la semana ($2\pi dow / 7$).
4. **Atributos Geoespaciales y Físicos:** Capacidad total de anclajes de la estación, latitud y longitud.

## 6. Modelado y Benchmarking (Models)
* **Baseline Heurístico:** `DummyClassifier(strategy='stratified')` que predice según las frecuencias históricas marginales (Macro F1: $\approx 0.33$).
* **Modelo Candidato:** `LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced')`. La ponderación balanceada penaliza fuertemente el error en las clases minoritarias (desabasto y saturación representan menos del 1% cada una).

## 7. Métricas de Evaluación (Metrics)
* **Métrica Principal:** **Macro F1-Score** (pondera por igual las 3 clases, impidiendo que el 98% de la clase Normal enmascare la incapacidad de detectar eventos críticos).
* **Métricas Operativas Secundarias:**
  * **Balanced Accuracy:** Precisión promedio balanceada entre clases ($\ge 87\%$).
  * **Recall en Desabasto:** Tasa de detección de estaciones que efectivamente se vacían ($\ge 88\%$).
  * **Recall en Saturación:** Tasa de detección de estaciones que efectivamente se colapsan ($\ge 90\%$).
* **Matriz de Confusión:** Inspección de falsos positivos preventivos vs. falsos negativos críticos.

## 8. Criterio de Decisión y Costos Asimétricos
En logística de bicicletas públicas, **el costo de un falso negativo es crítico** (un usuario varado que abandona el sistema por no encontrar bici o anclaje). Por el contrario, **el costo de un falso positivo es bajo** (enviar una furgoneta a inspeccionar una estación que aún tenía bicis). El modelo prioriza el *Recall* operacional manteniendo un nivel razonable de precisión.

## 9. Despliegue y Ejecución (Deployment & Serving)
* **Modo de Inferencia:** Procesamiento por lotes horario (*Batch Scoring* cada hora en el minuto :55) para emitir alertas de la ventana $[t+1, t+2]$.
* **Latencia:** Inferencia de las 370 estaciones ejecutada en menos de 10 segundos sobre CPUs estándar.
* **Artefactos Serializados:** Modelo en `models/lgbm_risk_classifier.joblib` y predicciones en `data/processed/test_predictions.parquet`.

## 10. Monitoreo, Linaje y Re-entrenamiento (MLOps Lifecycle)
* **Trazabilidad de Experimentos:** Registro de parámetros, métricas, artefactos gráficos y versiones en **MLflow**.
* **Linaje y Versionado de Datos:** Manifiesto criptográfico en `references/data_manifest.json` con hashes SHA-256 de los datasets de entrada.
* **Disparadores de Re-entrenamiento:** Caída del Macro F1 por debajo de 0.35 en ventanas mensuales móviles o deriva en la distribución de viajes por cambios en la infraestructura ciclista del AMG.
