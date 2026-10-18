"""Conversión de flujos horarios a inventario inicial y revisión de capacidad."""
import numpy as np
import pandas as pd
from src import p1_config as cfg
from src.data.p1_dataset import read_csv


def calculate_balance(table, peaks, departures='salidas_previstas', arrivals='llegadas_previstas'):
    t = table[['estacion', 'fecha', 'hora', departures, arrivals]].copy()
    weekday = t['fecha'].dt.dayofweek < 5
    t['hora_fin'] = np.where(weekday, peaks.loc['Laborable', 'hora_fin'],
                             peaks.loc['Fin de semana', 'hora_fin'])
    t = t[t['hora'] < t['hora_fin']].sort_values(['estacion', 'fecha', 'hora'])
    t['neto'] = t[arrivals] - t[departures]
    t['saldo_fin'] = t.groupby(['estacion', 'fecha'])['neto'].cumsum()
    t['saldo_antes'] = t['saldo_fin'] - t['neto']
    # Las llegadas de la hora se acreditan después de sus salidas.
    t['saldo_minimo_hora'] = t['saldo_antes'] - t[departures]
    return t


def summarize_balance(balance):
    summary = balance.groupby(['estacion', 'fecha']).agg(
        minimo=('saldo_minimo_hora', 'min'), maximo=('saldo_fin', 'max'))
    summary['necesidad'] = (-summary['minimo']).clip(lower=0)
    summary['maximo'] = summary['maximo'].clip(lower=0)
    return summary


def read_catalog(path):
    return read_csv(path)[['id', 'name', 'dpcapacity']].rename(
        columns={'id': 'estacion', 'name': 'nombre', 'dpcapacity': 'capacidad'})


def recommend(predictions, hourly, bundle, date, catalog):
    """Solo usa pronósticos y actividad anterior al día solicitado."""
    day = pd.Timestamp(date)
    balance = calculate_balance(predictions, bundle['peaks'])
    result = summarize_balance(balance).rename(columns={'necesidad': 'necesidad_prevista'})
    result['bicicletas_referencia'] = np.ceil(result['necesidad_prevista'] + bundle['margin']).astype(int)
    result = result.reset_index().merge(catalog, on='estacion', how='left', validate='many_to_one')
    recent = hourly[hourly['timestamp'].between(day - pd.Timedelta(days=28), day, inclusive='left')]
    activity = recent.groupby('estacion')[cfg.FLOWS].sum().sum(axis=1)
    result['actividad_28d_previa'] = result['estacion'].map(activity).fillna(0)
    result['ocupacion_maxima_prevista'] = result['bicicletas_referencia'] + result['maximo']
    conditions = [result['actividad_28d_previa'].eq(0),
                  result['capacidad'].isna() | result['capacidad'].le(0),
                  result['bicicletas_referencia'] > result['capacidad'],
                  result['ocupacion_maxima_prevista'] > result['capacidad']]
    labels = ['Confirmar operación', 'Revisar capacidad', 'Inventario inicial supera capacidad',
              'Falta espacio previsto para devoluciones']
    result['estado'] = np.select(conditions, labels, default='Sugerencia revisable')
    result['bicicletas_inicio_sugeridas'] = result['bicicletas_referencia'].where(
        result['estado'].eq('Sugerencia revisable')).astype('Int64')
    result['hora_inicio_plan'] = 0
    kind = np.where(result['fecha'].dt.dayofweek < 5, 'Laborable', 'Fin de semana')
    for col in ['inicio_pico', 'hora_fin']:
        result[col] = pd.Series(kind, index=result.index).map(bundle['peaks'][col])
    return result.drop(columns=['minimo', 'maximo'])
