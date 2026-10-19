"""Figuras del perfil horario, los modelos y el inventario inicial de P1."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src import p1_config as cfg
from src.models.p1_balance import calculate_balance


def save(name):
    plt.tight_layout()
    plt.savefig(cfg.FIGURES / name, dpi=150)
    plt.close()


def create_plots(hourly, train, test, comparison, recommendation, bundle):
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    history = hourly[hourly['timestamp'] < cfg.VALIDATION_START]
    index = pd.date_range(cfg.START, '2025-12-31 23:00:00', freq='h')
    profile = history.groupby('timestamp')[cfg.FLOWS].sum().reindex(index, fill_value=0)
    profile['hora'] = profile.index.hour
    profile['laborable'] = profile.index.dayofweek < 5
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for axis, weekday, label in zip(axes, [True, False], ['Laborable', 'Fin de semana']):
        profile[profile['laborable'] == weekday].groupby('hora')[cfg.FLOWS].mean().plot(ax=axis)
        peak = bundle['peaks'].loc[label]
        axis.axvspan(peak['inicio_pico'], peak['hora_fin'], alpha=0.12, color='gray')
        axis.set(title=label, xlabel='Hora', ylabel='Viajes medios por hora', xticks=range(0, 24, 2))
    fig.suptitle('2025: salidas y llegadas; primera franja matutina')
    save('01_perfil_flujos.png')

    comparison[['sin_contar_llegadas', 'necesidad_prevista']].mean().rename(
        {'sin_contar_llegadas': 'Sin llegadas', 'necesidad_prevista': 'Con llegadas'}
    ).plot.bar(figsize=(7, 4), rot=0, title='Referencia inicial media, sin margen')
    plt.ylabel('Bicicletas por estación y día')
    save('02_efecto_llegadas.png')

    valid = recommendation[recommendation['estado'].eq('Sugerencia revisable')]['estacion']
    activity = train.groupby('estacion')['salidas'].sum().reindex(valid).dropna()
    station = activity.idxmax() if len(activity) else bundle['stations'][0]
    row = recommendation.set_index('estacion').loc[station]
    day = row['fecha']
    predicted = calculate_balance(test, bundle['peaks'])
    observed = calculate_balance(test, bundle['peaks'], 'salidas', 'llegadas')
    fig, axis = plt.subplots(figsize=(11, 4))
    for balance, label in [(predicted, 'Flujos previstos'), (observed, 'Flujos observados')]:
        sample = balance[(balance['estacion'] == station) & (balance['fecha'] == day)]
        hours, stocks = [0], [row['bicicletas_referencia']]
        for _, hour in sample.iterrows():
            hours.extend([hour['hora'] + 0.5, hour['hora'] + 1])
            stocks.extend(row['bicicletas_referencia'] + hour[['saldo_minimo_hora', 'saldo_fin']])
        axis.plot(hours, stocks, marker='.', label=label)
    axis.axhline(0, color='red', linestyle='--', label='Sin bicicletas')
    axis.axhline(row['capacidad'], color='gray', linestyle='--', label='Capacidad del catálogo')
    axis.set(title=f'Estación {station}, {day.date()}: balance horario conservador',
             xlabel='Hora (orden supuesto dentro de cada intervalo)', ylabel='Saldo de bicicletas')
    axis.legend()
    save('03_balance_estacion.png')

    comparison.xs(station, level='estacion')[
        ['necesidad_observada', 'necesidad_prevista', 'bicicletas_referencia']
    ].plot(figsize=(12, 4), title=f'Estación {station}: requerimiento inicial en prueba')
    plt.xlabel('Fecha'); plt.ylabel('Bicicletas')
    save('04_necesidad_prueba.png')

    metrics = pd.read_csv(cfg.REPORTS / 'metricas.csv')
    test_metrics = metrics[metrics['particion'] == 'test'].set_index('metodo')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    test_metrics[['salidas_mae', 'llegadas_mae']].plot.bar(ax=axes[0], rot=15)
    axes[0].set(title='Error de flujos horarios', ylabel='MAE: viajes por estación/hora', xlabel='')
    test_metrics[['inventario_mae']].plot.bar(ax=axes[1], rot=15, legend=False)
    axes[1].set(title='Error de inventario inicial, sin margen', ylabel='MAE: bicicletas', xlabel='')
    save('05_comparacion_modelos.png')
