"""Modelo portable de los dos flujos horarios de P1, sin datos personales."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from src import p1_config as cfg


class FlowRegressor(RegressorMixin, BaseEstimator):
    """Predice salidas/llegadas; el balance posterior calcula inventario inicial.

    La firma solo requiere estación y variables disponibles a las 00:00.
    Los identificadores de viajes y usuarios nunca entran al modelo.
    """

    def __init__(self, method='arbol', max_depth=None, min_samples_leaf=None,
                 random_state=None):
        self.method = method
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        if self.method not in cfg.METHODS:
            raise ValueError(f'Método desconocido: {self.method}')
        targets = pd.DataFrame(np.asarray(y), index=X.index, columns=cfg.FLOWS)
        self.models_ = {}
        params = dict(cfg.TREE_PARAMS)
        params.update({k: v for k, v in self.get_params().items()
                       if k != 'method' and v is not None})
        self.effective_params_ = params if self.method == 'arbol' else {}
        for flow in cfg.FLOWS:
            if self.method == 'arbol':
                model = DecisionTreeRegressor(**params).fit(X[cfg.VARIABLES], targets[flow])
            elif self.method == 'historico':
                table = X[['estacion', 'dia_semana', 'hora']].copy()
                table['target'] = targets[flow]
                model = table.groupby(['estacion', 'dia_semana', 'hora'])['target'].mean()
            else:
                model = None
            self.models_[flow] = {'model': model, 'fallback': float(targets[flow].mean())}
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'models_')
        result = pd.DataFrame(index=X.index)
        for flow in cfg.FLOWS:
            if self.method == 'semanal':
                values = X[f'{flow}_semana'].to_numpy()
            elif self.method == 'historico':
                keys = pd.MultiIndex.from_frame(X[['estacion', 'dia_semana', 'hora']])
                values = self.models_[flow]['model'].reindex(keys).fillna(
                    self.models_[flow]['fallback']).to_numpy()
            else:
                values = self.models_[flow]['model'].predict(X[cfg.VARIABLES])
            result[f'{flow}_previstas'] = np.maximum(0, values)
        return result


def model_input(table):
    """Excluye fecha y targets para impedir inferencia con información futura."""
    # Un único dtype numérico evita cambios de int8/int64 o float32/float64
    # al reconstruir el ejemplo JSON o recibir un batch desde otra máquina.
    return table[['estacion', *cfg.VARIABLES]].astype('float64')
