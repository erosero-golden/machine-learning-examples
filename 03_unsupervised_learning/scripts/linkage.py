import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

# --- Wrapper para linkage ---
class LinkageWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, method='ward'):
        self.method = method
        self.linkage_matrix_ = None

    def fit(self, X, y=None):
        # Asegurarse de que X es numérico
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.linkage_matrix_ = linkage(X, method=self.method)
        return self

    def transform(self, X):
        # Devuelve los mismos datos (útil para pipelines largos)
        return X

    def get_linkage_matrix(self):
        return self.linkage_matrix_
    