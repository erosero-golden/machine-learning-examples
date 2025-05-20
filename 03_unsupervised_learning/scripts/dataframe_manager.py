import pandas as pd
import numpy as np

class ArrayToDFTransformer():
    def __init__(self, columns):
        self.columns = columns
        self.dataframe = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            self.dataframe = pd.DataFrame(X, columns=self.columns)
            return self.dataframe
        else:
            self.dataframe = pd.DataFrame(X, columns=self.columns)
            return self.dataframe
        
    def get_dataframe(self):
        if self.dataframe is not None:
            return self.dataframe
        else:
            raise ValueError("DataFrame not created. Please call transform() first.")