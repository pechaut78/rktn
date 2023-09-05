from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np


class CopyWhereNan(BaseEstimator, TransformerMixin):

    __model_detect_lang = None
    def __init__(self,dest:str, src:str) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        
                
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        
        if self.src not in X.columns or self.dest not in X.columns:
            raise ValueError(f"Columns {self.src} or {self.dest} not found in DataFrame.")
    
        mask = pd.isnull(X[self.dest])
        X.loc[mask, self.dest] = X.loc[mask, self.src]
        return X