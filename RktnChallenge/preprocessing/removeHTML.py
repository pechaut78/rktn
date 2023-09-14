from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from unidecode import unidecode

class removeHTML(BaseEstimator, TransformerMixin):
    def __init__(self, dest:str) -> None:
        super().__init__()
        self.dest = dest
    
    
    def transform(self, X):            
        if self.dest not in X.columns:
            raise ValueError(f"Columns {self.src} or {self.dest} not found in DataFrame.")
        
        X[self.dest] = X[self.dest].apply(lambda x: "" if(pd.isna(x)) else unidecode(BeautifulSoup(x, "html.parser").text))
        
        return X