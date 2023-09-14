from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class filterChar(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str, target:str) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        self.target = target
        
        
    def fit(self, X, y=None):
        return self  #    

    
    def transform(self, X):
       
        def replace(x) :
            for char in self.target:
                s = x.replace(char, ' ')
            return s
    
        X[self.dest] = X[self.src].apply(lambda x:x if pd.isna(x) else replace(x))
        
        return X