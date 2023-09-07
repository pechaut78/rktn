from sklearn.base import BaseEstimator, TransformerMixin
import re
import pandas as pd



class regularExprSub(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str, filtr:str) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        self.filtr = filtr
        
        
    def fit(self, X, y=None):
        return self  #    

    
    def transform(self, X):
                   
        X[self.dest] = X[self.src].apply(lambda x: "" if pd.isna(x) else re.sub(re.compile(self.filtr), '', x))
    
        
        return X