from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class filterChar(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str, target:[]) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        self.target = target
        
        
    def fit(self, X, y=None):
        return self  #    

    
    def transform(self, X):
       
        def replace(x) :
            s=x
            for char in self.target:
                if char in s:                    
                    s = s.replace(char, ' ')
                
            print(x,"-",s)
            return s
    
        X[self.dest] = X[self.src].apply(lambda x:x if pd.isna(x) else replace(x))
        
        return X