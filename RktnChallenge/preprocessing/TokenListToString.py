from sklearn.base import BaseEstimator, TransformerMixin

class TokenListToString(BaseEstimator, TransformerMixin):
    
    def __init__(self, dest:str, src:str, separator=' '):
        self.separator = separator
        self.src = src
        self.dest = dest

    def fit(self, X, y=None):
        return self  #
    
    def transform(self, X, y=None):        
        
        # Jointure des tokens pour chaque sous-liste
        X[self.dest] = X[self.src].apply(lambda x: 
        self.separator.join(x) )