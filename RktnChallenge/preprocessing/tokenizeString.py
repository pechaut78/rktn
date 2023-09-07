from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd

class tokenizeString(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        self.tokenizer = nltk.RegexpTokenizer(r'\w+')

    
    def transform(self, X):
        X[self.dest] = X[self.src].apply(lambda contenu: [] if pd.isnull(contenu) else self.tokenizer.tokenize(contenu))
        
        return X