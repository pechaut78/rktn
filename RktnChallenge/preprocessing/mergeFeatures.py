from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd

class mergeFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src1:str, src2:str) -> None:
        super().__init__()
        self.src1 = src1
        self.src2 = src2
        self.dest = dest        

    
    def transform(self, X):
        X[self.dest] = X[self.src1] + X[self.src2]         
        return X