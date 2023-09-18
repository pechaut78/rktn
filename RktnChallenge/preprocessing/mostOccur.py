from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from collections import Counter

class mostOccur(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str, maxlen:int, size:int) -> None:
        super().__init__()
        self.src = src        
        self.dest = dest        
        self.maxLen = maxlen
        self.size = size

    
    def transform(self, X):
        
        def most_frequent_words(X, maxLen, size):
            # Si la taille de X est inférieure à Z, retourne X
            if len(X) < maxLen:
                return X

            # Calcule la fréquence de chaque mot
            word_counts = Counter(X)
            
            needed = size - len(X)
            to_add = [word for word, _ in word_counts.most_common(needed)]            
            X.extend(to_add)
    
            return X

        
        X[self.dest] = X[self.src].apply(lambda X: most_frequent_words( X, self.maxLen, self.size))         
        return X