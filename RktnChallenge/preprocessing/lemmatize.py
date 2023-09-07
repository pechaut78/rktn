from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from nltk.stem import WordNetLemmatizer

class lemmatize(BaseEstimator, TransformerMixin):
    

    
    def __init__(self,dest:str, src:str) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        self.nlp = spacy.load("fr_core_news_sm")
        
        

    
    def transform(self, X):
       
        def lemm(x) :
            text = " ".join(x)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc]
    
        X[self.dest] = X[self.src].apply(lambda x:lemm(x))
        
        return X