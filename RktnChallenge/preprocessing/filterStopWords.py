import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

class filterStopWords(BaseEstimator, TransformerMixin):
    
    def __init__(self,dest:str, src:str, lang:[str], addendum=None) -> None:
        super().__init__()
        self.src = src
        self.dest = dest
        nltk.download('stopwords')
        
        self.stop_words = stopwords.words(lang[0])
        for i in range(1,len(lang)):
            self.stop_words.extend(stopwords.words(lang[i]))
        
        if(addendum != None):
            self.stop_words.extend(addendum)
        
        

    
    def transform(self, X):
        def filter_stopword(x):
            y = [word.lower() for word in x]
            return([word.lower() for word in y if word not in self.stop_words])

        X[self.dest] = X[self.src].apply(filter_stopword)
        
        return X