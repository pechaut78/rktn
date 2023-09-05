
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
import pandas as pd
import numpy as np
import translators as ts


class LanguageDetector(BaseEstimator, TransformerMixin):

    __model_detect_lang = None
    def __init__(self,dest:str, src:str, confiance=0.8, defaultLang="fr") -> None:
        super().__init__()
        self.langues = []
        self.src = src
        self.dest = dest
        self.confiance = confiance
        self.defaultLang = defaultLang
        
                
    def setSrcDest(self, src, dest) -> None:
        self.src = src
        self.dest = dest
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        
        if self.src not in X.columns:
            raise ValueError(f"Columns {self.src} or {self.dest} not found in DataFrame.")        
        self.langues = []
        if(LanguageDetector.__model_detect_lang == None):
            print("Loading lid Lang model")
            LanguageDetector.__model_detect_lang =  fasttext.load_model('modeles/lid.176.bin')
            
        def detecter_langue(texte):
            texte = texte.replace("\n", " ")            
            langue,precision=  LanguageDetector.__model_detect_lang.predict(texte)

            if(precision<0.8) :
                lg = 'fr'
            else :
                lg = langue[0].replace("__label__", "")
            if lg not in self.langues:
                self.langues.append(lg)

            return lg
        
        X[self.dest] = X[self.src].apply(lambda x: self.defaultLang if pd.isnull(x) else detecter_langue(x))
        print("fin de detection")
        return X

class Translator(BaseEstimator, TransformerMixin):

    __model_translate = None
    def __init__(self, dest:str, src:str, detected_lang:str, defaultLang="fr", source = "yandex",verbose=False) -> None:
        super().__init__()
        self.detected_lang= detected_lang
        self.src = src
        self.dest = dest
        self.defaultLang = defaultLang
        self.verbose = verbose
        self.source = source
        
                
    def setSrcDest(self, src, dest)-> None:
        self.src = src
        self.dest = dest
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        print("translating ...",self.src)
        
        if self.src not in X.columns:
            raise ValueError(f"Columns {self.src} or {self.dest} not found in DataFrame.")
        X[self.dest] = np.nan
    
        for index,row in X[X[self.detected_lang]!=self.defaultLang].iterrows():    
            try:
                translated_text= ts.translate_text(row[self.src],translator=self.source, from_language=row[self.detected_lang],to_language='fr')
            except:
                print("erreur: ", row[self.src])                
                translated_text = row[self.src]
            if(self.verbose):
                print(translated_text)
            X.loc[index, self.dest] = translated_text
        print("fin de translation")
        return X
    