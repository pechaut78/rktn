
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
import pandas as pd
import numpy as np
import translators as ts
from tqdm import tqdm

class LanguageDetector(BaseEstimator, TransformerMixin):

    __model_detect_lang = None
    def __init__(self,dest:str, src:str, confidence=0.8, defaultLang="fr",modelPath="modeles/lid.176.bin") -> None:
        super().__init__()
        self.langues = []
        self.src = src
        self.dest = dest
        self.confidence = confidence
        self.defaultLang = defaultLang
        self.modelPath = modelPath
        
    def __repr__(self):
        return '<LanguageDetector : "%s" -> "%s", confidence : %s, defaultLang lang : "%s">' % (self.src, self.dest,self.confidence,self.defaultLang)
            
    def setSrcDest(self, src, dest) -> None:
        self.src = src
        self.dest = dest
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        
        if self.src not in X.columns:
            raise ValueError(f"Columns {self.src}  not found in DataFrame.")        
        
        self.langues = []
        if(LanguageDetector.__model_detect_lang == None):
            print("Loading lid Lang model")
            LanguageDetector.__model_detect_lang =  fasttext.load_model(self.modelPath)
            
        def detect_lang(texte):
            texte = texte.replace("\n", " ")            
            langue,precision=  LanguageDetector.__model_detect_lang.predict(texte)

            if(precision<self.confidence):
                lg = 'fr'
            else :
                lg = langue[0].replace("__label__", "")
            if lg not in self.langues:
                self.langues.append(lg)

            return lg
        
        for index, row in tqdm(X.iterrows(), total=len(X)):
            X.at[index, self.dest] = self.defaultLang if pd.isnull(row[self.src]) else detect_lang(row[self.src])
        
        return X

class Translator(BaseEstimator, TransformerMixin):

    __model_translate = None
    def __init__(self, dest:str, src:str, detected_lang:str=None, defaultLang="fr", source = "yandex",verbose=False) -> None:
        super().__init__()
        self.detected_lang= detected_lang
        self.src = src
        self.dest = dest
        self.defaultLang = defaultLang
        self.verbose = verbose
        self.source = source
        #ts.preaccelerate_and_speedtest()
        
    def __repr__(self):
        return '<Translator : "%s" -> "%s", lang : %s, detected lang : "%s", Source: %s>' % (self.src, self.dest,self.defaultLang,self.detected_lang,self.source)
          
            
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        print("translating ... ",self.src)

        #checking existence of source column
        if self.dest not in X.columns:
            raise ValueError(f"Columns {self.dest} not found in DataFrame.")

        X[self.dest] = np.nan
    
        for index, row in tqdm(X[X[self.detected_lang] != self.defaultLang].iterrows()):    
            try:
                if(self.detected_lang!=None):
                    translated_text= ts.translate_text(row[self.src],translator=self.source, from_language=row[self.detected_lang],to_language=self.defaultLang)
                else:
                    translated_text= ts.translate_text(row[self.src],translator=self.source,to_language=self.defaultLang)
            except:
                prefix = ""
                if(self.detected_lang!=None):
                    prefix = f"[{row[self.detected_lang]}] - "
                print("Traslation error: ",prefix, row[self.src])
                
                                    
                translated_text = row[self.src]
            if(self.verbose):
                print(translated_text)
            X.loc[index, self.dest] = translated_text
        
        return X
    