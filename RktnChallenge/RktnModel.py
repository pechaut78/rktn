import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import importlib

import RktnChallenge.GPU as GPU

from sklearn.pipeline import Pipeline
import fasttext

class ModelTrainer:
    def __init__(self,data_file:str, imgPATH="", downloadStopWords=False):
        if downloadStopWords:
            download('stopwords')
        self.data = pd.read_csv(data_file)        
        self.imgPATH = imgPATH
        self.custom_vectorizer = None
        self.tokenizer = None
        self.cat_dict = {
            '10':"Livres anciens",
            '40':"Jeux import",
            "50" : "accessoires jeux consoles ?",
            "60": "consoles rétro",
            "1140" :"figurines",
            "1160": "cartes à collectionner",
            "1180": "figurine miniatures",
            "1280": "jouet enfant",
            "1281": "jouet enfants",
            "1300": "Modèles réduits et accessoires",
            "1301": "vêtements enfant",
            "1302": "jeux d'extérieur",
            "1320": "Accessoire puériculture",
            "1560": "Cuisine et accessoire maison",
            "1920": "literie",
            "1940": "ingrédients culinaires",
            "2060": "Déco Maison",
            "2220": "accessoires animalerie",
            "2280": "Magazines",
            "2403": "livres anciens",
            "2462": "consoles et accessoires occasion",
            "2522": "papeterie",
            "2582": "?? La maison",
            "2583": "piscine et accessoires",
            "2585": "Le Jardin",
            "2705": "livres",
            "2905": "jeux en téléchargement (cf désignation) ?",
        }
        self.EmbeddingModel = None
        self.model_lang = None        
        self.preprocess_steps = []
        self._preprocess = None
    
    
    #####################################
    # Preprocessing
    #####################################
    def initPreprocess(self):
        self.preprocess_steps = []
        
    def addPreprocessStep(self, name:str,step):
        self.preprocess_steps.append((name,step))
    
    def preprocess(self,X=None):
        if(X==None):
            X = self.data
        
        self._preprocess = Pipeline(self.preprocess_steps)
       # _preprocess.fit(X)
        Y= self._preprocess.transform(X)
        return Y

    
    def saveCSV(self,name:str):
        self.data.to_csv(name)
    
        
    def initGPU(self):
        GPU.setup()
        
    