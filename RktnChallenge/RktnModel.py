import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import importlib

import RktnChallenge.GPU as GPU
from RktnChallenge import imgLoader
from RktnChallenge import callBacks
from sklearn.pipeline import Pipeline
import fasttext


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self,data_file:str, imgPATH="", downloadStopWords=False):
        if downloadStopWords:
            download('stopwords')
        self.data = pd.read_csv(data_file)        
        self.imgPATH = imgPATH
        self.custom_vectorizer = None
        self.encoder = None
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
        
        self._preprocess = Pipeline(self.preprocess_steps, verbose=True)
       # _preprocess.fit(X)
        Y= self._preprocess.transform(X)
        return Y

    
    def saveCSV(self,name:str):
        self.data.to_csv(name)
    
        
    def initGPU(self):
        GPU.setup()
    
    ######################################
    # Vectorizer
    
    def create_vectorizer(self,name:str, X,max_features=3000,ngram_range=(1,3),num_words=None):
        if(name=="tfidf"):
            self.vectorizer = TfidfVectorizer(max_features = max_features, ngram_range=ngram_range)
            self.vectorizer.fit(X)
            self.vocab_size = len(self.vectorizer.vocabulary_)
        if(name=="bow"):
            self.vectorizer = CountVectorizer(max_features = max_features, ngram_range=ngram_range)
            self.vectorizer.fit(X)
        if(name=="hashing"):
            self.vectorizer = HashingVectorizer(n_features = max_features, ngram_range=ngram_range)
            self.vectorizer.fit(X)
        if(name=="custom"):
            self.vectorizer = self.custom_vectorizer
        if(name=="tokenizer"):
            self.vectorizer = Tokenizer(lower=False, filters='', split=' ',num_words=num_words)
            self.tokenizer = self.vectorizer
            self.tokenizer.fit_on_texts(X)

            self.word2idx =  self.tokenizer.word_index
            self.idx2word =  self.tokenizer.index_word
            self.vocab_size = len(self.word2idx)+1     
            
    def avg_vector(self,text:str, vector_size):
            v=[]
            txt = text.split()
            found = False
            #pour chaque mot dans le texte
            for mot in txt:
                #si le mot est dans le dictionnaire
                if mot in self.custom_vectorizer:
                    #on ajoute le vecteur du mot
                    v.append(self.custom_vectorizer[mot])
                    found=True
            #si le mot n'est pas dans le dictionnaire
            if not found:                
                #on ajoute un vecteur de 0
                return np.zeros((vector_size))
            
            #on renvoie la moyenne des vecteurs
            return np.mean(v,axis=0)
    
    
    def vectorizer_transform(self,text):
        if(self.vectorizer ==  self.custom_vectorizer):
            res = []
            vector_size = self.custom_vectorizer.vector_size
            txt = text.values
            #pour chaque texte
            for i in range(0,len(txt)):
                #on ajoute le vecteur du texte
                res.append(self.avg_vector(text = txt[i], vector_size = vector_size))
                
            #on récupère un tableau de vecteurs
            #on le transforme en dataframe dont chaque col
            #est une composante.
            arr= np.array(res)
            dfs = []
            
            for i in range(0,len(arr)):                
                new_row = pd.DataFrame(arr[i].reshape(1, -1))
                dfs.append(new_row)
            return pd.concat(dfs, ignore_index=True)
        if(self.vectorizer ==  self.tokenizer):   
  
            
            X = self.tokenizer.texts_to_sequences(text)
            self.max_seq_length = max(len(seq) for seq in X)
            return pad_sequences(X, maxlen=self.max_seq_length, padding='post', truncating='post')
        
        return self.vectorizer.transform(text).toarray() 
        
    
    def encodeLabel(self,src:str):
        if(self.encoder==None):
            self.encoder =  LabelEncoder()    
        return self.encoder.fit_transform(self.data[src])

    def getLabelSize(self):
        return len(self.encoder.classes_)
    
    def evaluateTestResults(self,y_true,y_pred, encoder=None):
        
        weighted_f1_score = f1_score(y_true, y_pred,average='weighted')
        print("weighted F1 score:",weighted_f1_score)
    
        if encoder==None:
            encoder = self.encoder
        class_labels = encoder.classes_
        conf_matrix = confusion_matrix(y_true, y_pred)

        row_sums = conf_matrix.sum(axis=0)
        normalized_conf_matrix = conf_matrix / row_sums[ np.newaxis,:]*100



        plt.figure(figsize=(10, 10))
        sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues',fmt='.0f',
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    linewidths=1.5)
        plt.xlabel('Prédictions')
        plt.ylabel('Réelles')
        plt.title('Matrice de Confusion')
        plt.show()
        
        #CNN
    def loadData(self,
                 PATH:str,
                 imgsize:int,
                 batchsize:int,
                 aumgentImages:bool,
                 preprocessing_function=None,
                 customized_data = False,
                 data = None,
                 sample_weight = [],
                 data_test = None,
                 data_train = None,
                 data_val = None
                 ):
        if(customized_data):
           self.data = data
        self.dataLoader = imgLoader.Loader(PATH, self.data, imgsize,batchsize=batchsize,aumgentImages=aumgentImages,preprocessing_function=preprocessing_function,sample_weight=sample_weight,
            data_test = data_test,
            data_train = data_train,
            data_val = data_val)        

        
    def train(self, model=None, X_custom=False,X = None, epochs:int=50,callbacks=[]):
        if model is None:
            model = self.tuned_model     
        if(X_custom==False):
            X=self.dataLoader.train_data
        if(len(callbacks) == 0):
            callbacks = callBacks.getList()   
        return model.fit(x=X,
                epochs=epochs, validation_data = self.dataLoader.validation_data,
                callbacks=callbacks 
                )
        
    def save(self):
        if(self.tuned_model is not None):
            self.tuned_model.save(self.project_name+".keras")
        else:
            print("No model to save")