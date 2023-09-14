from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

class Loader:
    

    
    
    def __init__(self,
                 PATH:str,
                 data:pd.DataFrame,
                 imgsize:int,
                 batchsize:int,
                 aumgentImages:bool,
                 preprocessing_function=None,
                 sample_weight=[],                 
                 data_test = None,
                 data_train = None,
                 data_val = None
                 ):
                 
        
        self.path=PATH
        self.data=data
        self.aumgentImages=aumgentImages
        
        
        self.classes = data['prdtypecode'].unique().tolist() 
        if(data_train is not None):
            self.data_train = data_train
        
        if(data_test is None):
            self.data_train,self.data_test  = train_test_split(data, test_size=0.2, random_state=42)
        else:
            self.data_test = data_test
            
        if(data_val is None):
            self.data_train, self.data_val = train_test_split(self.data_train, test_size=0.2, random_state=42)
        else:
            self.data_val = data_val
            
        # on créé les Arglist pour les bons générateurs en fonction des paramètres
        arglist = {}
        arglist['rescale'] = 1./255
 
        if preprocessing_function is not None:
            arglist['preprocessing_function'] = preprocessing_function
        
        test_data_generator = ImageDataGenerator(
            **arglist,
            )
        
        
        # si on souhaite générer des images supplémentaires avec des transformations
        if self.aumgentImages:                        
            #arglist['shear_range'] = 0.2
            arglist['zoom_range'] = 0.2
            arglist['horizontal_flip'] = True
            #arglist['validation_split'] = 0.2
        
        train_data_generator = ImageDataGenerator(         
            **arglist    
            
        )

        self.train_data = train_data_generator.flow_from_dataframe(
            dataframe=self.data_train,  
            x_col='imgname',  
            y_col='prdtypecode',
            seed=42,
            target_size=(imgsize, imgsize),
            batch_size=batchsize,
            subset="training",
            shuffle=False,
            classes = self.classes,
            class_mode="categorical",
            sample_weight = sample_weight        
            )
        
        self.validation_data = train_data_generator.flow_from_dataframe(
            dataframe=self.data_val,
            x_col='imgname',  
            y_col='prdtypecode',
            seed=42,
            target_size=(imgsize, imgsize),
            batch_size=batchsize,
            subset="validation",
            shuffle=True,
            classes = self.classes,
            class_mode="categorical"
            )
        
        self.test_data = test_data_generator.flow_from_dataframe(
            dataframe=self.data_test,        
            x_col='imgname',  # Colonne contenant les chemins des fichiers
            y_col='prdtypecode',  # Colonne contenant les catégories des fichiers
            seed=42,
            target_size=(imgsize, imgsize),
            batch_size=batchsize,            
            classes = self.classes,
            shuffle=False,
            class_mode="categorical"
            )
        
    def getClasses(self) :

    
        return self.classes
        