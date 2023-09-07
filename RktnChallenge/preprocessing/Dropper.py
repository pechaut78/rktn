from sklearn.base import BaseEstimator, TransformerMixin

class Dropper(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_drop) -> None:
        # Nom de la colonne à supprimer
        self.column_to_drop = column_to_drop

    def fit(self, X, y=None):        
        return self

    def transform(self, X):        
        # Supprime la colonne lors de la phase de "transform"
        return X.drop(columns=self.column_to_drop,inplace=True)