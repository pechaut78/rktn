import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def presentation_modele(st,data, model,class_labels, y_test):
    
    st.write('Notre modèle prend les embeddings de Camembert pour les descriptions et designations (séparemment), les embeddings de FlauBert pour les descriptions, les embeddings VIT pour les images et les tailles des champs de texte.')
    
    st.image("model.png",  use_column_width=True)
    #afficher une image du modele
    #afficher les embeddings en extrait
    #ajouter un bouton qui declanche le training
    if st.button("Prédire"):
        X1_test = data["embeddings_desi"].values
        X1_test = np.stack(X1_test).astype(np.float32)
        X2_test = data["embeddings_desc"].values
        X2_test = np.stack(X2_test).astype(np.float32)
        X3_test = data["embedding_vit"].values
        X3_test = np.stack(X3_test).astype(np.float32)
        X4_test = data["designation_length_normalized"].values
        X5_test = data["description_length_normalized"].values
        X6_test = data["embeddings_desi_Flaubert"].values
        X6_test = np.stack(X6_test).astype(np.float32)
        y_pred = model.predict([X1_test, X2_test,X3_test,X4_test,X5_test,X6_test])
        y_pred_ids = np.argmax(y_pred, axis=-1)

        weighted_f1_score = f1_score(y_test, y_pred_ids, average='weighted')
        st.write("weighted F1 score:",weighted_f1_score)
    
 
        conf_matrix = confusion_matrix(y_test, y_pred_ids)

        row_sums = conf_matrix.sum(axis=0)
        normalized_conf_matrix = conf_matrix / row_sums[ np.newaxis,:]*100


        st.title("Matrice de Confusion Normalisée")
        plt.figure(figsize=(10, 10))
        sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues',fmt='.0f',
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    linewidths=1.5)
        plt.xlabel('Prédictions')
        plt.ylabel('Réelles')
        plt.title('Matrice de Confusion')
        st.pyplot(plt)
        
    #afficher la matrice de conf.
    st.dataframe(data.head(10))
    
    