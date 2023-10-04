import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def detection_langage_et_traduction(st,data, sum_data):
    
    data_lang = data[(data["desi_langue"] != "fr") | (data["desc_langue"] != "fr")]
    data_lang = data_lang[["designation","desi_langue","tr_designation", "description","desc_langue","tr_description"]]
    st.write('Utilisation de FastText pour détecter la langue: champ "desi_langue", champ "desc_langue"')
    st.dataframe(data_lang)
    st.write("Nous remarquons que la langue n'est pas toujours correctement détectée: nous acceptons ce problême, plutôt que de traduire le texte sans fournir la langue d'origine, le résultat étant nettement moins bon.")
    
    st.markdown("---")
    st.write("")
    st.subheader("Génération de résumés")
    st.write("")
    st.write("Certaines descriptions dépassent notre limitation en terme de tokens d'entrée du modèle, aussi, plutôt de couper le texte à l'aveugle, nous choisissons de résumer les descriptions.")
    st.write("")
    st.markdown("Le modèle Barthez [moussaKam/barthez-orangesum-abstract](https://huggingface.co/moussaKam/barthez-orangesum-abstract) propose de résumer des textes en francais, il utilise des mots et morceaux de phrases provenant du texte lui-même. Notre objectif est de conserver le sujet du texte et avoir les caractéristiques principales")
    
    st.image("summarize.png", use_column_width=False)
    st.write("Si le nombre de mots est supérieur à 200, nous retournons la description originale, sinon nous la résumons avec un objectif de 200 mots. Notre limite de token est de 250, en prenant 200 nous gardons une marge de 50.")
    
    data_sum = sum_data[["description","tr_description_sum"]]
    st.dataframe(data_sum)  
    st.write("")
    st.markdown("---")
    st.write("")
    st.write("Nous appliquons un prétraitement aux images, qui détecte la présence de padding dans les images, et le réduit au minimum possible. Ensuite, nous transformons les images en 224x224 pour correspondre au format VGG16 et VIT")
    st.write("")
    
    st.image("resize.png",  use_column_width=True)
    st.image("samples.png",  use_column_width=True)
    st.write("Un grand nombre de catégories peuvent ainsi éviter la perte due au downscale des images, sauvant presque toutes les cartes à collectionner, les cartes postales, les revues, etc. Nous allions ainsi une taille idéale pour le modèle (celle sur laquelle il a été pré-entrainé) et une perte d'information minimisée.")