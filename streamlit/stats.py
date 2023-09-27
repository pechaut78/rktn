import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def repartition_par_categorie(st,data):
    val_code = data['prdtypecode'].unique()

    target_count = (data["prdtypecode"].value_counts(normalize=True)*100).reset_index()
    target_count.columns=["prdtypecode","pourcentage"]

    plt.figure(figsize=(10,3))
    ax = sns.barplot( x="prdtypecode", y="pourcentage", data=target_count)

    ax.axhline(y=100/len(val_code),color="green",linewidth=2, alpha=0.5)

    plt.xticks(rotation=45)
    plt.xlabel('Code produit')
    plt.ylabel('Pourcentage')
    plt.grid()
    plt.title("Distribution des valeurs de la target")

    # Afficher le graphique avec Streamlit
    col1, col2,col3 = st.columns([6,1,3])
    with col1:
        st.pyplot(plt)

    with col3:
        st.markdown('<div class="rounded-border"></div>', unsafe_allow_html=True)
        st.write("\n\n\n\n\n\n")
        st.write("La catégorie la plus présente représente 12% du corpus.")
        st.write("Si la base était uniformément répartie:")
        st.write(f"=> chaque code serait représenté à {100/len(val_code):.2f}% de la base")


def repartition_longueur_categorie(st,data):
    data["designation_length"] = data["designation"].str.len()
    data["description_length"] = data["description"].str.len()
    
    plt.figure(figsize=(10,4))
    ax = sns.histplot(x='designation_length', data=data,bins=50);
    ax.axhline(data["designation_length"].mean(),color="r",linewidth=2, alpha=0.5)
    
    plt.xticks(rotation=45)
    plt.xlabel("Longueur de la designation en caractères");
    plt.ylabel("nb d'occurences");
    plt.grid()
    plt.title("Répartition des longueurs des designations");
    
    col1, col2,col3 = st.columns([2,6,3])
    with col1:
        st.write(data["designation_length"].describe())
        
    with col2:
        st.pyplot(plt)

    with col3:        
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.write(f'=> Longueur de la designation est comprise entre {data["designation_length"].min()} et {data["designation_length"].max()} caractères')
        st.write("on a une majeure partie de la distribution entre 45 et 100 caractères, puis un pic à 250 caractères")
        