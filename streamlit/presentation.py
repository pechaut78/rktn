import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from st_on_hover_tabs import on_hover_tabs
import streamlit as st


from analyse import repartition_par_categorie
from analyse import repartition_longueur_categorie

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.linewidth'] = 1


data = pd.read_csv( 'data.csv')

st.set_page_config(layout="wide")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

st.title("Mon Application")

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Introduction', "Analyse", "Preprocessing", "Modèle", "Pistes exploratoires"], 
                         iconName=['apps', 'bar_chart', "sync", "memory", "topic"], default_choice=0)

st.markdown("""
<style>
    .rounded-border-parent {
        border-radius: 15px !important;
        border: 1px solid blue !important;
        background-color: lightgray !important;
    }
</style>
""", unsafe_allow_html=True)


if tabs == "Introduction":
    st.write("# Introduction")
    st.write("Ici")

elif tabs == "Analyse":
    st.write("# Analyse")
    
    st.dataframe(data.head(30))
    st.write("Ici")
    
    repartition_par_categorie(st, data)
    repartition_longueur_categorie(st, data)

elif tabs == "Preprocessing":
    st.write("# Preprocessing")
    st.write("Ici")

elif tabs == "Modèle":
    st.write("# Modèle")
    st.write("Ici")

elif tabs == "Pistes exploratoires":
    st.write("# Pistes exploratoires")
    st.write("Ici")


