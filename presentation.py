import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib


from st_on_hover_tabs import on_hover_tabs

import streamlit as st

import streamlit_presentation
import streamlit_presentation.analyse
importlib.reload(streamlit_presentation.analyse)
from streamlit_presentation.analyse import repartition_par_categorie
from streamlit_presentation.analyse import repartition_longueur_categorie


import streamlit_presentation.preprocessing
importlib.reload(streamlit_presentation.preprocessing)
from streamlit_presentation.preprocessing import detection_langage_et_traduction

import streamlit_presentation.modele
importlib.reload(streamlit_presentation.modele)
from streamlit_presentation.modele import presentation_modele
from sklearn.metrics import f1_score

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.linewidth'] = 1

#on charge les donnees utilisees
data = pd.read_csv( 'data.csv')
extract_data = pd.read_csv( 'data_tr_extract.csv')
sum_data = pd.read_csv( 'data_sum_extract.csv')
test_data = pd.read_pickle( 'data_test.pkl')

from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import ast


def f1_weighted(true, pred):  

    # Classes
    classes = K.arange(0, 27) 
    true = K.one_hot(K.cast(true, 'int32'), 27)
    
    # Calcule les TP, FP, FN pour chaque classe
    tp = K.dot(K.transpose(true), K.round(pred))
    fp = K.dot(K.transpose(1-true), K.round(pred))
    fn = K.dot(K.transpose(true), 1-K.round(pred))

    # Calcule le score F1 pour chaque classe
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())

    
    weighted_f1 = K.sum(f1 * K.sum(true, axis=0) / K.sum(true))
    return weighted_f1

model = load_model("final_model_kfold.h5", custom_objects={'f1_weighted': f1_weighted})





from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
print(test_data.columns)
y_test = encoder.fit_transform(test_data["prdtypecode"])
class_labels = encoder.classes_
label_size = 27



####### Page principale
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
    st.write("")
    
    repartition_par_categorie(st, data)
    repartition_longueur_categorie(st, data)

elif tabs == "Preprocessing":
    detection_langage_et_traduction(st, extract_data, sum_data)

elif tabs == "Modèle":
    presentation_modele(st, test_data, model,class_labels,y_test)

elif tabs == "Pistes exploratoires":
    st.write("# Pistes exploratoires")
    st.write("Ici")


