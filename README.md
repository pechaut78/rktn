# rktn
Rakuten Challenge

NLP_PReprocess: preprocess le data.csv pour traiter le texte

Final Model: utilise les embeddings de camembert-desc et camembert-desi + VIT_embeddings
camembert description: entraine et genere les embeddings desc
camembert descignation: entraine et genere les embeddings desi

Simple Model : utilise une simple couche embeddings + Dense
Simple Model tokenLayer: idem, mais utilise un layer pour tokenizer et tester les ngarms
Simple Model size: utilise la taille de desc et desi comme nouvelle feature
NLP_GRU : test d'un GRU, sans amélioration
NL_RF: que donne un random forrest ?
trans_google: genere la traduction

NLP_desi: genere les embeddings de desi sur modele simple
NLP_desc: genere les embeddings de desc sur modele simple
NLP_descdesi: exploite les deux embeddings pour fabriquer un modele.