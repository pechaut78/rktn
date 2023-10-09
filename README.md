# rktn
Rakuten Challenge 2021

Le repertoire RktnChallenge contient une classe Helper qui permet de charger, traiter les données, encoder les labels et afficher la matrice de conf.
Dans preprocessing, on trouvera le source de ce qui nous a permis de construire les pipelines de traitement.

## Preprocessing
Code pour le preprocess
```
* NLP_PReprocess: preprocess le data.csv pour traiter le texte
```

## Modeles finaux utilises
Modele retenu pour le challenge
```
* Final Model KFold: utilise les embeddings de camembert-desc et camembert-desi + VIT_embeddings
* camembert description: entraine et genere les embeddings desc
* camembert descignation: entraine et genere les embeddings desi
* VIT Embeddings: Genere les embeddings pour les images
* Flaubert - desi: genere les embeddings via Flaubert
```


## Premieres tentatives

Les premiers essais de traitement du texte, ne donnant pas de resultats satisfaisants.

```
* Simple Model : utilise une simple couche embeddings + Dense
* Simple Model tokenLayer: idem, mais utilise un layer pour tokenizer et tester les ngarms
* Simple Model size: utilise la taille de desc et desi comme nouvelle feature
* NLP_GRU : test d'un GRU, sans amélioration
* NL_RF: que donne un random forrest ?
* trans_google: genere la traduction
```

Le modele lui-meme
```
* NLP_desi: genere les embeddings de desi sur modele simple
* NLP_desc: genere les embeddings de desc sur modele simple
* NLP_descdesi: exploite les deux embeddings pour fabriquer un modele.
```
