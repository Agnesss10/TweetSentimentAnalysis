# Analyse des sentiments sur les tweets

Ce notebook contient un code Python pour entraîner et évaluer des modèles d'analyse de sentiments sur des tweets en utilisant des réseaux de neurones récurrents bidirectionnels (BLSTM).

## Fonctionnalités

- Prétraitement des données textuelles (nettoyage, tokenization, lemmatization, etc.)
- Chargement de vecteurs de mots pré-entraînés (GoogleNews-vectors-negative300.bin)
- Construction d'une matrice d'embeddings pour initialiser les poids d'embeddings du modèle
- Définition d'une architecture de réseau de neurones LSTM bidirectionnel
- Entraînement et évaluation du modèle sur une tâche de classification en 3 classes (positif, neutre, négatif)
- Transfert d'apprentissage pour une tâche de classification en 7 classes d'émotions
- Calcul du coefficient de corrélation de Pearson pour évaluer les performances

## Données

Le notebook utilise deux ensembles de données :

1. `data_three/twitter_training.csv` et `data_three/twitter_validation.csv` pour la classification en 3 classes
2. `data_seven/data_train.csv` et `data_seven/data_dev.csv` pour la classification en 7 classes

## Dépendances

Les principales dépendances sont :

- NumPy
- Pandas
- Matplotlib
- NLTK
- TensorFlow
- scikit-learn
- Gensim

## Utilisation

1. Assurez-vous d'avoir installé les dépendances nécessaires présentes dans le fichier `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
2. Téléchargez le référentiel qui héberge le modèle de vecteurs de mots pré-entraîné word2vec pour le corpus Google (3 millions de vecteurs de mots anglais en 300 dimensions). 
    ```bash
    kaggle datasets download -d leadbest/googlenewsvectorsnegative300
    ```
3. Exécutez les cellules du notebook dans l'ordre pour entraîner et évaluer les modèles.

## Auteur

Ce notebook a été créé par **Agnès AOUCHICHE** et **Mourad SEKKAR** avec une participation de chacun d'eux à 50%.