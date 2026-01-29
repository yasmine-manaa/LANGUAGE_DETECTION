# LANGUAGE_DETECTION
# 🌍 Détection de Langue - NLP & Machine Learning

Ce projet implémente un système capable d'identifier automatiquement la langue d'un texte parmi **17 langues différentes**. Réalisé dans le cadre du cours "Intelligences humaine et artificielle du langage et du son" à **SUP'COM**.

## 🎯 Objectif
L'objectif est de transformer des données textuelles brutes en vecteurs numériques pour qu'un algorithme de Machine Learning puisse les classer par langue.



## 🧠 Approches de Vectorisation
Nous avons exploré deux méthodes de traitement du langage naturel (NLP) :
* **Bag of Words (BoW) :** Une approche basée sur l'occurrence (fréquence) simple des mots.
* **TF-IDF :** Une approche plus fine qui pondère l'importance des mots en fonction de leur rareté dans le corpus.

L'algorithme utilisé est le **Multinomial Naive Bayes**, reconnu pour son efficacité en classification de texte.

## 📁 Structure du Projet
- `language_detection_bow.py` : Script utilisant l'approche Sac de Mots.
- `language_detection_tfidf.py` : Script utilisant l'approche TF-IDF (Version optimisée).
- `Language_Detection.csv` : Le dataset contenant les textes et les labels.
- `requirements.txt` : Liste des dépendances Python.

## 🛠️ Installation et Utilisation
1. Installez les bibliothèques nécessaires :
   ```bash
   pip install -r requirements.txt
2. Lancer les 2 scripts séparément:
   python language_detection_bow.py
   python language_detection_tfidf.py
   
📊 Résultats
Le modèle TF-IDF a montré une meilleure capacité à ignorer les mots inutiles (stop words) et à se concentrer sur les termes linguistiques discriminants, atteignant une précision de plus de 95%.
