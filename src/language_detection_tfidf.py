import pandas as pd
import numpy as np
import re
import pickle
import string as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Étape 1 : Ressources NLTK ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Étape 2 : Prétraitement ---
word_net = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_preprocess(text):
    # Enlever ponctuation et chiffres
    text = re.sub(r'[!@#$(),\n"%^?;~0-9]', ' ', text)
    tokens = nltk.word_tokenize(text.lower())
    # Filtrage et Lemmatisation
    tokens = [word_net.lemmatize(w) for w in tokens if len(w) > 3 and w not in stop_words]
    return " ".join(tokens)

# --- Étape 3 : Chargement ---
print("Chargement des données...")
data = pd.read_csv("language_detection.csv")

le = LabelEncoder()
y = le.fit_transform(data["Language"])

print("Nettoyage du texte...")
data['clean_text'] = data['Text'].apply(clean_preprocess)

# --- Étape 4 : TF-IDF (SANS .toarray()) ---
# Enlever .toarray() permet de garder la matrice en format "sparse" (léger)
tfidf = TfidfVectorizer(max_features=10000) # Limiter à 10k mots pour plus de stabilité
X_tfidf = tfidf.fit_transform(data['clean_text']) 

print(f"Matrice créée avec succès (Format Sparse)")

# --- Étape 5 : Entraînement ---
x_train, x_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=42)

model_tfidf = MultinomialNB()
model_tfidf.fit(x_train, y_train)

# Sauvegarde
pickle.dump(model_tfidf, open('language_detection_model_tfidf.sav', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# --- Étape 6 : Évaluation ---
y_pred = model_tfidf.predict(x_test)
print(f"\nPrécision : {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Affichage Matrice
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de Confusion (Sparse Mode)")
plt.show()

# --- Étape 7 : Prédiction ---
def predict_language(text):
    proc = clean_preprocess(text)
    vec = tfidf.transform([proc]) # Pas de .toarray() ici non plus
    pred = model_tfidf.predict(vec)
    print(f"Texte : {text[:40]}... -> Prédit : {le.inverse_transform(pred)[0]}")

print("\n--- TESTS ---")
predict_language("Ceci est un test de performance.")
predict_language("This is a performance test.")