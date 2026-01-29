# Étape 1 : Charger et explorer les données
import pandas as pd

data = pd.read_csv("language_detection.csv")
print("Noms des colonnes dans le dataset :")
print(data.columns)

print("\n12 premières lignes :")
print(data.head(12))

print("\nNombre de textes par langue :")
print(data["Language"].value_counts())

# Étape 2 : Séparer les variables et encoder les langues
from sklearn.preprocessing import LabelEncoder

X = data["Text"]
y = data["Language"]  # Utilisation de "Language" avec majuscule

le = LabelEncoder()
y = le.fit_transform(y)
print("\nLangues encodées (12 premières) :", y[:12])

# Étape 3 : Prétraitement du texte
import re

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^?;~0-9]', ' ', text)
    data_list.append(text)
print("\nPremier texte nettoyé :", data_list[0])

# Étape 4 : Sac de mots (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()

import gc
del data_list
gc.collect()
print("\nForme de X (vecteurs de texte) :", X.shape)

# Étape 5 : Fractionner les données
from sklearn.model_selection import train_test_split
import numpy as np

y = y.astype(np.int8)
X = X.astype(np.int16)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nForme des données d'entraînement :", x_train.shape, "Forme des données de test :", x_test.shape)

# Étape 6 : Entraîner le modèle Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)

# Sauvegarder le modèle
import pickle
filename = 'language_detection_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("\nModèle entraîné et sauvegardé sous 'language_detection_model.sav'")

# Étape 7 : Évaluer le modèle
# Étape 7 : Évaluer le modèle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
accuracy = 100 * accuracy_score(y_test, y_pred)
print(f"\nPrécision du modèle : {accuracy:.2f}%")

cm = confusion_matrix(y_test, y_pred)
cm = 100 * cm / cm.astype(float).sum(axis=1)  # Correction ici

plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt=".1f")
plt.title("Matrice de Confusion (%)")
plt.show()


# Étape 8 : Tester avec de nouvelles phrases
def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print(f"\nLangue prédite : {lang[0]}")

# Exemples de test
predict("Est-ce que cet exercice vous a permis d'avoir un aperçu introductif au traitement naturel du langage ?")
predict("Did this exercise give you an introductory overview to natural language processing?")
predict("هل أعطاك هذا التمرين نظرة عامة تمهيدية حول معالجة اللغة الطبيعية؟")
