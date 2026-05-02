import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

# Afficher les colonnes
print(f"\nColonnes : {list(df.columns)}")

# Afficher les diagnostics
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")     # (500,)

from sklearn.model_selection import train_test_split

# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% pour le test
    random_state=42,    # Pour avoir les mêmes résultats à chaque fois
    stratify=y          # Garder les mêmes proportions de diagnostics
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,   # 100 arbres de décision
    random_state=42     # reproductibilité
)

# Entraîner le modèle
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions avec la réalité
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})

print(comparison)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.2%}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Matrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


# Visualiser avec seaborn
import os
os.makedirs('figures', exist_ok=True)  # Créer le dossier figures s'il n'existe pas

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
# Sauvegarde de la figure
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardée dans figures/confusion_matrix.png")


import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")

print("Modèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")


# Sauvegarder les encodeurs (indispensable pour nouvelles données)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")


# Simuler ce que fera l'API en Lab 3 :
# Charger le modèle depuis le fichier (pas depuis la mémoire)

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")


# Nouveau patient (simulation centre de santé de Médina)
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': 1,
    'fatigue': 1,
    'maux_tete': 1,
    'region': 'Dakar'
}

# Encoder les variables catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Construire le vecteur de features (IMPORTANT: même ordre que l'entraînement)
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    nouveau_patient['toux'],
    nouveau_patient['fatigue'],
    nouveau_patient['maux_tete'],
    region_enc
]

# Faire la prédiction
prediction = model_loaded.predict([features])[0]

print("Résultat prédiction :", prediction)


# =========================
# NOUVEAU PATIENT
# =========================

nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# =========================
# ENCODAGE VARIABLES
# =========================

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# =========================
# FEATURES VECTOR
# =========================

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# =========================
# PRÉDICTION
# =========================

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

# =========================
# AFFICHAGE
# =========================

print("\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")

for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")


 
print("\n--- Importance des features ---")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 50)
    print(f"  {name:20s} : {imp:.3f}  {bar}")


patients_fictifs = [
    {'age': 20, 'sexe': 'M', 'temperature': 37.0, 'tension_sys': 120,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar',
     'description': 'Jeune sans symptômes'},
 
    {'age': 35, 'sexe': 'F', 'temperature': 40.2, 'tension_sys': 105,
     'toux': False, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar',
     'description': 'Adulte avec forte fièvre'},
 
    {'age': 68, 'sexe': 'M', 'temperature': 38.8, 'tension_sys': 135,
     'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Dakar',
     'description': 'Patient âgé avec toux'},
]
 
print("\n--- Exercice 2 : Prédictions sur 3 patients fictifs ---")
for p in patients_fictifs:
    s_enc = le_sexe_loaded.transform([p['sexe']])[0]
    r_enc = le_region_loaded.transform([p['region']])[0]
    feat  = [p['age'], s_enc, p['temperature'], p['tension_sys'],
             int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r_enc]
    diag  = model_loaded.predict([feat])[0]
    prob  = model_loaded.predict_proba([feat])[0].max()
    print(f"  {p['description']:35s} → {diag:8s} ({prob:.1%})")
 
print("\n✓ Lab 2 terminé ! Modèle sérialisé dans models/model.pkl")
print("  Prochaine étape → Lab 3 : API FastAPI (tag v2)")
 