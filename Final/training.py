import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from epoch import sleepRecording
import scipy.signal 
import scipy.stats
from scipy.signal import welch
from itertools import product

# load data


FREQ_BANDS = {'SO':[0.5, 1],
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 13],
            'sigma': [13,15],
            'beta': [15, 30],
            'gamma': [30, 60]}

s = sleepRecording()

s.init_from_file("data/SC4001E0-PSG.edf","data/SC4001EC-Hypnogram.edf")


# Étape 1: Préparation des données

def extract_features1(epochs):
    # Exemple de fonction pour extraire des caractéristiques des époques
    features = []
    labels = []
    for epoch in epochs:
        # Exemple de caractéristique: moyenne du premier canal EEG
        mean_feature = np.mean(epoch.get_channle_by_name("EEG Pz-Oz"))
        features.append([mean_feature])
        labels.append(epoch.label)
    return features, labels

def extract_features(epochs):
    features = []
    labels = []

    for epoch in epochs:
        epoch_features = []

        for channel_name in ["EEG Pz-Oz", "EEG Fpz-Cz"]:
            channel_data = epoch.get_channle_by_name(channel_name)

            # Statistiques de base
            epoch_features.append(np.mean(channel_data))
            epoch_features.append(np.var(channel_data))
            epoch_features.append(np.std(channel_data))
            epoch_features.append(np.max(channel_data))
            epoch_features.append(np.min(channel_data))

            # Caractéristiques fréquentielles (en utilisant la méthode de Welch)
            f, Pxx = scipy.signal.welch(channel_data, epoch.freq)
            for band in FREQ_BANDS.values():
                power_band = np.trapz(Pxx[(f >= band[0]) & (f <= band[1])], f[(f >= band[0]) & (f <= band[1])])
                epoch_features.append(power_band)

            # Caractéristiques temporelles
            epoch_features.append(scipy.stats.skew(channel_data))
            epoch_features.append(scipy.stats.kurtosis(channel_data))

        features.append(epoch_features)
        labels.append(epoch.label)

    return features, labels


features, labels = extract_features(s.epochs)
df = pd.DataFrame(features)
df['label'] = labels

# Convertir les étiquettes en valeurs numériques
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
df['label'] = df['label'].map(label_mapping)

# Séparer les caractéristiques et les étiquettes
X = df.drop('label', axis=1)
y = df['label']


# Étape 2: Entraînement du modèle XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_distribution = y_test.value_counts()

# Afficher le nombre d'échantillons pour chaque stade de sommeil
print("Distribution des étiquettes (stades de sommeil) :")
print(label_distribution)


model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_mapping))
model.fit(X_train, y_train)

# Évaluation du modèle
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Précision: {accuracy * 100.0}%")

# Après avoir effectué des prédictions
predictions = model.predict(X_test)

# Afficher la matrice de confusion

conf_matrix = confusion_matrix(y_test, predictions)

print("Matrice de confusion :")
print(conf_matrix)

plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar()
tick_marks = np.arange(len(label_mapping))
plt.xticks(tick_marks, label_mapping, rotation=45)
plt.yticks(tick_marks, label_mapping)

# Ajouter des étiquettes aux cellules
thresh = conf_matrix.max() / 2.
for i, j in product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, conf_matrix[i, j],
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Étiquette réelle')
plt.xlabel('Étiquette prédite')
plt.show()




