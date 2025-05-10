import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Veri setini oku
df = pd.read_csv("../dataset.csv")

# Kategorik verileri sayıya çevir
le_gender = LabelEncoder()
le_region = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Region"] = le_region.fit_transform(df["Region"])

# Özellikler
X = df[[ 
    "YoungHighlyEducated",
    "Education Level Code", 
    "Average Age of Mother (years)",
    "Average Birth Weight (g)",
    "IsAdvancedDegree", 
    "IsYoungMother",
    "IsOlderMother",
    "IsLowBirthWeight",
    "Gender",
    "Region"
]]

# Hedef değişken
y = df["BirthRateClass"]

# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA uygulama
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# MLP (Yapay Sinir Ağı) modeli
mlp = MLPClassifier(hidden_layer_sizes=(110,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Değerlendirme metrikleri
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, pos_label='High'),
    'recall': make_scorer(recall_score, pos_label='High'),
    'f1': make_scorer(f1_score, pos_label='High'),
    'specificity': make_scorer(lambda y_true, y_pred: 
                                confusion_matrix(y_true, y_pred)[0,0] / 
                                (confusion_matrix(y_true, y_pred)[0,0] + confusion_matrix(y_true, y_pred)[0,1])),
    'mcc': make_scorer(matthews_corrcoef)
}

# 10 katlı cross-validation
results = cross_validate(mlp, X_pca, y, cv=10, scoring=scoring)

# Sonuçları yazdır
print("MLP (Neural Network) Modeli Sonuçları (10-Fold Cross Validation) (PCA)\n")
print(f"Accuracy:   {results['test_accuracy'].mean():.4f}")
print(f"Precision:  {results['test_precision'].mean():.4f}")
print(f"Recall:     {results['test_recall'].mean():.4f}")
print(f"Specificity:{results['test_specificity'].mean():.4f}")
print(f"F1 Score:   {results['test_f1'].mean():.4f}")
print(f"MCC:        {results['test_mcc'].mean():.4f}")


"""
MLP (Neural Network) Modeli Sonuçları (10-Fold Cross Validation) (PCA)

Accuracy:   0.7463
Precision:  0.7390
Recall:     0.7762
Specificity:0.7762
F1 Score:   0.7560
MCC:        0.4940

"""