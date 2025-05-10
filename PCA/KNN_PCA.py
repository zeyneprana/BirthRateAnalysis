import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Veri setini oku
df = pd.read_csv("../dataset.csv")

# Cinsiyet ve bölge gibi kategorik verileri sayıya çeviriyoruz
le_gender = LabelEncoder()
le_region = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])      # F -> 0, M -> 1 gibi
df["Region"] = le_region.fit_transform(df["Region"])      # South -> 3 gibi  

# Özellikler (Features)
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

# Hedef değişken (Target)
y = df["BirthRateClass"]

# 1. Veriyi standartlaştırma (Normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA uygulama
pca = PCA(n_components=5)  # 5 bileşen seçiyoruz
X_pca = pca.fit_transform(X_scaled)

# KNN Modelini Tanımla
knn = KNeighborsClassifier(n_neighbors=5)  # K=5 komşu ile çalışıyoruz

# Değerlendirme Metriklerini Belirle
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

# 10 Katlı Cross Validation Yap
results = cross_validate(knn, X_pca, y, cv=10, scoring=scoring)

# Sonuçları yazdırıyoruz
print("KNN Modeli Sonuçları (10 Cross Validation) (PCA)\n")
print(f"Accuracy:   {results['test_accuracy'].mean():.4f}")
print(f"Precision:  {results['test_precision'].mean():.4f}")
print(f"Recall:     {results['test_recall'].mean():.4f}")
print(f"Specificity:{results['test_specificity'].mean():.4f}")
print(f"F1 Score:   {results['test_f1'].mean():.4f}")
print(f"MCC:        {results['test_mcc'].mean():.4f}")

"""
KNN Modeli Sonuçları (10 Cross Validation) (PCA)

Accuracy:   0.7300
Precision:  0.7257
Recall:     0.7573
Specificity:0.7573
F1 Score:   0.7383
MCC:        0.4627

"""