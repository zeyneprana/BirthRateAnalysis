import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression modeli
log_reg = LogisticRegression(random_state=42, max_iter=1000)

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
results = cross_validate(log_reg, X_pca, y, cv=10, scoring=scoring)

# Sonuçları yazdır
print("Logistic Regression Modeli Sonuçları (10-Fold Cross Validation) (PCA)\n")
print(f"Accuracy:   {results['test_accuracy'].mean():.4f}")
print(f"Precision:  {results['test_precision'].mean():.4f}")
print(f"Recall:     {results['test_recall'].mean():.4f}")
print(f"Specificity:{results['test_specificity'].mean():.4f}")
print(f"F1 Score:   {results['test_f1'].mean():.4f}")
print(f"MCC:        {results['test_mcc'].mean():.4f}")


"""
Logistic Regression Modeli Sonuçları (10-Fold Cross Validation) (PCA)

Accuracy:   0.6558
Precision:  0.6386
Recall:     0.7213
Specificity:0.7213
F1 Score:   0.6740
MCC:        0.3181

"""