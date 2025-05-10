import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Veri setini oku
df = pd.read_csv("../dataset.csv")

# Kategorik verileri sayıya çevir
le_gender = LabelEncoder()
le_region = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Region"] = le_region.fit_transform(df["Region"])

# Özellikler ve hedef değişken
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
y = df["BirthRateClass"]

# Özellikleri ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modeli tanımla
logreg = LogisticRegression(random_state=42, max_iter=1000)

# Skorlayıcılar
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

# Değerlendirme
results = cross_validate(logreg, X_scaled, y, cv=10, scoring=scoring)

# Sonuçları yazdır
print("Logistic Regression Modeli Sonuçları (10-Fold Cross Validation)\n")
print(f"Accuracy:   {results['test_accuracy'].mean():.4f}")
print(f"Precision:  {results['test_precision'].mean():.4f}")
print(f"Recall:     {results['test_recall'].mean():.4f}")
print(f"Specificity:{results['test_specificity'].mean():.4f}")
print(f"F1 Score:   {results['test_f1'].mean():.4f}")
print(f"MCC:        {results['test_mcc'].mean():.4f}")

"""
Logistic Regression Modeli Sonuçları (10-Fold Cross Validation)

Accuracy:   0.7100
Precision:  0.6900
Recall:     0.7747
Specificity:0.7747
F1 Score:   0.7281
MCC:        0.4258

"""