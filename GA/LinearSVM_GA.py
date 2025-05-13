import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from geneticalgorithm import geneticalgorithm as ga
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. VERİYİ OKU VE ÖN İŞLE
df = pd.read_csv("../dataset.csv")

# Kategorik verileri sayısal hale getir
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

# Hedef
y = df["BirthRateClass"]

# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_np = np.array(X_scaled)
y_np = np.array(y)

# 2. FITNESS FONKSİYONU
def fitness_function(bitstring):
    indices = [i for i, bit in enumerate(bitstring) if bit == 1]
    if len(indices) == 0:
        return 1  # ceza
    X_selected = X_np[:, indices]
    svm = LinearSVC(random_state=42, max_iter=10000)
    scores = cross_val_score(svm, X_selected, y_np, cv=5, scoring='accuracy')
    return 1 - scores.mean()

# 3. GA PARAMETRELERİ
dim = X_np.shape[1]
varbound = np.array([[0, 1]] * dim)

algorithm_param = {
    'max_num_iteration': 50,
    'population_size': 20,
    'mutation_probability': 0.1,
    'elit_ratio': 0.05,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 10
}

model = ga(
    function=fitness_function,
    dimension=dim,
    variable_type='int',
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param
)

# 4. GA'YI ÇALIŞTIR
model.run()

# 5. SEÇİLEN ÖZELLİKLER
best_solution = model.output_dict['variable']
selected_indices = [i for i, bit in enumerate(best_solution) if bit == 1]
print("Seçilen Özellik İndeksleri:", selected_indices)

# 6. X MATRİSİ (SEÇİLEN ÖZELLİKLER)
X_selected = X_np[:, selected_indices]

# 7. SVM VE METRİKLER
svm = LinearSVC(random_state=42, max_iter=10000)

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

# 8. MODELİ DEĞERLENDİR
results = cross_validate(svm, X_selected, y_np, cv=10, scoring=scoring)

print("\nLinear SVM Modeli (GA ile Seçilen Özelliklerle - 10 Katlı CV)\n")
print(f"Accuracy:   {results['test_accuracy'].mean():.4f}")
print(f"Precision:  {results['test_precision'].mean():.4f}")
print(f"Recall:     {results['test_recall'].mean():.4f}")
print(f"Specificity:{results['test_specificity'].mean():.4f}")
print(f"F1 Score:   {results['test_f1'].mean():.4f}")
print(f"MCC:        {results['test_mcc'].mean():.4f}")


"""
Linear SVM Modeli (GA ile Seçilen Özelliklerle - 10 Katlı CV)

Accuracy:   0.7292
Precision:  0.7100
Recall:     0.7860
Specificity:0.7860
F1 Score:   0.7445
MCC:        0.4634

"""