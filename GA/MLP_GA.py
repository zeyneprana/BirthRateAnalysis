import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from geneticalgorithm import geneticalgorithm as ga
from sklearn.preprocessing import LabelEncoder

# 1. Veriyi oku
df = pd.read_csv("../dataset.csv")  

#Kategorik verileri önce dönüştür!
le_gender = LabelEncoder()
le_region = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Region"] = le_region.fit_transform(df["Region"])

# 2. Hedef ve özellikleri ayır
X = df.drop("BirthRateClass", axis=1)
y = df["BirthRateClass"]

# 3. Hedefi sayısallaştır
le = LabelEncoder()
y_np = le.fit_transform(y)



# 4. Veriyi eğitim ve gerçek test setine ayır (%80 eğitim, %20 test)
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X, y_np, test_size=0.2, random_state=42)

# 5. GA + MLP sınıfı
class MLP_GA:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.25, random_state=42)  # 60/20 split

    def train_mlp(self, X_train_selected, X_val_selected, y_train, y_val):
        model = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=200,
            alpha=0.01,
            solver='adam',
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)
        return accuracy_score(y_val, y_pred)

    def evaluate(self, selected_features):
        try:
            selected = np.array(selected_features) > 0.5
            num_selected = selected.sum()

            if num_selected < 3 or num_selected > 10:
                return 1  # Çok az ya da çok fazla özellik

            X_train_sel = self.X_train.iloc[:, selected]
            X_val_sel = self.X_val.iloc[:, selected]
            acc = self.train_mlp(X_train_sel, X_val_sel, self.y_train, self.y_val)
            print(f"Seçilen özellik sayısı: {num_selected}, Accuracy: {acc:.4f}")
            return 1 - acc  # Hata oranı (minimize edilir)
        except Exception as e:
            print("Hata:", e)
            return 1  # Kötü durum

    def run(self):
        varbound = np.array([[0, 1]] * self.n_features)
        algorithm_param = {
            'max_num_iteration': 50,
            'population_size': 10,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        model = ga(
            function=self.evaluate,
            dimension=self.n_features,
            variable_type='bool',
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
            function_timeout=300
        )

        model.run()
        return model.output_dict['variable']


# 6. GA ile en iyi özellikleri bul
if __name__ == "__main__":
    ga_model = MLP_GA(X_train_full, y_train_full)
    best_solution = ga_model.run()
    selected_indices = [i for i, bit in enumerate(best_solution) if bit == 1]

    print("\n✅ GA Seçilen Özellik İndeksleri:", selected_indices)

    # 7. En iyi özelliklerle final eğitim
    X_train_selected = X_train_full.iloc[:, selected_indices]
    X_test_selected = X_test_final.iloc[:, selected_indices]

    final_mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=200,
        alpha=0.01,
        solver='adam',
        early_stopping=True,
        random_state=42
    )

    final_mlp.fit(X_train_selected, y_train_full)
    y_pred = final_mlp.predict(X_test_selected)

    cm = confusion_matrix(y_test_final, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0

    print("\n MLP Modeli (GA ile Seçilen Özelliklerle - 10 Katlı CV)\n")
    print(f"Accuracy:    {accuracy_score(y_test_final, y_pred):.4f}")
    print(f"Precision:   {precision_score(y_test_final, y_pred):.4f}")
    print(f"Recall:      {recall_score(y_test_final, y_pred):.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1_score(y_test_final, y_pred):.4f}")
    print(f"MCC:         {matthews_corrcoef(y_test_final, y_pred):.4f}")


    # Overfitting var mı yok mu anlamak için
    y_train_pred = final_mlp.predict(X_train_selected)

train_acc = accuracy_score(y_train_full, y_train_pred)
train_prec = precision_score(y_train_full, y_train_pred)
train_rec = recall_score(y_train_full, y_train_pred)
train_f1 = f1_score(y_train_full, y_train_pred)
train_mcc = matthews_corrcoef(y_train_full, y_train_pred)

print("\nEğitim Verisindeki Performans:")
print(f"Accuracy:  {train_acc:.4f}")
print(f"Precision: {train_prec:.4f}")
print(f"Recall:    {train_rec:.4f}")
print(f"F1 Score:  {train_f1:.4f}")
print(f"MCC:       {train_mcc:.4f}")


"""

MLP Modeli (GA ile Seçilen Özelliklerle - 10 Katlı CV)

Accuracy:    0.9982
Precision:   0.9965
Recall:      1.0000
Specificity: 0.9963
F1 Score:    0.9982
MCC:         0.9964

Eğitim Verisindeki Performans:
Accuracy:  0.9968
Precision: 0.9936
Recall:    1.0000
F1 Score:  0.9968
MCC:       0.9937

"""