"""
BAGIAN B: IMPLEMENTASI & EVALUASI
Prediksi "Lulus Tepat Waktu" - Decision Tree vs Naive Bayes
=============================================================
Mata Kuliah: Machine Learning Dasar

Struktur:
1. Load & Preprocessing (missing value, encoding, split 80:20)
2. Modeling: DecisionTree(max_depth=3), DecisionTree(max_depth=None), GaussianNB
3. Evaluasi: Akurasi, Precision, Recall, F1 + Confusion Matrix
4. Visualisasi: Plot pohon (depth=3) + bar chart perbandingan metrik
5. Analisis Error: 5 contoh salah prediksi
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("mahasiswa_lulus.csv")
print("Shape awal:", df.shape)
print("\nMissing value per kolom:\n", df.isna().sum())

TARGET = "Lulus_Tepat_Waktu"
NUM_COLS = ["IPK", "Kehadiran", "Jam_Belajar"]
CAT_COLS = ["Organisasi", "Penghasilan_Ortu", "Jenis_Kelamin", "Status_Beasiswa"]

# =========================================================
# 2. PREPROCESSING
# =========================================================

# 2a. Handle missing value
#   - Numerik  -> isi dengan median (robust terhadap outlier)
#   - Kategorik -> isi dengan modus (nilai paling sering muncul)
for col in NUM_COLS:
    df[col] = df[col].fillna(df[col].median())

for col in CAT_COLS:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing value setelah imputasi:\n", df.isna().sum().sum(), "(harus 0)")

# 2b. Encoding
#   - Fitur kategorik -> Label Encoding (semua biner/ordinal sederhana,
#     untuk kategori nominal >2 level seperti Penghasilan_Ortu tetap
#     dipakai LabelEncoder demi kesederhanaan pipeline; opsi lain: One-Hot)
encoders = {}
df_enc = df.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

target_le = LabelEncoder()
df_enc[TARGET] = target_le.fit_transform(df_enc[TARGET])  # Tidak=0, Ya=1 (cek urutan di bawah)
print("\nMapping target:", dict(zip(target_le.classes_, target_le.transform(target_le.classes_))))

FEATURES = NUM_COLS + CAT_COLS
X = df_enc[FEATURES]
y = df_enc[TARGET]

# 2c. Train-test split 80:20 (stratify supaya proporsi kelas terjaga)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")

# =========================================================
# 3. MODELING
# =========================================================
models = {
    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "Decision Tree (max_depth=None)": DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
    "Gaussian Naive Bayes": GaussianNB(),
}

fitted = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    predictions[name] = model.predict(X_test)

# =========================================================
# 4. EVALUASI
# =========================================================
results = []
for name, y_pred in predictions.items():
    results.append({
        "Model": name,
        "Akurasi": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).set_index("Model")
print("\n=== TABEL PERBANDINGAN METRIK ===")
print(results_df.round(3))
results_df.round(4).to_csv("hasil_metrik.csv")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name, fontsize=10)
    classes = target_le.classes_
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("\nConfusion matrix disimpan -> confusion_matrices.png")

for name, y_pred in predictions.items():
    print(f"\n--- Classification report: {name} ---")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# =========================================================
# 5. VISUALISASI
# =========================================================

# 5a. Plot pohon keputusan (max_depth=3)
plt.figure(figsize=(20, 10))
plot_tree(
    fitted["Decision Tree (max_depth=3)"],
    feature_names=FEATURES,
    class_names=target_le.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree (max_depth=3) - Prediksi Lulus Tepat Waktu")
plt.savefig("decision_tree_depth3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Pohon keputusan disimpan -> decision_tree_depth3.png")

# 5b. Bar chart perbandingan metrik 3 model
metrics = ["Akurasi", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#2F3C7E", "#F96167", "#97BC62"]
for i, (name, row) in enumerate(results_df.iterrows()):
    ax.bar(x + (i - 1) * width, row[metrics].values, width, label=name, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Skor")
ax.set_title("Perbandingan Metrik Evaluasi - 3 Model")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
for i, (name, row) in enumerate(results_df.iterrows()):
    for j, m in enumerate(metrics):
        ax.text(j + (i - 1) * width, row[m] + 0.02, f"{row[m]:.2f}",
                ha="center", fontsize=7)
plt.tight_layout()
plt.savefig("perbandingan_metrik.png", dpi=150)
plt.close()
print("Bar chart perbandingan metrik disimpan -> perbandingan_metrik.png")

# 5c. Feature importance (Decision Tree, informatif untuk laporan)
fig, ax = plt.subplots(figsize=(7, 4.5))
importances = fitted["Decision Tree (max_depth=3)"].feature_importances_
order = np.argsort(importances)
ax.barh(np.array(FEATURES)[order], importances[order], color="#2F3C7E")
ax.set_title("Feature Importance - Decision Tree (max_depth=3)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Feature importance disimpan -> feature_importance.png")

# =========================================================
# 6. ANALISIS ERROR (5 data salah prediksi, dari model terbaik & DT dalam)
# =========================================================
# Gunakan Decision Tree (max_depth=None) sebagai contoh karena paling mungkin
# overfit / salah pada data yang "tidak biasa"; kita tampilkan datanya secara utuh.
best_for_error = "Decision Tree (max_depth=None)"
y_pred_err = predictions[best_for_error]

error_mask = (y_pred_err != y_test.values)
error_idx = X_test.index[error_mask][:5]

error_rows = df.loc[error_idx].copy()
error_rows["Prediksi_Model"] = target_le.inverse_transform(
    fitted[best_for_error].predict(X_test.loc[error_idx])
)
error_rows["Aktual"] = df.loc[error_idx, TARGET]

print(f"\n=== 5 CONTOH DATA SALAH PREDIKSI ({best_for_error}) ===")
print(error_rows[NUM_COLS + CAT_COLS + ["Aktual", "Prediksi_Model"]].to_string())
error_rows.to_csv("analisis_error.csv", index=False)

# =========================================================
# 7. SIMULASI (Bagian C.3): IPK tinggi, Kehadiran rendah
# =========================================================
sim = pd.DataFrame([{
    "IPK": 3.8,
    "Kehadiran": 55,
    "Jam_Belajar": df["Jam_Belajar"].median(),
    "Organisasi": encoders["Organisasi"].transform(["Tidak Aktif"])[0],
    "Penghasilan_Ortu": encoders["Penghasilan_Ortu"].transform(["Menengah"])[0],
    "Jenis_Kelamin": encoders["Jenis_Kelamin"].transform(["L"])[0],
    "Status_Beasiswa": encoders["Status_Beasiswa"].transform(["Tidak"])[0],
}])[FEATURES]

print("\n=== SIMULASI: IPK tinggi (3.8), Kehadiran rendah (55%) ===")
for name in ["Decision Tree (max_depth=3)", "Gaussian Naive Bayes"]:
    pred = fitted[name].predict(sim)[0]
    proba = fitted[name].predict_proba(sim)[0]
    label = target_le.inverse_transform([pred])[0]
    print(f"{name}: Prediksi = {label} | Prob(Ya)={proba[1]:.3f} Prob(Tidak)={proba[0]:.3f}")

print("\nSELESAI. Semua output (CSV & PNG) tersimpan di folder kerja.")
