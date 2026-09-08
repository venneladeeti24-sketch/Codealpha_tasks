"""
Iris Flower Classification using Machine Learning
Dataset: Fisher's Iris Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:\n", df["species"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="linear", random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {results[name] * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=target_names))

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest performing model: {best_model_name}")
print(f"Accuracy: {results[best_model_name] * 100:.2f}%")

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(results)), [v * 100 for v in results.values()])
ax.set_xticks(range(len(results)))
ax.set_xticklabels(results.keys(), rotation=20)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison on Iris Dataset")
ax.set_ylim(80, 101)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

sample = np.array([[5.9, 3.0, 5.1, 1.8]])
prediction = best_model.predict(scaler.transform(sample))
print(f"\nPrediction: {target_names[prediction[0]]}")
