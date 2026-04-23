"""
TASK 1: Iris Flower Classification
Data Science Intern Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  TASK 1: IRIS FLOWER CLASSIFICATION")
print("  Data Science Intern Report")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df["target"] = iris.target

print("\n[1] DATASET OVERVIEW")
print(f"  Shape: {df.shape}")
print(f"  Classes: {list(iris.target_names)}")
print(f"  Class distribution:\n{df['species'].value_counts().to_string()}")
print(f"\n  First 5 rows:\n{df.head().to_string()}")
print(f"\n  Statistical summary:\n{df.describe().to_string()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Task 1 — Iris Flower Classification: EDA", fontsize=16, fontweight="bold")

features = iris.feature_names
colors = ["#E74C3C", "#2ECC71", "#3498DB"]
species_list = iris.target_names

for idx, feature in enumerate(features[:3]):
    ax = axes[0, idx]
    for i, sp in enumerate(species_list):
        subset = df[df["species"] == sp][feature]
        ax.hist(subset, alpha=0.6, color=colors[i], label=sp, bins=15)
    ax.set_title(feature.replace(" (cm)", ""), fontsize=10)
    ax.set_xlabel("cm")
    ax.legend(fontsize=7)

ax = axes[1, 0]
corr = df[features].corr()
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
short = ["SL", "SW", "PL", "PW"]
ax.set_xticklabels(short, fontsize=8)
ax.set_yticklabels(short, fontsize=8)
ax.set_title("Feature Correlation")
plt.colorbar(im, ax=ax)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

ax = axes[1, 1]
for i, sp in enumerate(species_list):
    sub = df[df["species"] == sp]
    ax.scatter(sub["petal length (cm)"], sub["petal width (cm)"],
               c=colors[i], label=sp, alpha=0.7, s=40)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal: Length vs Width")
ax.legend(fontsize=8)

ax = axes[1, 2]
df.boxplot(column="sepal length (cm)", by="species", ax=ax,
           patch_artist=True)
ax.set_title("Sepal Length by Species")
ax.set_xlabel("Species")
plt.sca(ax)
plt.title("Sepal Length Distribution")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/task1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[2] EDA charts saved.")

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

print(f"\n[3] MODEL EVALUATION — Random Forest")
print(f"  Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"  CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Task 1 — Model Performance", fontsize=14, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(iris.target_names, rotation=15)
ax.set_yticklabels(iris.target_names)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else "black")

ax = axes[1]
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
bars = ax.bar(range(4), importances[idx], color=["#3498DB","#E74C3C","#2ECC71","#F39C12"])
ax.set_xticks(range(4))
ax.set_xticklabels([features[i].replace(" (cm)","") for i in idx], rotation=15)
ax.set_title("Feature Importances")
ax.set_ylabel("Importance Score")
for bar, val in zip(bars, importances[idx]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/task1_model.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n[4] KEY INSIGHTS:")
top_feat = features[np.argmax(model.feature_importances_)]
print(f"  • Top feature: {top_feat}")
print(f"  • Model achieved {acc*100:.1f}% accuracy on unseen test data")
print(f"  • 5-fold CV confirms robust performance: {cv_scores.mean()*100:.1f}%")
print("  • Setosa is perfectly linearly separable from the other two species")
print("  • Petal dimensions are far more discriminative than sepal dimensions")
print("\n  Task 1 COMPLETE ✓")
