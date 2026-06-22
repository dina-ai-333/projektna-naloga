import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

files = [
    "ok_dataset.csv",
    "peace_dataset.csv",
    "surfer_dataset.csv",
    "alien_dataset.csv",
    "korean_heart_dataset.csv"
]

df_list = []
for f in files:
    df_list.append(pd.read_csv(os.path.join(BASE, "dataset", f), header=None))

df = pd.concat(df_list)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

#normalizacija
def normalize_landmarks(X):
    X_norm = []

    for row in X:
        row = np.array(row).reshape(21, 3)

        # centriranje glede na zapestje (landmark 0)
        row = row - row[0]

        #velikost roke
        scale = np.max(np.linalg.norm(row, axis=1))
        #zaščita pred deljenjem z 0
        if scale > 0:
            row = row / scale

        X_norm.append(row.flatten())

    return np.array(X_norm)

X = normalize_landmarks(X)

# testna množica (končna evalvacija modela)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# učna + validacijska množica
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.2,
    random_state=42,
    stratify=y_train_val
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print("\nVALIDATION accuracy:", accuracy_score(y_val, val_pred))

y_pred = model.predict(X_test)

print("\nTEST Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

MODEL_PATH = os.path.join(BASE, "modeli", "gesture_model.pkl")
joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)

results = {
    "validation_accuracy": float(accuracy_score(y_val, val_pred)),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "report": classification_report(y_test, y_pred, output_dict=True)
}

RESULT_PATH = os.path.join(BASE, "rezultati", "results.json")

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to:", RESULT_PATH)

report = classification_report(y_test, y_pred, output_dict=True)

labels = list(report.keys())[:-3]

f1_scores = [report[label]["f1-score"] for label in labels]

plt.figure(figsize=(8,5))
plt.bar(labels, f1_scores)
plt.title("F1-score per gesture")
plt.ylabel("F1-score")
plt.xticks(rotation=30)

plt.tight_layout()

GRAPH_PATH = os.path.join(BASE, "rezultati", "f1_scores.png")
plt.savefig(GRAPH_PATH)

print("Graph saved to:", GRAPH_PATH)