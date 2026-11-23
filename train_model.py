
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Loading dataset my first step 
df = pd.read_csv("dataset/creditcard.csv")

print("Dataset loaded:", df.shape)

# 2. Separate features and target 
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Create models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# 5. Train and evaluate both models
evaluation_results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)

    evaluation_results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-Score:", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# 6. Create folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 7. Save trained Random Forest model
joblib.dump(
    models["Random Forest"],
    "models/fraud_detection_model.pkl"
)

# 8. Save evaluation results
with open("outputs/model_evaluation.json", "w") as file:
    json.dump(evaluation_results, file, indent=4)

print("\n--------------------------------")
print("Model training completed!")
print("Random Forest model saved.")
print("Evaluation results saved.")
print("--------------------------------")


