import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from preprocessing.preprocess import prepare_data

# same pipeline as DNN
X_train, X_test, y_train, y_test = prepare_data()

# model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# training
model.fit(X_train, y_train)

# predicitons
prediction = model.predict(X_test)

# probabilities (for roc-auc)
prediction_prob = model.predict_proba(X_test)

# metrics

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction_prob[:,1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        prediction,
        target_names=["Not Up", "Up"]
    )
)