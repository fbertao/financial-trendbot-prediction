import pandas as pd
import numpy as np
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

# confusion matrix

cm = confusion_matrix(y_test, prediction)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Up", "Up"]
)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")

# plt.savefig(
#    "results/confusion_matrix/logistic.png",
#    dpi=300,
#    bbox_inches='tight'
# )

plt.show()

# ROC curve

fpr, tpr, _ = roc_curve(
    y_test,
    prediction_prob[:,1]
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    'k--'
)

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("roc curve - logistic regression")
plt.legend()

# plot.savefig(
#    "results/roc_curve/logistic.png",
#    dpi=300,
#    bbox_inches='tight'
# )

plt.show()

# features importance

coef = model.coef_[0]

days = [f"day {i}" for i in range(1, 21)]

plt.figure(figsize=(8,6))

plt.barh(days, coef)
plt.title("feature importance - logistic regression")

# plt.savefig(
#    "results/feature_importance/logistic.png",
#    dpi=300,
#    bbox_inches='tight'
# )

plt.show()

# threshold analysis

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

results = []

prob_yes = prediction_prob[:,1]

for threshold in thresholds:
    pred_i = (prob_yes >= threshold).astype(int)

    operations = np.sum(pred_i)

    gain = np.sum(
        (pred_i == 1) &
        (y_test == 1)
    )

    hit_rate = gain/operations if operations > 0 else np.nan

    results.append({
        "threshold": threshold,
        "operations": operations, 
        "hit rate": hit_rate
    })

df_results = pd.DataFrame(results)

plt.figure(figsize=(7,5))
plt.plot(
    df_results["threshold"],
    df_results["hit rate"],
    marker='o'
)

plt.xlabel("confidence threshold")
plt.ylabel("hit rate")
plt.title("threshold analysis")

plt.grid()

# plt.savefig(
#    "results/threshold_analysis/logistic.png",
#    dpi=300,
#    bbox_inches='tight'
# )

plt.show()
