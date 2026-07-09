import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from xgboost import XGBClassifier 
from sklearn.metrics import (
        classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from preprocessing.preprocess import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

# model xgboost

model = XGBClassifier(
    n_estimators=100, #numero maximo de arvores a serem construidas 
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',    
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

prediction_prob = model.predict_proba(X_test)

# metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    prediction_prob[:,1]
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    target_names=["Not Up","Up"]
))

# confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Up","Up"]
)

disp.plot(cmap="Blues")

# curve roc

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
    "--",
    color="gray"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid()

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

# feature importance

importance = model.feature_importances_

features = [
    f"Day {i}"
    for i in range(1,21)
]

importance_df = pd.DataFrame({

    "Feature":features,
    "Importance":importance

}).sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title("XGBoost Feature Importance")

plt.xlabel("Importance")
plt.grid(axis="x")

plt.show()

# operations vs threshold

plt.figure(figsize=(7,5))

plt.plot(
    df_results["threshold"],
    df_results["operations"],
    marker="o"
)

plt.xlabel("Confidence Threshold")
plt.ylabel("Operations")
plt.title("Operations vs Threshold")
plt.grid()

plt.show()