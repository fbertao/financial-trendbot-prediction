import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

# model random forest
model_RF = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    # max_depth=10,
    random_state=42
)

# train
model_RF.fit(X_train, y_train)

# predictions
predictions = model_RF.predict(X_test)

# probabilities
predictions_prob = model_RF.predict_proba(X_test)

# metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions_prob[:,1])

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not Up", "Up"]
    )
)

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Up", "Up"]
)

disp.plot(cmap="Blues")

plt.title("Random Forest - COnfusion Matrix")

# plt.savefig(
    #     "results/random_forest_confusion_matrix.png",
    #     dpi=300,
    #     bbox_inches="tight"
# )

plt.show()

fpr, tpr, _ = roc_curve(
    y_test,
    predictions_prob[:,1]
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"roc (auc = {auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    "--"

)

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("Random Forest - ROC curve")
plt.legend()

# plt.savefig(
#   "results/random_forest_roc_curve.png",
#    dpi=300,
#    bbox_inches="tight"
# )

plt.show()

# importances features

importance = model_RF.feature_importances_

features = [
    f"Day {i}"
    for i in range (1, 21)
]

importance_df = pd.DataFrame({
    "Features":features,
    "Importance":importance
})

importance_df = importance_df.sort_values(
    by="Importance", 
    ascending=False
)

plt.figure(figsize=(8,6))

plt.barh(
    importance_df["Features"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Random Forest - Feature Importances")
plt.xlabel("Importance")

# plt.savefig(
#     "results/random_forest_feature_importances.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()

# threshold analysis

thresholds = [0.5,0.6,0.7,0.8,0.9]

results=[]

prob_yes = predictions_prob[:,1]

for threshold in thresholds:

    pred=(prob_yes>=threshold).astype(int)

    operations=np.sum(pred)

    gain=np.sum(
        (pred==1)&
        (y_test==1)
    )

    hit_rate=np.nan

    if operations>0:
        hit_rate=gain/operations

    results.append({

        "Threshold":threshold,
        "Operations":operations,
        "Hit Rate":hit_rate

    })

df_results=pd.DataFrame(results)

print(df_results)

# hit rate vs threshold

plt.figure(figsize=(7,5))

plt.plot(
    df_results["Threshold"],
    df_results["Hit Rate"],
    marker="o"
)

plt.grid()

plt.xlabel("Confidence Threshold")
plt.ylabel("Hit Rate")

plt.title("Random Forest - Hit Rate")

# plt.savefig(
#    "results/random_forest_hit_rate.png",
#    dpi=300,
#    bbox_inches="tight"
# )

plt.show()

# operations vs threshold

plt.figure(figsize=(7,5))

plt.plot(
    df_results["Threshold"],
    df_results["Operations"],
    marker="o"
)

plt.grid()

plt.xlabel("Confidence Threshold")
plt.ylabel("Operations")

plt.title("Random Forest - Operations")

# plt.savefig(
#     "results/random_forest_operations.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()