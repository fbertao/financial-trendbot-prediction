from preprocessing.preprocess import prepare_data
from models.dnn import create_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import os 
os.makedirs('results', exist_ok=True)

X_train, X_val, y_train, y_val = (
    prepare_data()
)

model = create_model(
    X_train.shape[1]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=200,
    batch_size=16
)

# training curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train')
plt.plot(epochs_range, val_acc, label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train')
plt.plot(epochs_range, val_loss, label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig(
    "results/training_curve.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# predictions

y_pred_prob = model.predict(X_val)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

# metrics

precision = precision_score(
    y_val, y_pred
)
recall = recall_score(
    y_val, y_pred
)
f1 = f1_score(
    y_val, y_pred
)
accuracy = np.mean(
    y_val == y_pred
)   
roc_auc = roc_auc_score(
    y_val, y_pred_prob[:,1]
)

# table of metrics

metrics_df = pd.DataFrame(
    {
        'Metric': [
            'Precision',
            'Recall',
            'F1-Score',     
            'Accuracy',
            'ROC AUC'
        ],
        'Value': [
            precision,
            recall,
            f1,
            accuracy,
            roc_auc
        ]
    }
)   

metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)

# classification report

print(
    classification_report(
        y_val, y_pred,
        target_names=['Not Up', 'Up']
    )
)

# confusion matrix

cm = confusion_matrix(
    y_val, y_pred
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Up', 'Up']
)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_pred_prob[:,1])
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# threshold analysis

prob_sim = y_pred_prob[:,1]

limits = [0.5, 0.6, 0.7, 0.8, 0.9
]

results = []

for limit in limits:

    pred_i = (
        prob_sim >= limit
    ).astype(int)

    operation = np.sum(
        pred_i
    )

    gain = np.sum(
        (pred_i == 1)
        &
        (y_val == 1)
    )

    if operation > 0:

        hit_rate = (
            gain
            /
            operation
        )

    else:

        hit_rate = np.nan

    results.append({

        "Threshold":
        limit,

        "Operations":
        operation,

        "Hit Rate":
        hit_rate

    })

df_results.to_csv(
    "results/threshold_analysis.csv",
    index=False
)

print(df_results)

model.save(
    "results/dnn_model.keras"
)