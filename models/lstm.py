import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
from preprocessing.preprocess import prepare_data

# prepare data
X_train, X_test, y_train, y_test = prepare_data()

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)
X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

# model

model = Sequential([
    LSTM(
        units=32,
        input_shape=(X_train.shape[1],X_train.shape[2]),
        return_sequences=False,
        recurrent_dropout=0.2
    ),
    BatchNormalization(),
    Dropout(0.2),
    Dense(
        16,
        activation="relu"
    ),
    Dense(
        2,
        activation="softmax"
    )
])

# compilation
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=40,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=20,
    min_lr=1e-6
)

# training
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=200,
    batch_size=16,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# accuracy/loss
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs,acc,label="Train")
plt.plot(epochs,val_acc,label="Validation")
plt.title("Training Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs,loss,label="Train")
plt.plot(epochs,val_loss,label="Validation")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.show()

# predictions
prediction_prob = model.predict(X_test)

predictions = np.argmax(
    prediction_prob,
    axis=1
)

# metrics
accuracy = accuracy_score(y_test,predictions)

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

print()

print(classification_report(
    y_test,
    predictions,
    target_names=[
        "Not Up",
        "Up"
    ]
))

# confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Up",
        "Up"
    ]
)

disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# roc
fpr,tpr,_ = roc_curve(
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
    "k--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# threshold analysis
thresholds = [0.5,0.6,0.7,0.8,0.9]
results = []

prob_yes = prediction_prob[:,1]
for threshold in thresholds:
    pred_i = (
        prob_yes >= threshold
    ).astype(int)
    operations = np.sum(pred_i)
    gain = np.sum(
        (pred_i==1)
        &
        (y_test==1)
    )

    hit_rate = (
        gain/operations
        if operations>0
        else np.nan
    )

    results.append({
        "Threshold":threshold,
        "Operations":operations,
        "Hit Rate":hit_rate

    })

df_results = pd.DataFrame(results)

print(df_results)

# hit rate vs threshold
plt.figure(figsize=(7,5))

plt.plot(
    df_results["Threshold"],
    df_results["Hit Rate"],
    marker="o"
)

plt.xlabel("Confidence Threshold")
plt.ylabel("Hit Rate")
plt.title("Threshold Analysis")
plt.grid()
plt.show()

# operations vs threshold
plt.figure(figsize=(7,5))

plt.plot(
    df_results["Threshold"],
    df_results["Operations"],
    marker="o"
)

plt.xlabel("Confidence Threshold")
plt.ylabel("Operations")
plt.title("Operations vs Threshold")
plt.grid()
plt.show()

