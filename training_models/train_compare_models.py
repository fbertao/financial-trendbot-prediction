import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

metrics = ["accuracy", "precision", "recall", "f1", "auc"]

dnn = [0.58, 0.54, 0.25, 0.34, 0.58]
lr = [0.58, 0.52, 0.38, 0.44, 0.56]
rf = [0.57, 0.51, 0.31, 0.38, 0.53]
xg = [0.60, 0.61, 0.21, 0.31, 0.56]
gru = [0.60, 0.65, 0.18, 0.28, 0.59]
lstm= [0.61, 0.64, 0.19, 0.30, 0.56]

df = pd.DataFrame({
    "Metric": metrics,
    "DNN": dnn,
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xg,
    "GRU": gru,
    "LSTM": lstm
})

ax = df.plot(
    x="Metric",
    y=["DNN", "Logistic Regression", "Random Forest", "XGBoost", "GRU", "LSTM"],
    kind="bar",
    figsize=(8,5)
)

plt.title("DNN vs LR vs RF vs XGBoost vs GRU vs LSTM")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()

plt.show()