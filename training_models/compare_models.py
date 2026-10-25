import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# brier e log loss
metrics = ["brier score", "log loss"]

dnn = [0.23, 0.66]
lr = [0.24, 0.67]
rf = [0.27, 0.82]
xg = [0.23, 0.66]
gru = [0.23, 0.66]
lstm= [0.23, 0.66]

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

plt.title("Probabilistic Performance (lower is better)")
plt.ylabel("Value")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()

plt.show()