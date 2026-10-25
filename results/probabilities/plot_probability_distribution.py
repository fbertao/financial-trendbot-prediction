import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ============================================================
# Load predicted probabilities
# ============================================================

probabilities = {
    "DNN": np.load("results/probabilities/dnn_prob.npy"),
    "Logistic Regression": np.load(
        "results/probabilities/logistic_regression_prob.npy"
    ),
    "Random Forest": np.load(
        "results/probabilities/random_forest_prob.npy"
    ),
    "XGBoost": np.load(
        "results/probabilities/xgboost_prob.npy"
    ),
    "GRU": np.load(
        "results/probabilities/gru_prob.npy"
    ),
    "LSTM": np.load(
        "results/probabilities/lstm_prob.npy"
    )
}


# ============================================================
# Figure configuration
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 7),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================
# Plot distributions
# ============================================================

for ax, (name, prob) in zip(
    axes,
    probabilities.items()
):

    prob = np.asarray(prob).flatten()

    # Histogram
    ax.hist(
        prob,
        bins=25,
        density=True,
        alpha=0.65
    )

    # KDE
    kde = gaussian_kde(prob)

    x = np.linspace(0, 1, 500)

    ax.plot(
        x,
        kde(x),
        linewidth=1.5
    )

    # Mean
    mean_prob = np.mean(prob)

    ax.axvline(
        mean_prob,
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {mean_prob:.2f}"
    )

    # Thresholds
    for threshold in thresholds:
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=0.7,
            alpha=0.35
        )

    # Titles
    ax.set_title(
        name,
        fontsize=12
    )

    # X limits
    ax.set_xlim(0, 1)

    # Remove unnecessary grid
    ax.grid(
        False
    )

    # Legend only for mean
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True
    )


# ============================================================
# Common labels
# ============================================================

fig.supxlabel(
    "Predicted Probability of Up",
    fontsize=11
)

fig.supylabel(
    "Density",
    fontsize=11
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout()

plt.subplots_adjust(
    bottom=0.10,
    left=0.08
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "results/probabilities/probability_distribution_models.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()