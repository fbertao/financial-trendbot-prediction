# Financial TrendBot 

Financial TrendBot stemmed from an initial idea: I need to understand how machine learning works, how to tune its parameters, and how to evaluate it. How can I do this within my own field? That is when I started an Undergraduate Research Fellowship at the Federal University of ABC (UFABC), under the advisement of Prof. Mateus Coelho.

The main focus is the probabilistic forecasting of directional movements (Up/Not Up) using rolling windows of historical prices.
## But what is TrendBot?

TrendBot is a decision support system based on Deep Learning that analyzes 20-day rolling windows of historical prices, generating upward movement probabilities for the next day. This allows for threshold adjustments to manage risk.
The goal is not to predict the market deterministically, but rather to generate probabilistic signals that can be filtered according to the desired risk profile.

# Pipeline Workflow: 
## Data Collection and Structure:
* Dataset: Kaggle - Stock Market Prediction
* Transformation of time series into 20-day rolling windows
* Conversion of prices into percentage changes (ou percentage returns)
* Removal of extreme outliers
* Normalization via RobustScaler 

## Modeling 
### Architecture Used:
* Dense (32) + ReLU
* Batch Normalization
* Dropout (0.2)
* Dense (16) + ReLU
* Batch Normalization
* Dropout (0.2)
* Dense (8) + ReLU
* Output Softmax (2 classes)

Applied Techniques:
* Regularization for overfitting control
* Class weights for dataset balancing
* Early stopping
* Learning rate reduction

## Model Evaluation
### Metrics Used
* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Confusion Matrix

Furthermore, an analysis was conducted across different confidence levels (thresholds), evaluating:
* Reduction in the number of trades
* Increase in precision
* Risk vs. Volume trade-off

# Next Steps
- Model comparison (ou Model benchmarking);
- Recommendation systems using LLMs
