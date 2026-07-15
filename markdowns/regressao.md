# Como funciona a Regressão Logística

Lidamos com um problema binário: 
- Up -> 1;
- Not Up -> 0

A regressão logística vai aprender uma função que estima: 

$P(Y = 1 \mid X)$

Ou seja, a probabilidade do próximo dia ser de alta dado o comportamento dos últimos 20 dias

# Intuição 

Ela recebe as 20 features do sliding window e cálcula a combinação linear: 

$z = w_1x_1 + w_2x_2 + \cdots + w_{20}x_{20} + b$

Depois aplica uma função sigmoide: 

$P(\text{Up}) = \frac{1}{1 + e^{-z}}$

A função sigmoide vai transformar qualquer número em uma probabilidade entre 0 e 1

# Por que usar Regressão Logística?

porque é um modelo: simpes, rápido, interpretável, difícil de overfitar e excelente de baseline

# Como ela aprende

Durante o treinamento vai ajustar os pesos das features (20-days) de forma a minimizar a Binary Cross Entropy pela função: 

$L = -y \log(p) - (1 - y) \log(1 - p)$

# Diferença entre o DNN e Regressão Logística

Regressão logíostica: Input -> sigmoide -> output (só uma camada)

DNN: Input -> Hidden layer -> ... -> Hidden layer -> output (várias camadas)

# Vantagens DNN vs Regressão Logística

**DNN:**
- Consegue aprender relações:
    - não lineares;
    - mais complexas;
    - interações entre features

**Regressão Logística:**
- Mais: 
    - estavél; 
    - interpretável;
    - rápida;
    - menos propensa a overfitting