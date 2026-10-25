# XGBoost

algoritmo baseado em árvores de decisão. não constroi elas independentes como na random forest, e sim uma árvore de cada vez, tentando corrigir os erros da árvore anterior


![XGBoost](results/xgboost.png)

# Hiperparâmetros

1. n_estimators
    número de árvores: mais árvores, mais capacidade de aprender, mais tempo de treinamento 
    risco: árvores demais -> overfitting

    n_estimators=100 -> valor padrão (testar outros valores, como 50, 100, 200)
2. learning_rate
    responde: "quanto cada árvore pode corrigir?"
    learning_rate=0.1 -> corrige só 10%, a próxima árvore corrige mais e vai seguindo o ciclo até convergir 

3. max_depth
    controla a profundidade das árvores
    árvores pequenas: menos overfitting
    árvores muito grandes: aprende tudo -> overfitting

4. subsample
    ex: subsample=0.8 -> cada árvore enxerga 80% dos dados
    isso reduz o overfiitng

5. colsample_bytree
    mesmo conceito mas por colunas, isso gera diversidade 

6. eval_metric
    problema de probabilidades: eval_metric="logloss"

