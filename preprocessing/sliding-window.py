import pandas as pd
import os


df = pd.read_csv('data/tickers.csv')

# Convert prices to float
df_total['close'] = (
    df_total['close']
    .astype(str)
    .str.replace(',', '.', regex=False)
)

df_total['close'] = pd.to_numeric(
    df_total['close'],
    errors='coerce'
)

# Remove invalid values
df_total = df_total.dropna(subset=['close'])

# Convert dates
df_total['date'] = pd.to_datetime(
    df_total['date'],
    errors='coerce'
)

df_total = df_total.dropna(subset=['date'])

# Sort by ticker and date
df_total = df_total.sort_values(
    ['ticker', 'date']
)

window_size = 20

rows_total = []

# Create windows for each asset.
for ticker in df_total['ticker'].unique():

    df_ticker = df_total[
        df_total['ticker'] == ticker
    ].copy()

    df_ticker = df_ticker.sort_values(
        'date'
    )

    close = df_ticker['close'].tolist()

    for i in range(
            len(close) - window_size):

        # 20 days
        janela = close[
            i:i + window_size
        ]

        # next day
        proximo = close[
            i + window_size
        ]

        # Target class
        subiu = (
            'Yes'
            if proximo > janela[-1]
            else 'No'
        )

        rows_total.append(
            janela + [subiu]
        )

# Create final dataframe
colunas = [
    f'Day {i+1}'
    for i in range(window_size)
]

colunas.append('subiu?')

df_final = pd.DataFrame(
    rows_total,
    columns=colunas
)

# salve dataset
df_final.to_csv(
    'data/dataset.csv',
    index=False
)
print("ok")