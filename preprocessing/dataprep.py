import duckdb

# since I had already used some tickers in the analyses, I did a simple cleanup here to avoid duplicates.

tickers_excluir = [
    'asle', 'asln', 'asmb', 'asnd',
    'aso', 'aspa', 'aspau', 'asrt'
]

tickers_sql = ", ".join(
    f"'{t}'" for t in tickers_excluir
    )

query = f"""
    select *
    from 'data/dataset.csv'
    where lower(Ticker) not in ({tickers_sql})
"""
df_filtered = duckdb.query(query).to_df()

df_filtered.to_csv("tickers_filtered.csv", index=False)

print("ok")
