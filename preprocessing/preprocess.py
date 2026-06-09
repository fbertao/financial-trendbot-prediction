import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

def convert_to_numeric(value): 
    if isinstance(value, str):
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

def prepare_data():

    ds = pd.read_csv(
        "data/dataset.csv"
    )

    # convert to float
    for col in ds.columns[:-1]:

        ds[col] = ds[col].apply(
            convert_to_numeric
        )

    colunas_num = ds.select_dtypes(
        include=[np.number]
    ).columns

    # percentual returns
    ds[colunas_num] = (
        ds[colunas_num]
        .pct_change()
        * 100
    )

    ds = ds.dropna().reset_index(
        drop=True
    )

    # clipping
    ds[colunas_num] = ds[
        colunas_num
    ].clip(-100,100)

    # X e y
    X = ds.iloc[:,:-1].values
    y = ds.iloc[:,-1].values

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    scaler = RobustScaler()

    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val
    )