import numpy as np
import pandas as pd


def ensure_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    else:
        try:
            data_pd = pd.DataFrame(data)
            return data_pd
        except Exception as e:
            raise ValueError(e)


def OneHotEncoder(data, columns):
    df = ensure_dataframe(data)
    encoded_df = df.copy()
    for col in columns:
        encoded_df = pd.get_dummies(encoded_df, columns=[col], prefix=str(col))
    return encoded_df


def OrdinalEncoder(data, columns):
    df = ensure_dataframe(data)
    encoded_df = df.copy()
    for col in columns:
        unique_vals = encoded_df[col].unique()
        mapping = {value: index + 1 for index, value in enumerate(unique_vals)}
        encoded_df[f"{col}_encoded"] = encoded_df[col].map(mapping)
    return encoded_df


data1 = {"Letters": ["A", "B", "C", "D"], "Colours": ["Red", "Blue", "Green", "Blue"]}
df1 = pd.DataFrame(data1)
encoded_df1 = OneHotEncoder(df1, ["Letters", "Colours"])
print("One Hot Encoded Data:\n")
print(encoded_df1)

encoded_df2 = OrdinalEncoder(df1, ["Letters", "Colours"])
print("Ordinal Encoded Data:\n")
print(encoded_df2)
