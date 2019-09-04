from sklearn.preprocessing import LabelEncoder
import pandas as pd


def clear_data(df):
    df = df[~(df.TotalCharges == " ")]  # delete empty string
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    return df


def normalize_data(df, features):
    dict_label_encoder = {}
    for feature in features:
        dict_label_encoder[feature] = LabelEncoder()
        df[feature] = dict_label_encoder[feature].fit_transform(df[feature])
    return df, dict_label_encoder

