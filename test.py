import pandas as pd
import torch
from sklearn import tree
import lib
from global_def import discrete_features

if __name__ == '__main__':
    df = pd.read_csv("data/data.csv")
    print(df.shape)  # show data's size
    print(df.head()) # show a little data
    print(df.isna().sum())  # check NaN values
    df = df[~(df.TotalCharges == " ")] # delete empty string
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    print(df.describe(include='all'))

    df, dict_label_encoders = lib.normalize_data(df, discrete_features)
    print(df.head())