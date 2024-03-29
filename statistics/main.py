import pandas as pd
import torch
from sklearn import tree

if __name__ == '__main__':
    df = pd.read_csv("../data/data.csv")
    print(df.shape)  # show data's size
    print(df.head()) # show a little data
    print(df.isna().sum())  # check NaN values
    df = df[~(df.TotalCharges == " ")] # delete empty string
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    print(df.describe(include='all'))