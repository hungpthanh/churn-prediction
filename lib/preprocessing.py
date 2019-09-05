from sklearn.preprocessing import LabelEncoder
import pandas as pd
from global_def import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lib
import numpy as np


def clear_data(df, args):
    """
    Slipting data to train and test dataset
    :param df: Pandas Frame
    :param args: argument
    :return: slipted data
    """
    df = df.drop(CUSTOMER_ID, 1)
    df = df[~(df.TotalCharges == " ")]  # delete empty string
    df.TotalCharges = pd.to_numeric(df.TotalCharges) # change value to number
    df, dict_label_encoders = lib.normalize_data(df, discrete_features)
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=args.seed)
    df_train_input = df_train.iloc[:, :-3]  # -1: use all feature, -3: use discrete feature
    df_train_target = np.array(df_train[CHURN])
    df_test_input = df_test.iloc[:, :-3]
    df_test_target = np.array(df_test[CHURN])
    scaler = StandardScaler()
    df_train_input_sc = scaler.fit_transform(df_train_input)
    df_test_input_sc = scaler.transform(df_test_input)
    return df_train_input_sc, df_train_target, df_test_input_sc, df_test_target


def normalize_data(df, features):
    """
    Digitizing data
    :param df: Pandas frame
    :param features: discrete feature
    :return: digitized data
    """
    dict_label_encoder = {}
    for feature in features:
        dict_label_encoder[feature] = LabelEncoder()
        df[feature] = dict_label_encoder[feature].fit_transform(df[feature])
    return df, dict_label_encoder

