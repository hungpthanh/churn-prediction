from sklearn.preprocessing import LabelEncoder


def normalize_data(df, features):
    dict_label_encoder = {}
    for feature in features:
        dict_label_encoder[feature] = LabelEncoder()
        df[feature] = dict_label_encoder[feature].fit_transform(df[feature])
    return df, dict_label_encoder

