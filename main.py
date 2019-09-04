import argparse
import torch
import numpy as np
import pandas as pd
import lib
from global_def import discrete_features, CUSTOMER_ID
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("-seed", type=int, default=7, help="Seed for random initialization")
parser.add_argument('--raw_data', default='data/data.csv', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    print("Loading train data from {}".format(args.raw_data))
    df = pd.read_csv(args.raw_data)
    df = df.drop(CUSTOMER_ID, 1)

    print(df.shape)
    df = df[~(df.TotalCharges == " ")]  # delete empty string
    df.TotalCharges = pd.to_numeric(df.TotalCharges)

    df, dict_label_encoders = lib.normalize_data(df, discrete_features)

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=args.seed)
    # print(df_train.shape)
    # print(df_test.shape)

    scaler = StandardScaler()

    df_train_sc = scaler.fit_transform(df_train)
    df_test_sc = scaler.transform(df_test)

    print(df_train_sc.shape)
    print(df_test_sc.shape)
    print(type(df_train_sc))
    print(type(df_test_sc))
    # print(df_train_sc[:10])
    # print(df_test_sc[:10])


if __name__ == '__main__':
    main()
