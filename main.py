import argparse
import torch
import numpy as np
import pandas as pd
import lib
from global_def import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='FFN', type=str)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--input_dim", type=int, default=19)
parser.add_argument("--output_dim", type=int, default=125)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("-seed", type=int, default=7, help="Seed for random initialization")
parser.add_argument('--raw_data', default='data/data.csv', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

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
    df_train_input_sc, df_train_target, df_test_input_sc, df_test_target = lib.clear_data(df, args)

    if args.algo == 'decisiontree':
        dtc = tree.DecisionTreeClassifier(max_depth=6, random_state=42)
        dtc.fit(df_train_input_sc, df_train_target)
        y_pred_dtc = dtc.predict(df_test_input_sc)

    if args.algo == 'FFN':
        model = lib.FFN(df_train_input_sc.shape[1], args.output_dim, args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        dataloader = lib.DataLoader(df_train_input_sc, df_train_target, args.batchsize)

        print(df_train_input_sc.shape)
        # training
        model.train()
        for epoch in range(args.num_epochs):
            sum_loss = 0
            cnt = 0
            for it, (input_data, target_data) in enumerate(dataloader):
                cnt += 1
                input_data = torch.Tensor(input_data)
                target_data = torch.LongTensor(target_data)
                optimizer.zero_grad()
                logit = model(input_data)
                loss = F.nll_loss(logit, target_data)
                pred = logit.data.max(1)[1]
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("Epoch: {} - loss: {}".format(epoch, float(sum_loss) / cnt))

        # testing
        model.eval()
        with torch.no_grad():
            input_data_test = torch.Tensor(df_test_input_sc)
            target_data_test = torch.LongTensor(df_test_target)
            logit = model(input_data_test)
            loss = F.nll_loss(logit, target_data_test)
            y_pred_dtc = logit.data.max(1)[1]

    print(classification_report(df_test_target, y_pred_dtc))


if __name__ == '__main__':
    main()
