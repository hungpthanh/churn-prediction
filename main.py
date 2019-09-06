import argparse
import torch
import numpy as np
import pandas as pd
import lib
import torch.optim as optim
import torch.nn.functional as F
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='logisticregression', type=str, help="choosing algorithm: FFN, decisiontree, randomforest, logisticregression")
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--input_dim", type=int, default=19)
parser.add_argument("--output_dim", type=int, default=125)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--all_feature", type=int, default=1, help="using all fearure 1, only using discrete feature 0")
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


def finding_parameter():
    """
    Finding parameter for machine learning model
    :return: parameter for traditional machine learning algorithm
    """
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    df = pd.read_csv(args.raw_data)
    df_train_input_sc, df_train_target, df_test_input_sc, df_test_target = lib.clear_data(df, args)

    if args.algo == 'decisiontree':
        clf = tree.DecisionTreeClassifier()
        param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": [0.05, 0.1, 1],
                      "criterion": ["gini", "entropy"],
                      "splitter": ["best", "random"],
                      "class_weight": ["balanced", None]
                       }

    if args.algo == 'randomforest':
        print("Finding parameter for random forest")
        # build a classifier
        clf = RandomForestClassifier(n_estimators=1000)
        param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        # Utility function to report best scores

    if args.algo == "logisticregression":
        print("Finding parameter for logisticregression")
        clf = LogisticRegression()
        param_dist = {
            "penalty": ["l1", "l2"],
            "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            "C": [0.05, 0.1],
            "fit_intercept": [True, False],
            "intercept_scaling": [0.01, 0.1, 1],
            "max_iter": [10, 100, 1000]
        }

        # specify parameters and distributions to sample from

    # run randomized search
    n_iter_search = 10000
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, iid=False)

    start = time()
    random_search.fit(df_train_input_sc, df_train_target)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def main():
    print("Loading train data from {}".format(args.raw_data))
    df = pd.read_csv(args.raw_data)
    df_train_input_sc, df_train_target, df_test_input_sc, df_test_target = lib.clear_data(df, args)

    if args.algo == 'decisiontree':
        # min_samples_leaf: 0.05, min_samples_split: 10, class_weight: None, splitter: best, max_features: 10
        # criterion: entropy, max_depth: 7 for all feature

        # min_samples_leaf: 0.05, min_samples_split: 3, class_weight: None, splitter: best, max_features: 8
        # criterion: entropy, max_depth: 6 for discrete feature

        model = tree.DecisionTreeClassifier(min_samples_leaf=0.05,
                                            min_samples_split=3,
                                            class_weight=None,
                                            splitter="best",
                                            max_features=8,
                                            criterion="entropy",
                                            max_depth=6)
        model.fit(df_train_input_sc, df_train_target)
        y_pred = model.predict(df_test_input_sc)

    if args.algo == 'randomforest':
        #  random_state: 42, n_estimators: 1000, criterion: gini, max_depth: 7, bootstrap: True, max_features: 5,
        #  min_samples_leaf: 7, min_samples_split: 7 for all feature

        #  random_state: 42, n_estimators: 100, criterion: gini, max_depth: 7, bootstrap: True, max_features: 5,
        #  min_samples_leaf: 7, min_samples_split: 7 for discrete feature

        model = RandomForestClassifier(random_state=42,  # pafam for using all feature
                                       n_estimators=1000,
                                       criterion="gini",
                                       max_depth=7,
                                       bootstrap=True,
                                       max_features=5,
                                       min_samples_leaf=7,
                                       min_samples_split=7)

        model.fit(df_train_input_sc, df_train_target)
        y_pred = model.predict(df_test_input_sc)

    if args.algo == 'logisticregression':
        # penalty: l1, random_state: 42, C: 0.05, tol: 0.01, intercept_scaling: 3, fit_intercept: True,
        # max_iter: 10 for all feature

        # penalty: l2, random_state: 42, C: 0.05, tol: 0.1, intercept_scaling: 1, fit_intercept: True,
        # max_iter: 10 for discrete feature

        model = LogisticRegression(penalty="l1",
                                   random_state=42,
                                   C=.05,
                                   tol=0.01,
                                   intercept_scaling=3,
                                   fit_intercept=True,
                                   max_iter=10)

        model.fit(df_train_input_sc, df_train_target)
        y_pred = model.predict(df_test_input_sc)

    if args.algo == 'FFN':
        model = lib.FFN(df_train_input_sc.shape[1], args.output_dim, args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        dataloader = lib.DataLoader(df_train_input_sc, df_train_target, args.batchsize)

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
            y_pred = logit.data.max(1)[1]

    print(classification_report(df_test_target, y_pred))


if __name__ == '__main__':
    main()
    # finding_parameter()