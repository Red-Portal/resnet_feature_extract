import xgboost as xgb
import numpy as np
import argparse
import sklearn.metrics as metrics

def main():
    train_x = np.load("train_fmap.npy")
    train_y = np.load("train_labels.npy")

    val_x = np.load("val_fmap.npy")
    val_y = np.load("val_labels.npy")

    train_mat = xgb.DMatrix(train_x, label=train_y)
    val_mat = xgb.DMatrix(val_x, label=val_y)

    param = {
        'max_depth': args.max_depth,  # the maximum depth of each tree
        'eta': args.eta,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : args.lambda_reg,
        'subsample' : args.subsample,
        'num_class': 10}  # the number of classes that exist in this datset
    num_round = args.rounds  # the number of training iterations

    watchlist = [ (train_mat,'train') ]
    tree = xgb.train(param, train_mat, num_round, watchlist)

    predictions = tree.predict(val_mat)
    pred_y = np.asarray([np.argmax(line) for line in predictions])
    print("precision", metrics.classification_report(val_y, pred_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line parameters")
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--lambda_reg', type=float, default=3.0)
    parser.add_argument('--subsample', type=float, default=0.1)
    args = parser.parse_args()
    main()

