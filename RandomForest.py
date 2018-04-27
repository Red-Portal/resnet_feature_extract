
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
import xgboost as xgb
import numpy as np
import argparse

def main():
    train_x = np.load("train_fmap.npy")
    train_y = np.load("train_labels.npy")

    val_x = np.load("val_fmap.npy")
    val_y = np.load("val_labels.npy")

    train_mat = xgb.DMatrix(train_x, label=train_y)
    val_mat = xgb.DMatrix(val_x, label=val_y)

    forest = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=10,
        oob_score=False,
        n_jobs=6,
        verbose=1)

    forest.fit(train_x, train_y)

    predictions = forest.predict_proba(train_x)
    pred_y = np.asarray([np.argmax(line) for line in predictions])
    print("precision", sk.metrics.classification_report(val_y, pred_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line parameters")
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--lambda_reg', type=float, default=3.0)
    parser.add_argument('--subsample', type=float, default=0.1)
    args = parser.parse_args()
    main()
