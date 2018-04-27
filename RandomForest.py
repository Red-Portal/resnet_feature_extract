
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

    forest = RandomForestClassifier(
        n_estimators=args.trees,
        criterion="entropy",
        max_depth=args.max_depth,
        oob_score=False,
        n_jobs=6,
        verbose=3)

    forest.fit(train_x, train_y)

    predictions = forest.predict_proba(val_x)
    pred_y = np.asarray([np.argmax(line) for line in predictions])
    print("precision", sk.metrics.classification_report(val_y, pred_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line parameters")
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--trees', type=int, default=100)
    args = parser.parse_args()
    main()
