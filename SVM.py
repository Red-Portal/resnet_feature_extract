
import sklearn as sk
import numpy as np
import thundersvmScikit as svm

def main():
    train_x = np.load("train_fmap.npy")
    train_y = np.load("train_labels.npy")

    val_x = np.load("val_fmap.npy")
    val_y = np.load("val_labels.npy")

    forest = svm.SVC(
        kernel=args.kernel,
        degree=args.poly_degree,
        probability=args.probability,
        max_iter=args.max_iter,
        verbose=True)

    forest.fit(train_x, train_y)

    pred_y = forest.predict(val_x)
    #pred_y = np.asarray([np.argmax(line) for line in predictions])
    print("precision", sk.metrics.classification_report(val_y, pred_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line parameters")
    parser.add_argument('--max-iter', type=int, default=1)
    parser.add_argument('--probability', type=bool, default=True)
    parser.add_argument('--kernel', type=str, default="rbf")
    parser.add_argument('--poly-degree', type=int, default=3)
    parser.add_argument('--penalty', type=float, default=1.0)
    args = parser.parse_args()
    main()
