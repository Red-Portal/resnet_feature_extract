import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score

train_x = np.load("train_fmap.npy")
train_y = np.load("train_labels.npy")

val_x = np.load("val_fmap.npy")
val_y = np.load("val_labels.npy")

train_mat = xgb.DMatrix(train_x, label=train_y)
val_mat = xgb.DMatrix(val_x, label=val_y)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'tree_method' : 'gpu_hist',
    'predictor' : 'gpu_predictor',
    'num_class': 10}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

tree = xgb.train(param, train_mat, num_round)

predictions = tree.predict(val_mat)
best_y = np.asarray([np.argmax(line) for line in predictions])
print("precision", precision_score(val_y, pred_y, average='macro'))
