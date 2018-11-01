import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True))
print(pd.qcut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True))
