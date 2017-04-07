from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print type(y_digits)
svc = svm.SVC(C=1, kernel='linear')

X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()

# for k in range(3):
# 	X_train = list(X_folds)
# 	X_test  = X_train.pop(k)
# 	X_train = np.concatenate(X_train)
# 	y_train = list(y_folds)
# 	y_test  = y_train.pop(k)
# 	y_train = np.concatenate(y_train)
# 	scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

# print scores