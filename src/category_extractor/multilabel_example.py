import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])

y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
                ["new york"],["london"],["london"],["london"],["london"],
                ["london"],["london"],["new york","london"],["new york","london"]]

X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too'])
target_names = ['New York', 'London']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LogisticRegression()))])

X_folds = np.array_split(X_train, 3)
y_folds = np.array_split(Y, 3)

scores = list()
for k in range(2):
  X_train_fold = list(X_folds)
  X_test_fold  = X_train_fold.pop(k)
  X_train_fold = np.concatenate(X_train_fold)
  y_train_fold = list(y_folds)
  y_test_fold  = y_train_fold.pop(k)
  y_train_fold = np.concatenate(y_train_fold)
  print "\n\n\n"
  print X_train_fold, y_train_fold
  print X_test_fold, y_test_fold

  train = classifier.fit(X_train_fold, y_train_fold)
  
  scores.append(train.score(X_test_fold, y_test_fold))

  print "Predict"
  predicted = classifier.predict(X_test_fold)
  print predicted
  # all_labels = mlb.inverse_transform(predicted)
  # for item, labels in zip(X_test_fold, all_labels):
  #     print('{0} => {1}'.format(item, ', '.join(labels)))

  print classification_report(y_test_fold, predicted, target_names=target_names)
  print f1_score(y_test_fold, predicted, average='micro')

# classifier.fit(X_train, Y)
# predicted = classifier.predict(X_test)
# all_labels = mlb.inverse_transform(predicted)

# for item, labels in zip(X_test, all_labels):
#     print('{0} => {1}'.format(item, ', '.join(labels)))