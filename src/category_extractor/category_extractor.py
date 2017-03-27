from sklearn import datasets, linear_model

dataset = datasets.load_digits()

logreg = linear_model.LogisticRegression()
model = logreg.fit(dataset.data, dataset.target)

expected = dataset.target
predicted = model.predict(dataset.data)

print metrics.classification_report(expected, predicted)