from feature_extractor import FeatureExtractor
from item_selector import ItemSelector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score
from nltk.util import skipgrams
import numpy as np
import csv, re, pickle, functools

class SentimentExtractor:
	def __init__(self, filename):
		self.categories = ['food', 'service', 'price', 'place']
		self.target_names = ['positive', 'negative', 'neutral']
		self.train_data = []
		self.train_target = []
		train_data, train_target = self.read_data(filename)
		for i in range(len(self.categories)):
			self.train_data.append(np.array(train_data[i]))
			self.train_target.append(np.array(train_target[i]))

	def read_data(self, filename):
		data = [[], [], [], []]
		targets = [[], [], [], []]
		regex = re.compile('[^0-9a-zA-Z]+')
		with open (filename, "rb") as f:
			reader = csv.reader(f, delimiter=';', quotechar='"')
			next(reader)
			for row in reader:
				for i in range(1, len(self.categories) + 1):
					if (row[i] != "-"):
						data[i-1].append(regex.sub(' ', row[0]))
						targets[i-1].append(self.target_names.index(row[i]))

		return data, targets

	def get_pipeline(self, index):
		skipper_2 = functools.partial(skipgrams, n=2, k=2)
		vocabulary_skipper_2 = self.get_skip_bigrams_vocabulary(2, 5, index)
		# skipper_3 = functools.partial(skipgrams, n=2, k=3)
		# vocabulary_skipper_3 = self.get_skip_bigrams_vocabulary(3, 5, index)
		# skipper_4 = functools.partial(skipgrams, n=2, k=4)
		# vocabulary_skipper_4 = self.get_skip_bigrams_vocabulary(4, 5, index)
		# skipper_5 = functools.partial(skipgrams, n=2, k=5)
		# vocabulary_skipper_5 = self.get_skip_bigrams_vocabulary(5, 5, index)

		return Pipeline([
            ('data', FeatureExtractor()),

            ('features', FeatureUnion(
                transformer_list=[

                    ('bag_of_ngram', Pipeline([
                        ('selector', ItemSelector(key='sentence')),
                        ('ngram', CountVectorizer(ngram_range=(1, 5))),
                    ])),

                    # ('bag_of_headword', Pipeline([
                    #     ('selector', ItemSelector(key='headword')),
                    #     ('ngram', CountVectorizer(ngram_range=(1, 5))),
                    # ])),

                    # ('bag_of_skipbigram_2', Pipeline([
                    #     ('ngram', CountVectorizer(analyzer=skipper_2, vocabulary=vocabulary_skipper_2)),
                    # ])),

                    # ('bag_of_skipbigram_3', Pipeline([
                    #     ('selector', ItemSelector(key='sentence')),
                    #     ('ngram', CountVectorizer(analyzer=skipper_3, vocabulary=vocabulary_skipper_3)),
                    # ])),

                    # ('bag_of_skipbigram_4', Pipeline([
                    #     ('selector', ItemSelector(key='sentence')),
                    #     ('ngram', CountVectorizer(analyzer=skipper_4, vocabulary=vocabulary_skipper_4)),
                    # ])),

                    # ('bag_of_skipbigram_5', Pipeline([
                    #     ('selector', ItemSelector(key='sentence')),
                    #     ('ngram', CountVectorizer(analyzer=skipper_5, vocabulary=vocabulary_skipper_5)),
                    # ])),

                    ('bag_of_word2vec', Pipeline([
                        ('selector', ItemSelector(key='word2vec')),
                        ('ngram', CountVectorizer(ngram_range=(1, 2))),
                    ])),

                    ('bag_of_glove', Pipeline([
                        ('selector', ItemSelector(key='glove')),
                        ('ngram', CountVectorizer(ngram_range=(1, 2))),
                    ]))

                ]
            )),

            ('clf', LogisticRegression())
        ])

	def get_skip_bigrams_vocabulary(self, k, frequency, index):
		skipper = functools.partial(skipgrams, n=2, k=k)
		cv = CountVectorizer(analyzer=skipper)

		skipngrams = cv.fit_transform([data.split() for data in self.train_data[index]]).toarray()

		sum_skipngrams = []
		for i in range(len(cv.vocabulary_)):
			sum_skipngrams.append(0)

		for skipngram in skipngrams:
			for i, value in enumerate(skipngram):
				sum_skipngrams[i] += value

		vocabulary = []
		for key in cv.vocabulary_:
			if sum_skipngrams[cv.vocabulary_[key]] > frequency:
				vocabulary.append(key)
		return vocabulary

	def train(self, index):
		pipeline = self.get_pipeline(index)
		model = pipeline.fit(self.train_data[index], self.train_target[index])
		pickle.dump(model, open("../../data/category_extraction/category_extractor"+str(index)+".model", "wb"))

	def evaluate_cross_validation(self, index):
		n = 10
		X_folds = np.array_split(self.train_data[index], n)
		y_folds = np.array_split(self.train_target[index], n)

		precision_scores = []
		recall_scores = []
		f1_scores = []

		pipeline = self.get_pipeline(index)

		for k in range(n):
			X_train = list(X_folds)
			X_test  = X_train.pop(k)
			X_train = np.concatenate(X_train)
			y_train = list(y_folds)
			y_test  = y_train.pop(k)
			y_train = np.concatenate(y_train)

			model = pipeline.fit(X_train, y_train)
			predicted = pipeline.predict(X_test)

			# print classification_report(y_test, predicted)

			precision_scores.append(precision_score(y_test, predicted, average=None).mean())
			recall_scores.append(recall_score(y_test, predicted, average=None).mean())
			f1_scores.append(f1_score(y_test, predicted, average=None).mean())

		print "Precision: ", np.array(precision_scores).mean()
		print "Recall: ", np.array(recall_scores).mean()
		print "F1-score: ", np.array(f1_scores).mean()
		

	def evaluate(self, test_filename):
		test_data, test_target = self.read_data(test_filename)
		test_data = np.array(test_data)

		for i in range(0, 4):
			pipeline = self.get_pipeline(i)
			pipeline.fit(self.train_data[i], self.train_target[i])
			predicted = pipeline.predict(test_data[i])

			print "Precision: ", precision_score(test_target[i], predicted, average=None)
			print "Recall: ", recall_score(test_target[i], predicted, average=None)
			print "F1-score: ", f1_score(test_target[i], predicted, average=None)

if __name__ == '__main__':
	sentiment_extractor = SentimentExtractor("../../data/sentiment_extraction/train_data.csv")
	# sentiment_extractor.evaluate_cross_validation(0)
	# sentiment_extractor.evaluate_cross_validation(1)
	# sentiment_extractor.evaluate_cross_validation(2)
	# sentiment_extractor.evaluate_cross_validation(3)

	sentiment_extractor.evaluate("../../data/sentiment_extraction/test_data.csv")