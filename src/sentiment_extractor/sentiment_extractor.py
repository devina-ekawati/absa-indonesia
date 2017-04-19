from sentiment_feature_extractor import SentimentFeatureExtractor
from item_selector import ItemSelector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, \
    accuracy_score
from sklearn.externals import joblib
from nltk.util import skipgrams
from collections import OrderedDict
import numpy as np
import csv, re, functools, os


class SentimentExtractor:
    def __init__(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

        self.categories = ['food', 'service', 'price', 'place']
        self.target_names = ['positive', 'negative', 'neutral']
        self.train_data = []
        self.train_target = []
        self.train_data, train_target, all_data = self.read_data(os.path.join(project_path, "../data/sentiment_extraction/train_data.csv"))
        for i in range(len(self.categories)):
            # self.train_data.append(np.array(train_data[i]))
            self.train_target.append(np.array(train_target[i]))
        self.model_filenames = [
            os.path.join(project_path, "../data/sentiment_extraction/sentiment_food.model"),
            os.path.join(project_path, "../data/sentiment_extraction/sentiment_service.model"),
            os.path.join(project_path, "../data/sentiment_extraction/sentiment_price.model"),
            os.path.join(project_path, "../data/sentiment_extraction/sentiment_place.model")
        ]

    def read_data(self, filename):
        all_data = []
        data = [{}, {}, {}, {}]
        targets = [[], [], [], []]
        regex = re.compile('[^0-9a-zA-Z]+')
        with open(filename, "rb") as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            next(reader)
            j = 0
            for row in reader:
                all_data.append(regex.sub(' ', row[0]))
                for i in range(1, len(self.categories) + 1):
                    if row[i] != "-":
                        data[i - 1][j] = regex.sub(' ', row[0])
                        targets[i - 1].append(self.target_names.index(row[i]))
                j += 1
        return data, targets, all_data

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
            ('data', SentimentFeatureExtractor()),

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

        skipngrams = cv.fit_transform([data.split() for data in self.train_data[index].values()]).toarray()

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

    def train(self):
        for index in range(4):
            train_data = np.array(self.train_data[index].values())
            pipeline = self.get_pipeline(index)
            model = pipeline.fit(train_data, self.train_target[index])
            joblib.dump(model, self.model_filenames[index])

    def evaluate_cross_validation(self):
        n = 10

        for index in range(4):
            X_folds = np.array_split(self.train_data[index].values(), n)
            y_folds = np.array_split(self.train_target[index], n)

            precision_scores = []
            recall_scores = []
            f1_scores = []

            pipeline = self.get_pipeline(index)

            for k in range(n):
                X_train = list(X_folds)
                X_test = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_test = y_train.pop(k)
                y_train = np.concatenate(y_train)

                model = pipeline.fit(X_train, y_train)
                predicted = pipeline.predict(X_test)

                # print classification_report(y_test, predicted)

                precision_scores.append(precision_score(y_test, predicted, average=None).mean())
                recall_scores.append(recall_score(y_test, predicted, average=None).mean())
                f1_scores.append(f1_score(y_test, predicted, average=None).mean())

            print "CATEGORY: " + self.categories[index]
            print "\tPrecision: ", np.array(precision_scores).mean()
            print "\tRecall: ", np.array(recall_scores).mean()
            print "\tF1-score: ", np.array(f1_scores).mean()

    def evaluate(self, test_filename):
        test_data, test_target, all_data = self.read_data(test_filename)

        predictions = [{}, {}, {}, {}]
        for i in range(4):
            model = joblib.load(self.model_filenames[i])
            predicted = model.predict(test_data[i].values())

            # print "CATEGORY: " + self.categories[i]
            # print "\tPrecision: ", np.array(precision_score(test_target[i], predicted, average=None)).mean()
            # print "\tRecall: ", np.array(recall_score(test_target[i], predicted, average=None)).mean()
            # print "\tF1-score: ", np.array(f1_score(test_target[i], predicted, average=None)).mean()
            k = 0
            for j in range(len(all_data)):
                if j in test_data[i]:
                    predictions[i][j] = self.target_names[predicted[k]]
                    k += 1
                    # print "\t", j, test_data[i][j], self.target_names[predicted[j]]

        with open("../../data/sentiment_extraction/result_test_data_cumulative.csv", "wb") as f:
            writer = csv.writer(f, delimiter=';', quotechar='"')
            writer.writerow(["sentence", "food", "service", "price", "place"])
            for i, sentence in enumerate(all_data):
                data = [sentence, "-", "-", "-", "-"]
                for j in range(4):
                    if i in test_data[j]:
                        if data[0] == "-":
                            data[0] = test_data[j][i]
                        data[j+1] = predictions[j][i]
                writer.writerow(data)

    def evaluate_accumulative(self, actual_data_filename, test_filename):
        actual_data, actual_target, all_data = self.read_data(actual_data_filename)
        test_data, test_target, all_data = self.read_data(test_filename)

        for i in range(4):
            print "CATEGORY: " + self.categories[i]
            model = joblib.load(self.model_filenames[i])
            predicted = model.predict(test_data[i].values())

            test_data[i] = OrderedDict(test_data[i])
            correct = 0.0
            for key, target in zip(actual_data[i], actual_target[i]):
                if key in test_data[i]:
                    if target == predicted[test_data[i].keys().index(key)]:
                        correct += 1
            precision = correct / len(test_target[i])
            recall = correct/len(actual_target[i])
            f1 = 0
            if recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)

            print "\tPrecision: ", precision
            print "\tRecall: ", recall
            print "\tF1-score: ", f1


    def predict(self, category, test_data):
        results = []
        if category == "food":
            model = joblib.load(self.model_filenames[0])
        elif category == "service":
            model = joblib.load(self.model_filenames[1])
        elif category == "price":
            model = joblib.load(self.model_filenames[2])
        elif category == "place":
            model = joblib.load(self.model_filenames[3])

        predicted = model.predict(np.array(test_data))
        for j in range(len(predicted)):
            results.append(self.target_names[predicted[j]])

        return results

if __name__ == '__main__':
    sentiment_extractor = SentimentExtractor()
    # sentiment_extractor.train()
    # sentiment_extractor.evaluate_cross_validation()

    # sentiment_extractor.evaluate("../../data/sentiment_extraction/test_data_cumulative.csv")
    sentiment_extractor.evaluate_accumulative("../../data/sentiment_extraction/test_data.csv",
                                              "../../data/sentiment_extraction/test_data_cumulative.csv")
