import sys
import numpy as np
sys.path.append('../model')
from conll_table import CONLLTable
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self

	def transform(self, sentences):
		print len(sentences)
		CONLL_table = CONLLTable("../../data/output2.conll", False)
		features = np.recarray(shape=(len(sentences),),
                               dtype=[('sentence', object), ('headword', object), ('char', object), ('word2vec', object), ('glove', object)], )
		word2vec_cluster = self.read_word_embedding_cluster("../../data/word_embedding/word2vec_cluster_100.txt")
		glove_cluster = self.read_word_embedding_cluster("../../data/word_embedding/glove_cluster_100.txt")
		for i, text in enumerate(sentences):
			features[i]['headword'] = CONLL_table.get_head_word_of_sentence(i)
			features[i]['sentence'] = text
			features[i]['word2vec'] = self.get_word_embedding(text, word2vec_cluster)
			features[i]['glove'] = self.get_word_embedding(text, glove_cluster)
		return features

	def get_word_embedding(self, sentence, cluster_dict):
		tokens = sentence.split()
		result = ""
		for token in tokens:
			if token in cluster_dict:
				result += cluster_dict[token] + " "
			else:
				result += "0 "
		return result[:-1]

	def read_word_embedding_cluster(self, filename):
		cluster_dict = {}
		with open(filename, "r") as f:
			for line in f:
				line = line.rstrip()
				tokens = line.split()
				cluster_dict[tokens[0]] = tokens[1]
		return cluster_dict
