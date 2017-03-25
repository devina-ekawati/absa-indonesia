from gensim import corpora
from gensim import models
from sklearn.cluster import KMeans
from gensim.matutils import corpus2dense


def read_reviews(filename):
	documents = []
	with open (filename, "r") as f:
		for line in f:
			line = line.rstrip()
			documents.append(line)
	return documents

def build_lda(documents, num_topics):
	texts = [[word for word in document.split()] for document in documents]

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	return models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics, workers=4, iterations=1000, passes=100)

def save_word_matrix(lda, filename):
	matrix = {}

	for item in lda.state.get_lambda():
		for i in range(len(lda.id2word)):
			if (not lda.id2word[i] in matrix):
				matrix[lda.id2word[i]] = [item[i]]
			else:
				matrix[lda.id2word[i]].append(item[i])

	with open(filename, "w") as f:
		for key, values in matrix.iteritems():
			line = key
			for value in values:
				line += " " + str(value)

			f.write(line + "\n")

if __name__ == "__main__":
	documents = read_reviews("../data/MST.txt")
	lda = build_lda(documents, 50)
	print "Finish build model"
	save_word_matrix(lda, "../data/MST/lda.model.txt")