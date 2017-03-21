import os, re
from preprocess import Preprocess

def read_reviews(path):
	reviews = []
	for filename in os.listdir(path):
		with open(path + "/" + filename, "r") as f:
			for line in f:
				line = line.rstrip()[1:-1]
				reviews.append(line)
	return reviews

def write_to_files(filename, data):
	with open(filename, "w") as f:
		for item in data:
			f.write(item+'\n')

def preprocess(delete_stopword):
	data = []

	p = Preprocess()
	regex = re.compile('[^0-9a-zA-Z]+')

	for review in reviews:
		sentences = p.splitIntoSentence(review)
		text = ""
		for sentence in sentences:
			sentence = p.formalizeSentence(sentence)

			if (delete_stopword):
				sentence = p.deleteStopWord(sentence)

			sentence = regex.sub(' ', sentence.lower())

			text += ' '.join(sentence.split()) + ' '

		data.append(text[:-1])

	return data

if __name__ == "__main__":
	path = "../data/reviews/full"
	reviews = read_reviews(path)
	data = preprocess(False)
	write_to_files("MST.txt", data)