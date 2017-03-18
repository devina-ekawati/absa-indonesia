import crfutils
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

# Separator of field values.
separator = ' '

# Field names of the input data.
fields = 'w pos'

# templates = [(('pos', -2),('pos', -1),('pos', 0),('pos', 1),('pos', 2)), (('w', -2),('w', -1),('w', 0),('w', 1),('w', 2))]

templates = []

def generate_ngram_templates(filename, symbol):
	global fields, templates
	with open(filename, "r") as f:
		i = 0
		for line in f:
			line = line.rstrip()
			if line:
				coloumn = separator + symbol + str(i)
				fields += coloumn
				templates += [((coloumn[1:], 0),)]
				i += 1

def generate_templates(unigram_filename, bigram_filename, ngram_filename, ngram_pos_tag_filename):
	global fields
	# generate_ngram_templates(unigram_filename, "U")
	# generate_ngram_templates(bigram_filename, "B")
	# generate_ngram_templates(ngram_filename, "T")
	generate_ngram_templates(ngram_pos_tag_filename, "TP")

	fields += ' y'


def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature


if __name__ == '__main__':
	generate_templates("../../data/list_unigrams.txt", "../../data/list_bigrams.txt", "../../data/list_trigrams.txt", "../../data/list_pos_tag_trigrams.txt")
	crfutils.main(feature_extractor, fields=fields, sep=separator)
	
