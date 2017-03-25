import crfutils
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

# Separator of field values.
separator = ' '

# Field names of the input data.
# fields = 'w pos dict'
fields = 'w pos'

# templates = [(('w', 0),)]
# templates = [(('w', 0),), (('dict', 0),)]
templates = []

def generate_templates_from_file(filename, symbol):
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

def generate_templates(unigram_filename, bigram_filename, ngram_filename, ngram_pos_tag_filename, dependency_tags_filename):
	global fields
	generate_templates_from_file(unigram_filename, "U")
	# generate_templates_from_file(bigram_filename, "B")
	# generate_templates_from_file(ngram_filename, "T")
	# generate_templates_from_file(ngram_pos_tag_filename, "TP")
	# generate_templates_from_file(dependency_tags_filename, "D")

	fields += ' y'


def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature


if __name__ == '__main__':
	generate_templates("../../data/crf/list_unigrams.txt", "../../data/crf/list_bigrams.txt", "../../data/crf/list_trigrams.txt",
		"../../data/crf/list_pos_tag_trigrams.txt", "../../data/crf/dependency_tags.txt")
	crfutils.main(feature_extractor, fields=fields, sep=separator)
	
