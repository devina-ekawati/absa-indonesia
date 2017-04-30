import crfutils
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

# Separator of field values.
separator = ' '

# Field names of the input data.
fields = 'w pos head word2vec100 word2vec5000 y'
# fields = 'w pos dict head word2vec'

# templates = [(('w', 0),), (('dict', 0),), (('head', 0),)]
# templates = [(('dict', 0),), (('head', 0),)]

templates = [
	# (('dict', 0), ),
	# (('head', 0), ),
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -2), ('w',  -1)),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('w',  1), ('w',  2)),
    (('w', -2), ('w', -1), ('w',  0)),
    (('w', -1), ('w',  0), ('w',  1)),
    (('pos', -2), ),
    (('pos', -1), ),
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),
    (('pos',  1), ('pos',  2)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    (('word2vec5000', -2), ),
    (('word2vec5000', -1), ),
    (('word2vec5000',  0), ),
    (('word2vec5000',  1), ),
    (('word2vec5000',  2), ),
    (('word2vec100', -2), ('word2vec100',  -1)),
    (('word2vec100', -1), ('word2vec100',  0)),
    (('word2vec100',  0), ('word2vec100',  1)),
    (('word2vec100',  1), ('word2vec100',  2))
    ]

unigram_filename = "../../data/crf/list_unigrams.txt"

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

def generate_templates():
	global fields, templates
	generate_templates_from_file(unigram_filename, "U")

	fields += ' y'


def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature


if __name__ == '__main__':
	# generate_templates()
	crfutils.main(feature_extractor, fields=fields, sep=separator)
	
