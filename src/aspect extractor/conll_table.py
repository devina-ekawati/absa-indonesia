import collections

class CONLLTable :

	id_word = 0
	id_pos_tag = 1
	id_parent = 2
	id_tree_tag = 3
	id_label = 4

	def __init__(self, filename):
		self.tables = []
		self.stopword = []
		self.read_CONLL_from_file(filename)
		self.read_stopword_list()

	def get_word(self, value):
		return value[self.id_word]

	def get_pos_tag(self, value):
		return value[self.id_pos_tag]

	def get_parent(self, value):
		return value[self.id_parent]

	def get_tree_tag(self, value):
		return value[self.id_tree_tag]

	def get_label(self, value):
		return value[self.id_label]

	def get_tables(self):
		return self.tables

	def get_row(self, id_sentence, id_token):
		return self.tables[id_sentence][id_token]

	def get_parent_tag(self, id_sentence, value):
		id_parent = self.get_parent(value)
		return self.get_row(id_sentence, id_parent)

	def get_sentences_size(self):
		return len(self.tables)

	def get_sentence_size(self, id_sentence, remove_punctuation=True, remove_stopword=False):
		size = 0
		if (remove_punctuation):
			for key, value in self.tables[id_sentence].iteritems():
				if (remove_stopword):
					if (value[self.id_pos_tag] != "PUNCT" and value[self.id_word] not in self.stopword):
						size += 1
				else:
					if (value[self.id_pos_tag] != "PUNCT"):
						size += 1
		else:
			if (remove_stopword):
				if (value[self.id_word] not in self.stopword):
						size += 1
			else:
				size = len(self.tables[id_sentence])

		return size

	def get_sentence(self, id_sentence, remove_punctuation=True, remove_stopword=False):
		sentence = ""
		for key, value in self.tables[id_sentence].iteritems():
			if (remove_punctuation):
				if (remove_stopword):
					if (value[self.id_pos_tag] != "PUNCT" and value[self.id_word] not in self.stopword):
						sentence += value[self.id_word] + " "
				else:
					if (value[self.id_pos_tag] != "PUNCT"):
						sentence += value[self.id_word] + " "
			else:
				if (remove_stopword):
					if (value[self.id_word] not in self.stopword):
						sentence += value[self.id_word] + " "
					else:
						sentence += value[self.id_word] + " "
		return sentence[:-1]

	def get_sentences(self, start=0, end=None, remove_punctuation=True, remove_stopword=False):
		if (end == None):
			end = len(self.tables)

		sentences = []
		tables = self.tables
		for i in range(start, end):
			sentences.append(self.get_sentence(i, remove_punctuation, remove_stopword))

		return sentences

	def get_filtered_sentences(self, filter=None):
		sentences = []

		tables = self.tables
		for i in range(len(tables)):
			result = self.filter_words_by_pos_tag(i, filter)
			sentence = ""
			for key, value in result.iteritems():
				sentence += value + " "
			sentences.append(sentence[:-1])

		return sentences

	def get_sentence_pos_tag(self, id_sentence, remove_punctuation=True, remove_stopword=False):
		sentence_pos_tag = ""
		for key, value in self.tables[id_sentence].iteritems():
			if (remove_punctuation):
				if (remove_stopword):
					if (value[self.id_pos_tag] != "PUNCT" and value[self.id_word] not in self.stopword):
						sentence_pos_tag += value[self.id_pos_tag] + " "
				else:
					if (value[self.id_pos_tag] != "PUNCT"):
						sentence_pos_tag += value[self.id_pos_tag] + " "
			else:
				if (remove_stopword):
					if (value[self.id_word] not in self.stopword):
						sentence += value[self.id_pos_tag] + " "
					else:
						sentence += value[self.id_pos_tag] + " "

		return sentence_pos_tag[:-1]

	def get_sentences_pos_tag(self, remove_punctuation=True, remove_stopword=False):
		sentences = []
		tables = self.tables
		for i in range(len(tables)):
			sentences.append(self.get_sentence_pos_tag(i, remove_punctuation, remove_stopword))

		return sentences

	def get_siblings(self, id_sibling, id_sentence, tag_filter=None, start=0, end=None, remove_punctuation=True):
		siblings = {}
		id_parent = self.get_parent(self.tables[id_sentence][id_sibling])

		siblings = self.get_children(id_parent, id_sentence, tag_filter=None, start=0, end=None, remove_punctuation=True)
		siblings.pop(id_sibling)

		return siblings


	def get_children(self, id_parent, id_sentence, tag_filter=None, start=0, end=None, remove_punctuation=True):
		children = {}

		if end is None:
			end = len(self.tables[id_sentence])


		for key, value in self.tables[id_sentence].iteritems():
			if (key >= start and key <= end):
				if tag_filter is None:
					if (remove_punctuation):
						if (self.get_parent(value) == id_parent and self.get_pos_tag(value) != "PUNCT"):
							children[key] = value
					else:
						if (self.get_parent(value) == id_parent):
							children[key] = value
				else:
					if (remove_punctuation):
						if (self.get_parent(value) == id_parent and self.get_tree_tag(value) == tag_filter and self.get_pos_tag(value) != "PUNCT"):
							children[key] = value
					else:
						if (self.get_parent(value) == id_parent and self.get_tree_tag(value) == tag_filter):
							children[key] = value
		return children

	def filter_words_by_pos_tag(self, id_sentence, filter):
		results = {}

		for key, value in self.tables[id_sentence].iteritems():
			if (value[self.id_pos_tag] in filter):
				results[key] = value[self.id_word]

		return results

	def read_stopword_list(self):
		stopword = []
		with open("../preprocess/resource/stopword.txt", "r") as f:
			for line in f:
				stopword.append(line.rstrip())
		self.stopword = stopword

	def read_CONLL_from_file(self, filename):
		tables = []
		with open(filename, "r") as f:
			line_sentence = []
			for line in f:
				line = line.rstrip()
				if line:
					line_sentence.append(line)
				else:
					CONLL_sentence = {}
					for line_word in line_sentence:
						tokens = line_word.split("\t")

						CONLL_sentence[int(tokens[0])] = (tokens[1], tokens[3], int(tokens[6]), tokens[7], tokens[10])

					tables.append(collections.OrderedDict(sorted(CONLL_sentence.items())))
					line_sentence = []
		self.tables = tables


if __name__ == "__main__":
	CONLL_table = CONLLTable("../../data/output1.conll")

	tables = CONLL_table.get_tables()

	# print tables[0]

	# print CONLL_table.get_children(2, 0)
	# print CONLL_table.get_siblings(8, 0)

	# print CONLL_table.get_sentence(0)

	print CONLL_table.get_sentences()

	# filter = ["NOUN", "ADJ", "ADV", "VERB"]
	# print CONLL_table.filter_words_by_pos_tag(0, filter)

	# print CONLL_table.get_sentence(2)
				