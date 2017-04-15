import sys
sys.path.append('model')
sys.path.append('tuple_generator')

from subprocess import call
from conll_table import CONLLTable
from tuple_generator import TupleGenerator

class Main():
	def __init__(self):
		self.conjunctions = ["tetapi sayangnya", "namun", "tetapi", "walaupun", "akan tetapi", "sayangnya", "hanya sayang", "sayang", "meski", "walau", "but"]
		self.results = [[], [], [], []]

	def get_conll_table(self):
		conll_table_filename = "../data/output1_test1.conll"
		# eksekusi command buat conll_table
		conll_table = CONLLTable(conll_table_filename, False)
		for sentence in conll_table.get_sentences():
			tokens = sentence.split()
			self.results[0].append(tokens)

	def parts(self, list_, indices):
		indices = [0]+indices+[len(list_)]
		return [list_[v:indices[k+1]] for k, v in enumerate(indices[:-1])]

	def get_aspects(self):
		# call("crfsuite tag -m data/crf/model.crf test.crfsuite.txt > result.txt")
		with open("result.txt", "r") as f:
			label = []
			for line in f:
				line = line.rstrip()
				if line:
					label.append(line)
				else:
					self.results[1].append(label)
					label = []

	def get_aspects_from_tokens(self, tokens, labels):
		result = []
		indices = [i for i, x in enumerate(labels) if x == "ASPECT-B"]

		for i in indices:
			aspect = tokens[i]
			j = i + 1
			print j
			if (j < len(tokens)):
				while labels[j] == "ASPECT-I":
					aspect += " " + tokens[j]
					j += 1
			result.append(aspect)
		return result

	def get_aspect_label(self, tokens):
		label = ["ASPECT-B"]
		for i in range(1, len(tokens)):
			label.append("ASPECT-I")
		return label

	def split_sentences(self):
		tokens = []
		sentences = []
		labels = []
		for i in range(len(self.results[0])):
			sentence = " ".join(self.results[0][i])
			splitted_tokens, label = self.split_sentence(sentence, self.results[1][i])

			for j in range(len(splitted_tokens)):
				tokens.append(splitted_tokens[j])
				labels.append(label[j])

		with open("category_test_data_cumulative.txt", "w") as f:
			for token in tokens:
				f.write(" ".join(token) + "\n")


	def split_sentence(self, sentence, label):
		sentences = []
		tokens = sentence.split()
		indices = []

		for conjunction in self.conjunctions:
			if conjunction in sentence:
				indices += [i for i, x in enumerate(tokens) if x == conjunction]
		indices.sort()

		del_indices = []
		for i in range(1, len(indices)):
			if indices[i] - indices[i-1] == 1:
				del_indices.append(i)

		for i in del_indices:
			indices.pop(i)

		sentence_partitions = self.parts(tokens, indices)
		label_partitions = self.parts(label, indices)
		if len(sentence_partitions) > 1:
			print sentence

		for i in range(1, len(label_partitions)):
			if "ASPECT-B" not in label_partitions[i] and "ASPECT-B" in label_partitions[i-1]:
				aspects = self.get_aspects_from_tokens(sentence_partitions[i-1], label_partitions[i-1])
				last_aspect_tokens = aspects[-1].split()

				sentence_partitions[i].pop(0)
				label_partitions[i].pop(0)
				if sentence_partitions[i][0] in self.conjunctions:
					sentence_partitions[i].pop(0)
					label_partitions[i].pop(0)

				sentence_partitions[i] = last_aspect_tokens + sentence_partitions[i]
				label_partitions[i] = self.get_aspect_label(last_aspect_tokens) + label_partitions[i]

		sentence_partitions = [x for x in sentence_partitions if x != []]
		label_partitions = [x for x in label_partitions if x != []]
		if len(sentence_partitions) > 1:
			print sentence_partitions
			print label_partitions
			print "\n"
		return sentence_partitions, label_partitions

if __name__ == '__main__':
	m = Main()
	# preproses kalimat
	# jadiin conll table
	m.get_conll_table()
	# ekstraksi aspek
	m.get_aspects()
	# split setiap sentence
	m.split_sentences()
	# ekstraksi kategori
	# ekstraksi sentimen
	# generate tuple aspek, kategori, sentimen