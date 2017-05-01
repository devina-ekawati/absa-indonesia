import re, os
from subprocess import call
from conll_table import CONLLTable
from tuple_generator import TupleGenerator
from category_extractor import CategoryExtractor
from sentiment_extractor import SentimentExtractor

class Main():
    def __init__(self):
        self.categories = ['food', 'service', 'price', 'place']
        self.conjunctions = ["tetapi sayangnya", "namun", "tetapi", "walaupun", "akan tetapi", "sayangnya",
                             "hanya sayang", "sayang", "meski", "walau", "but"]
        self.results = [[], [], [], []]
        self.env = {}
        with open(".env", "r") as f:
            for line in f:
                line = line.rstrip()
                tokens = line.split("=")
                self.env[tokens[0]] = tokens[1]

    def preprocess(self, input_filename, output_filename):
        call("jython preprocess/preprocess.py " + input_filename + " " + output_filename, shell=True)

    def get_conll_table(self, input_filename, conll_table_filename):
        prev_dir = os.getcwd()
        os.chdir(os.path.expanduser("models/syntaxnet"))
        command = "cat ../../" + input_filename + " | syntaxnet/models/parsey_universal/parse.sh $MODEL_DIRECTORY > ../../" \
                  + conll_table_filename
        call(command, env={"MODEL_DIRECTORY": self.env["MODEL_DIRECTORY"]},shell=True)
        os.chdir(prev_dir)
        conll_table = CONLLTable(conll_table_filename, False)
        for sentence in conll_table.get_sentences():
            tokens = sentence.split()
            self.results[0].append(tokens)

    def parts(self, list_, indices):
        indices = [0] + indices + [len(list_)]
        return [list_[v:indices[k + 1]] for k, v in enumerate(indices[:-1])]

    def get_aspects(self, conll_table_filename):
        call("python aspect_extractor/crf_data_generator.py ../data/test/test.txt " + conll_table_filename + " false",
             shell=True)
        call("cat ../data/test/test.txt | python aspect_extractor/crfsuite_data_generator.py > "
             "../data/test/test.crfsuite.txt", shell=True)
        call("./aspect_extractor/crfsuite-0.12/bin/crfsuite tag -m ../data/crf/crf.model "
             "../data/test/test.crfsuite.txt > ../data/test/crf_result.txt", shell=True)
        with open("../data/test/crf_result.txt", "r") as f:
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
            while j < len(tokens):
                if labels[j] == "ASPECT-I":
                    aspect += " " + tokens[j]
                    j += 1
                else:
                    break
            result.append(aspect)
        return result

    def get_aspect_label(self, tokens):
        label = ["ASPECT-B"]
        for i in range(1, len(tokens)):
            label.append("ASPECT-I")
        return label

    def split_sentences(self):
        tokens = []
        labels = []
        for i in range(len(self.results[0])):
            sentence = " ".join(self.results[0][i])
            splitted_tokens, label = self.split_sentence(sentence, self.results[1][i])

            for j in range(len(splitted_tokens)):
                tokens.append(splitted_tokens[j])
                labels.append(label[j])

        # with open("category_test_data_cumulative.txt", "w") as f:
        #     for token in tokens:
        #         f.write(" ".join(token) + "\n")

        self.results[0] = tokens
        self.results[1] = labels

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
            if indices[i] - indices[i - 1] == 1:
                del_indices.append(i)

        for i in del_indices:
            indices.pop(i)

        sentence_partitions = self.parts(tokens, indices)
        label_partitions = self.parts(label, indices)
        # if len(sentence_partitions) > 1:
        #     print sentence

        for i in range(1, len(label_partitions)):
            if "ASPECT-B" not in label_partitions[i] and "ASPECT-B" in label_partitions[i - 1]:
                aspects = self.get_aspects_from_tokens(sentence_partitions[i - 1], label_partitions[i - 1])
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
        # if len(sentence_partitions) > 1:
        #     print sentence_partitions
        #     print label_partitions
        #     print "\n"
        return sentence_partitions, label_partitions

    def get_categories(self):
        category_extractor = CategoryExtractor()
        regex = re.compile('[^0-9a-zA-Z]+')

        sentences = [regex.sub(" ", " ".join(x)) for x in self.results[0]]

        self.results[2] = category_extractor.predict(sentences)

    def get_sentiments(self):
        sentiment_extractor = SentimentExtractor()
        regex = re.compile('[^0-9a-zA-Z]+')

        separated_reviews = []
        for i in range(len(self.categories)):
            reviews = {}
            for j in range(len(self.results[0])):
                if self.categories[i] in self.results[2][j]:
                    reviews[j] = regex.sub(" ", ' '.join(self.results[0][j]))
            separated_reviews.append(reviews)

        self.results[3] = [[] for i in range(1000)]

        for i in range(len(self.categories)):
            results = sentiment_extractor.predict(self.categories[i], separated_reviews[i].values())

            for key, result in zip(separated_reviews[i], results):
                self.results[3][key].append(result)

        # for i in range(len(self.results[0])):
        #     print i+1, self.results[2][i], self.results[3][i]

    def get_tuples(self):
        tuples = {"food": [], "price": [], "place": [], "service": []}
        tuple_generator = TupleGenerator()

        for i in range(len(self.results[0])):
            aspects = self.get_aspects_from_tokens(self.results[0][i], self.results[1][i])
            category_sentiment = {}
            for j in range(len(self.results[2][i])):
                category = self.results[2][i][j]
                category_sentiment[category] = self.results[3][i][j]

            # print aspects, category_sentiment

            if len(aspects) > 0 and len(category_sentiment) > 0:
                result = tuple_generator.generate_tuples(aspects, category_sentiment)

                for key in result:
                    tuples[key] += result[key]

        return tuples

    def get_ratings(self, tuples):
        ratings = {"food": 0, "price": 0, "place": 0, "service": 0}

        for category in tuples:
            pos = len(tuples[category]["positive"])
            neg = len(tuples[category]["negative"])
            ratings[category] = (pos * 4 / (pos + neg)) + 1

        return ratings

if __name__ == '__main__':
    m = Main()
    input_filename = "../data/reviews/tizi_reviews.txt"
    conll_filename = "../data/test/test.conll"
    output_filename = "../data/test/preprocessed_reviews.txt"
    # preproses kalimat
    m.preprocess(input_filename, output_filename)
    # jadiin conll table
    m.get_conll_table(output_filename, conll_filename)
    # ekstraksi aspek
    m.get_aspects(conll_filename)
    # split setiap sentence
    m.split_sentences()
    # ekstraksi kategori
    m.get_categories()
    # ekstraksi sentimen
    m.get_sentiments()
    # generate tuple aspek, kategori, sentimen
    tuples = m.get_tuples()

    print m.get_ratings(tuples)
