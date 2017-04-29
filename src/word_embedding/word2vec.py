from subprocess import call
from gensim.models import word2vec, Word2Vec

import os


class MyWord2Vec():
    def __init__(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

        self.model_name = os.path.join(project_path, "../data/word_embedding/word2vec.model.txt")
        self.reviews_filename = os.path.join(project_path, '../data/reviews/preprocessed_reviews_sentence.txt')

        if os.path.exists(self.model_name):
            self.model = Word2Vec.load_word2vec_format(self.model_name, binary=False)
        else:
            self.train()

    def train(self):
        sentences = word2vec.LineSentence(self.reviews_filename)
        num_features = 300
        num_workers = 4
        context = 10
        iteration = 100

        print "Training Word2Vec model..."
        self.model = Word2Vec(sentences, workers=num_workers, size=num_features, iter=iteration, window=context,
                              min_count=1)

        self.model.init_sims(replace=True)
        self.model.save_word2vec_format(self.model_name, binary=False)

    def get_most_similar_words(self, word, n=10):
        return self.model.most_similar(word, topn=n)

    def get_similarity_score(self, word1, word2):
        return self.model.similarity(word1, word2)

    def get_seed_words(self, words):
        seed_words = []
        for word in words:
            for item in self.get_most_similar_words(word):
                if item[0] not in seed_words:
                    seed_words.append(item[0])

        return seed_words

    def get_max_similarity_score(self, key, words):
        max = 0
        for word in words:
            score = self.get_similarity_score(key, word)
            if score > max:
                max = score
        return max


if __name__ == '__main__':
    my_word2vec = MyWord2Vec()

    food = ["makanan", "minuman", "porsi", "menu"]
    food_seed_words = my_word2vec.get_seed_words(food)

    price = ["harga"]
    price_seed_words = my_word2vec.get_seed_words(price)

    place = ["tempat", "suasana", "pemandangan", "dekorasi", "toilet", "sofa", "kursi", "meja", "bantal", "lantai",
             "design", "cuaca", "parkir"]
    place_seed_words = my_word2vec.get_seed_words(place)

    service = ["pelayan", "pegawai"]
    service_seed_words = my_word2vec.get_seed_words(service)

    key = "gurame"
    food_score = my_word2vec.get_max_similarity_score(key, food_seed_words)
    price_score = my_word2vec.get_max_similarity_score(key, price_seed_words)
    place_score = my_word2vec.get_max_similarity_score(key, place_seed_words)
    service_score = my_word2vec.get_max_similarity_score(key, service_seed_words)

    print food_score, price_score, place_score, service_score