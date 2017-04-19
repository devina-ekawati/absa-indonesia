import operator, re

from word2vec import MyWord2Vec


class TupleGenerator():
    def __init__(self):
        self.my_word2vec = MyWord2Vec()

        food = ["makanan", "minuman", "porsi", "menu"]
        self.food_seed_words = self.my_word2vec.get_seed_words(food)

        price = ["harga"]
        self.price_seed_words = self.my_word2vec.get_seed_words(price)

        place = ["tempat", "suasana", "pemandangan", "dekorasi", "toilet", "sofa", "kursi", "meja", "bantal", "lantai",
                 "design", "cuaca", "parkir"]
        self.place_seed_words = self.my_word2vec.get_seed_words(place)

        service = ["pelayan", "pegawai"]
        self.service_seed_words = self.my_word2vec.get_seed_words(service)

    def generate_tuples(self, aspects, categories_sentiments):
        tuples = []
        regex = re.compile('[^0-9a-zA-Z]+')
        for aspect in aspects:
            if len(categories_sentiments) > 1:
                aspect = regex.sub(" ", aspect)
                similarity_scores = {"food": 0, "price": 0, "place": 0, "service": 0}
                tokens = aspect.split()
                for token in tokens:
                    if "food" in categories_sentiments:
                        similarity_scores["food"] += (self.my_word2vec.get_max_similarity_score(token, self.food_seed_words))/2
                    if "price" in categories_sentiments:
                        similarity_scores["price"] += (self.my_word2vec.get_max_similarity_score(token, self.price_seed_words))/2
                    if "place" in categories_sentiments:
                        similarity_scores["place"] += (self.my_word2vec.get_max_similarity_score(token, self.place_seed_words))/2
                    if "service" in categories_sentiments:
                        similarity_scores["service"] += (self.my_word2vec.get_max_similarity_score(token,
                                                                                             self.service_seed_words))/2
                # print similarity_scores
                category = max(similarity_scores.iteritems(), key=operator.itemgetter(1))[0]
            else:
                category = categories_sentiments.keys()[0]

            tuples.append((aspect, category, categories_sentiments[category]))
        return tuples


if __name__ == '__main__':
    aspects = ["makanannya", "kebiasaan pelayan"]
    categories_sentiments = {"food": "negative", "service": "negative"}
    tg = TupleGenerator()

    print tg.generate_tuples(aspects, categories_sentiments)
