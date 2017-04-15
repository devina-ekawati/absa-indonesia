import sys
import operator
sys.path.append('../word_embedding')

from word2vec import MyWord2Vec

class TupleGenerator():
	def __init__(self):
		self.my_word2vec = MyWord2Vec()

		food = ["makanan", "minuman", "porsi", "menu"]
		self.food_seed_words = self.my_word2vec.get_seed_words(food)

		price = ["harga"]
		self.price_seed_words = self.my_word2vec.get_seed_words(price)

		place = ["tempat", "suasana", "pemandangan", "dekorasi", "toilet", "sofa", "kursi", "meja", "bantal", "lantai", "design", "cuaca", "parkir"]
		self.place_seed_words = self.my_word2vec.get_seed_words(place)

		service = ["pelayan", "pegawai"]
		self.service_seed_words = self.my_word2vec.get_seed_words(service)

	def generate_tuples(self, aspects, categories_sentiments):
		tuples = []
		for aspect in aspects:
			similarity_scores = {}
			if "food" in categories_sentiments:
				similarity_scores["food"] = self.my_word2vec.get_max_similarity_score(aspect, self.food_seed_words)
			if "price" in categories_sentiments:
				similarity_scores["price"] = self.my_word2vec.get_max_similarity_score(aspect, self.price_seed_words)
			if "place" in categories_sentiments:
				similarity_scores["place"] = self.my_word2vec.get_max_similarity_score(aspect, self.place_seed_words)
			if "service" in categories_sentiments:
				similarity_scores["service"] = self.my_word2vec.get_max_similarity_score(aspect, self.service_seed_words)

			category = max(similarity_scores.iteritems(), key=operator.itemgetter(1))[0]
			tuples.append((aspect, category, categories_sentiments[category]))
		return tuples

if __name__ == '__main__':
	aspects = ["makanannya", "pizza"]
	categories_sentiments = {"food": "negative"}
	tg = TupleGenerator()

	print tg.generate_tuples(aspects, categories_sentiments)
