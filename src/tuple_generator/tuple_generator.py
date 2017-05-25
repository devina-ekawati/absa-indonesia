import operator, re
from word2vec import MyWord2Vec

class TupleGenerator():
    def __init__(self):
        self.my_word2vec = MyWord2Vec()

        # food = ["makanan", "minuman", "porsi", "menu"]
        # self.food_seed_words = self.my_word2vec.get_seed_words(food)
        #
        # price = ["harga"]
        # self.price_seed_words = self.my_word2vec.get_seed_words(price)
        #
        # place = ["tempat", "suasana", "pemandangan", "dekorasi", "toilet", "sofa", "kursi", "meja", "bantal", "lantai",
        #          "design", "cuaca", "parkir"]
        # self.place_seed_words = self.my_word2vec.get_seed_words(place)
        #
        # service = ["pelayan", "pegawai"]
        # self.service_seed_words = self.my_word2vec.get_seed_words(service)

        self.food_seed_words = [u'masakan', u'menu', u'hidangan', u'makanannya',
                                u'jajanan', u'minuman', u'kuliner', u'asian',
                                u'makanan', u'wine', u'yoghurt', u'minumannya',
                                u'dessertnya', u'ringan', u'dring',  u'snack',
                                u'porsinya', u'ukurannya', u'potongannya', u'ekor',
                                u'pcsnya', u'ukuran', u'berukuran', u'lumayan', u'mangkok',
                                u'porsi', u'menunya', u'topping', u'tenant', u'udon']
        self.price_seed_words = [u'harganya', u'hargapun', u'harganyapun', u'hrga', u'harga']
        self.place_seed_words = [u'restoran', u'resto', u'kafe', u'restaurant', u'tempatnya',
                                 u'cafe', u'restauran', u'rumah', u'ruang', u'area', u'suasananya',
                                 u'suasanya', u'nuansa', u'lingkungan', u'pemandangan', u'konsep',
                                 u'suasa', u'udara', u'atmosfer', u'atmosfir', u'view', u'viewnya',
                                 u'panorama', u'suasana', u'decoration', u'ketinggian',
                                 u'pemandangannya',  u'penataan', u'design', u'tema',
                                 u'interior', u'desain', u'lighting', u'dekorasinya',
                                 u'pajangan', u'hiasan', u'rodanya', u'musholla', u'kotor',
                                 u'mejanya', u'berdebu', u'bantalnya', u'signal', u'mejannya',
                                 u'channel', u'dibersihkan', u'kursi', u'bantal', u'lantai',
                                 u'layar', u'lesehan', u'meja', u'bed', u'smoking', u'teras',
                                 u'sofa', u'mainan', u'ruangan', u'kayu', u'duduk', u'antrian',
                                 u'sana', u'dibagian', u'bangunan',
                                 u'disediakn', u'tirainya', u'dipeluk', u'sholatnya', u'kelambu',
                                 u'tikar', u'sajadah', u'lt', u'dilantai', u'kamar', u'sejuk',
                                 u'exibition', u'berlantai', u'danau', u'ruangannya',
                                 u'dekorasi', u'dekor', u'homey', u'apik', u'tematik', u'suhu',
                                 u'hawa', u'diudara', u'cuacanya', u'udaranya', u'angin', u'hujan',
                                 u'sepoi', u'parkirnya', u'parkiran', u'mobil', u'parkirannya',
                                 u'truk', u'mencapainya', u'mencharge', u'macett', u'parking', u'toilet']
        self.service_seed_words = [u'waiter', u'staff', u'waitress', u'petugas', u'pelayannya',
                                 u'pramusaji', u'staf', u'waiternya', u'pegawai', u'pegawainya',
                                 u'warganya', u'karyawan', u'komputerisasi', u'helpfull', u'pelayan']

    def generate_tuples(self, aspects, categories_sentiments):
        tuples = {"food": {"positive": [], "negative": []}, "price": {"positive": [], "negative": []}
            , "place": {"positive": [], "negative": []}, "service": {"positive": [], "negative": []}}
        regex = re.compile('[^0-9a-zA-Z]+')
        for aspect in aspects:
            if len(categories_sentiments) > 1:
                aspect = regex.sub(" ", aspect)
                similarity_scores = {"food": 0, "price": 0, "place": 0, "service": 0}
                tokens = aspect.split()
                for token in tokens:
                    if "food" in categories_sentiments:
                        similarity_scores["food"] += self.my_word2vec.get_max_similarity_score(token, self.food_seed_words)
                    if "price" in categories_sentiments:
                        similarity_scores["price"] += self.my_word2vec.get_max_similarity_score(token, self.price_seed_words)
                    if "place" in categories_sentiments:
                        similarity_scores["place"] += self.my_word2vec.get_max_similarity_score(token, self.place_seed_words)
                    if "service" in categories_sentiments:
                        similarity_scores["service"] += self.my_word2vec.get_max_similarity_score(token, self.service_seed_words)
                if len(tokens) > 1:
                    for category in similarity_scores:
                        similarity_scores[category] /= len(tokens)

                print similarity_scores
                category = max(similarity_scores.iteritems(), key=operator.itemgetter(1))[0]
            else:
                category = categories_sentiments.keys()[0]

            tuples[category][categories_sentiments[category]].append(aspect)
            # tuples.append((aspect, category, categories_sentiments[category]))
        return tuples


if __name__ == '__main__':
    aspects = ["makanannya", "kebiasaan pelayan", "steak"]
    categories_sentiments = {"food": "negative", "service": "negative", "food": "positive"}
    tg = TupleGenerator()

    tuples = tg.generate_tuples(aspects, categories_sentiments)
    print tuples
