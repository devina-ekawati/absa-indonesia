from gensim.models import word2vec
from gensim.models import Word2Vec

sentences = word2vec.LineSentence('../data/MST.txt')

num_features = 300
num_workers = 4
context = 10
iteration = 100

print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers, size=num_features, iter=iteration, window=context)

model.init_sims(replace=True)

model_name = "../data/MST/word2vec.model.txt"
model.save_word2vec_format(model_name, binary=False)

print model.most_similar('makanan')