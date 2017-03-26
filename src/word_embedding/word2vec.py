from subprocess import call
from gensim.models import word2vec, Word2Vec

# call("jython ../preprocess/preprocess.py ../../data/reviews/reviews.txt ../../data/reviews/preprocessed_reviews_sentence.txt")

sentences = word2vec.LineSentence('../../data/reviews/preprocessed_reviews_sentence.txt')

num_features = 300
num_workers = 4
context = 10
iteration = 100

print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers, size=num_features, iter=iteration, window=context, min_count=1)

model.init_sims(replace=True)

model_name = "../../data/word_embedding/word2vec.model.txt"
model.save_word2vec_format(model_name, binary=False)

print model.most_similar('abiis')