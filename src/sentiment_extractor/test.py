import functools
from nltk.util import skipgrams
from sklearn.feature_extraction.text import CountVectorizer

sent = ["Insurgents killed in ongoing fighting".split(), "Insurgents wanted to kill the ongoing fighting".split()]

skipper = functools.partial(skipgrams, n=2, k=2)

cv = CountVectorizer(analyzer=skipper)
skipngrams = cv.fit_transform(sent).toarray()
print skipngrams
print cv.vocabulary_
vocabulary = []

sum_skipngrams = []
for i in range(len(cv.vocabulary_)):
    sum_skipngrams.append(0)

for skipngram in skipngrams:
    for i, value in enumerate(skipngram):
        sum_skipngrams[i] += value

for key in cv.vocabulary_:
    if sum_skipngrams[cv.vocabulary_[key]] > 1:
        vocabulary.append(key)

print vocabulary

cv = CountVectorizer(analyzer=skipper, vocabulary=vocabulary)
print cv.fit_transform(sent).toarray()