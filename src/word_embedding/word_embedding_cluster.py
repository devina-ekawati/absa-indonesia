from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy

class autovivify_list(dict):
    '''Pickleable class to replicate the functionality of collections.defaultdict'''
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
                return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
                return -1 * x
        raise ValueError

def build_word_vector_matrix(vector_file, n_words):
    '''Iterate over the GloVe array read from sys.argv[1] and return its vectors and labels as arrays'''
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            labels_array.append(sr[0])
            numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )

            if c == n_words:
                    return numpy.array( numpy_arrays ), labels_array

    return numpy.array( numpy_arrays ), labels_array

def find_word_clusters(labels_array, cluster_labels):
    '''Read in the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[ i ].append( labels_array[c] )
    return cluster_to_words

def find_id_cluster(word, cluster):
    i = 0
    for c in cluster_to_words:
        if (word not in cluster_to_words[c]):
            i += 1
        else:
            break
    return i

def write_to_file(cluster, filename, labels_array):
    with open(filename, "w") as f:
        for label in labels_array:
            f.write(label + " " + str(find_id_cluster(label, cluster)) + "\n")


if __name__ == "__main__":

    input_vector_file = sys.argv[1]
    n_words           = int(sys.argv[2])
    clusters_to_make  = int(sys.argv[3])
    cluster_filename  = sys.argv[4]
    df, labels_array  = build_word_vector_matrix(input_vector_file, n_words)
    kmeans_model      = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10)
    kmeans_model.fit(df)

    cluster_labels    = kmeans_model.labels_
    cluster_inertia   = kmeans_model.inertia_
    cluster_to_words  = find_word_clusters(labels_array, cluster_labels)

    # for c in cluster_to_words:
    #     print cluster_to_words[c]

    write_to_file(cluster_to_words, cluster_filename, labels_array)