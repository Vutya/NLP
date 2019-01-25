#!/usr/bin/python

# Spearman correlation with labels and cosines between sentence vectors (mean of the word vectores)
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

# 'ruscorpora_mystem_cbow_300_2_2015.bin.gz', 32739 absent words
# test (correlation=0.5062159062455752, pvalue=9.85814576710818e-126)
# train (correlation=0.5780291930506739, pvalue=0.0)

# 'web_mystem_skipgram_500_2_2015.bin.gz', 31881  absent words
# test (correlation=0.5282857455324481, pvalue=9.17614952825262e-139)
# train (correlation=0.5757310095826595, pvalue=0.0)

# 'news_mystem_skipgram_1000_20_2015.bin.gz',  31780 absent words
# test (correlation=0.5516623371319622, pvalue=1.1656447157737911e-153)
# train (correlation=0.5885017358958976, pvalue=0.0)


MODEL_NAME = 'news_mystem_skipgram_1000_20_2015.bin.gz'

model = KeyedVectors.load_word2vec_format(MODEL_NAME, binary=True)


def get_sent_vector_mean(s):
    global model
    vecs = []
    for w in s:
        if w in model.wv:
            vecs.append(model[w])
    return np.mean(np.array(vecs), axis=0)


def main():
    print("Mean")

    # test corpus
    with open('paraphraser_gold\paraphrases_gold_lemmas.txt', 'r', encoding='utf8') as inf1, \
            open('paraphraser_gold\labels.txt', 'r', encoding='utf8') as inf2:
        labels = []
        cosines = []
        for i in range(1924):
            s1 = inf1.readline()
            s2 = inf1.readline()
            v1 = get_sent_vector_mean(s1.split())
            v2 = get_sent_vector_mean(s2.split())
            c = 1 - cosine(v1, v2)
            label = int(inf2.readline())
            cosines.append(c)
            labels.append(label)

        p = spearmanr(np.array(labels), np.array(cosines))
        print(p)

    # train corpus
    with open('paraphraser\paraphrases_lemmas.txt', 'r', encoding='utf8') as inf1, \
            open('paraphraser\labels.txt', 'r', encoding='utf8') as inf2:
        labels = []
        cosines = []
        for i in range(7227):
            s1 = inf1.readline()
            s2 = inf1.readline()
            v1 = get_sent_vector_mean(s1.split())
            v2 = get_sent_vector_mean(s2.split())
            c = 1 - cosine(v1, v2)
            label = int(inf2.readline())
            cosines.append(c)
            labels.append(label)

        p = spearmanr(np.array(labels), np.array(cosines))
        print(p)


if __name__ == '__main__':
    main()
