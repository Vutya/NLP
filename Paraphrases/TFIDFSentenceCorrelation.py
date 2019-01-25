#!/usr/bin/python

# Spearman correlation with labels and cosines between sentence vectors (TF-IDF for sentenece vectors)
import math
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

# 'ruscorpora_mystem_cbow_300_2_2015.bin.gz', 32739 absent words
# test (correlation=0.524824532725846, pvalue=1.1759821998212466e-136)
# train (correlation=0.5992753060250473, pvalue=0.0)

# 'web_mystem_skipgram_500_2_2015.bin.gz', 31881  absent words
# test (correlation=0.5412581747436309, pvalue=6.942008360305112e-147)
# train (correlation=0.5949652774455758, pvalue=0.0)

# 'news_mystem_skipgram_1000_20_2015.bin.gz',  31780 absent words
# test (correlation=0.5555548949797993, pvalue=2.949915870841361e-156)
# train (correlation=0.605819275027462, pvalue=0.0)


MODEL_NAME = 'news_mystem_skipgram_1000_20_2015.bin.gz'

model = KeyedVectors.load_word2vec_format(MODEL_NAME, binary=True)


def get_train_corpus():
    corpus = []
    with open('paraphraser\paraphrases_lemmas.txt', 'r', encoding='utf8') as lines:
        for line in lines:
            corpus.append(line.strip())
    return corpus


def get_test_corpus():
    corpus = []
    with open('paraphraser_gold\paraphrases_gold_lemmas.txt', 'r', encoding='utf8') as lines:
        for line in lines:
            corpus.append(line.strip())
    return corpus


def tf(word, s):
    return s.count(word)


def idf(v):
    return math.log((v.shape[0] + 1) / (np.count_nonzero(v.todense()) + 1), math.e) + 1


def unk_tfidf(word, s, v):
    return tf(word, s) * idf(v)


def splitter(doc):
    return doc.split()


VECTORIZER = TfidfVectorizer(lowercase=False, norm=None, tokenizer=splitter)
train_TFIDF = VECTORIZER.fit_transform(get_train_corpus())
test_TFIDF = VECTORIZER.transform(get_test_corpus())
FEATURE_NAMES = VECTORIZER.get_feature_names()


def get_synonyms(word):
    global model, FEATURE_NAMES
    res = []
    for p in model.similar_by_word(word, topn=20):
        if p[0] in FEATURE_NAMES and len(res) < 5:
            res.append(p[0])
    return res


def get_sent_vector_tfidf(s, matrix, i):
    global model, FEATURE_NAMES
    vecs = []
    coefs = []
    res = []
    for w in s:
        if w not in model.wv:
            continue
        if w in FEATURE_NAMES:
            vecs.append(model[w])
            coefs.append(matrix[i, FEATURE_NAMES.index(w)])
        else:
            synonyms = get_synonyms(w)
            if len(synonyms) != 0:
                vecs.append(model[w])
                uti = []
                for syn in synonyms:
                    uti.append(unk_tfidf(w, s, train_TFIDF[:, FEATURE_NAMES.index(syn)]))
                coefs.append(sum(uti)/len(uti))
    for i in range(len(coefs)):
        res.append(vecs[i] * (coefs[i] / sum(coefs)))
    return sum(res)


def main():
    print("TF-IDF")

    # test corpus
    with open('paraphraser_gold\paraphrases_gold_lemmas.txt', 'r', encoding='utf8') as inf1, \
            open('paraphraser_gold\labels.txt', 'r', encoding='utf8') as inf2:
        labels = []
        cosines = []
        for i in range(1924):
            s1 = inf1.readline()
            s2 = inf1.readline()
            v1 = get_sent_vector_tfidf(s1.split(), test_TFIDF, 2 * i)
            v2 = get_sent_vector_tfidf(s2.split(), test_TFIDF, 2 * i + 1)
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
            v1 = get_sent_vector_tfidf(s1.split(), train_TFIDF, 2 * i)
            v2 = get_sent_vector_tfidf(s2.split(), train_TFIDF, 2 * i + 1)
            c = 1 - cosine(v1, v2)
            label = int(inf2.readline())
            cosines.append(c)
            labels.append(label)

        p = spearmanr(np.array(labels), np.array(cosines))
        print(p)


if __name__ == '__main__':
    main()
