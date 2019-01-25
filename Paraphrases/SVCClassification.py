#!/usr/bin/python

'''
-----Выбор 'news_mystem_skipgram_1000_20_2015.bin.gz', TF-IDF sentence-----

Далее различные наборы признаков и параметры модели
# best_params_; features
# averages for precission, recall, fscore

1) {'C': 0.1}; 2 TF_IDF_KNN sentence vectors
'micro': (0.4875259875259875, 0.4875259875259875, 0.4875259875259875, None)
'macro': (0.4406216159417671, 0.4124834629079683, 0.3886835791259502, None)
'weighted': (0.4651150594652801, 0.4875259875259875, 0.4505998877514731, None)

2) {'C': 0.03}; 2 TF_IDF_KNN sentence vectors, their sum, their absolute difference
'micro': (0.581081081081081, 0.581081081081081, 0.581081081081081, None)
'macro': (0.5702479232859171, 0.509335614485112, 0.506266333983747, None)
'weighted': (0.5870345865482134, 0.581081081081081, 0.5600835516323056, None)

3) {'C': 0.1}; 2 TF_IDF_KNN sentence vectors, cosine distance, jaccard
'micro': (0.5878378378378378, 0.5878378378378378, 0.5878378378378378, None)
'macro': (0.5894425892425234, 0.5370340698582136, 0.545428278838417, None)
'weighted': (0.6048091355150441, 0.5878378378378378, 0.5810225706539071, None)

!!!
4) {'C': 0.03}; 2 TF_IDF_KNN sentence vectors, their sum, their absolute difference, cosine distance, jaccard
'micro': (0.5977130977130977, 0.5977130977130977, 0.5977130977130977, None)
'macro': (0.6053451753141013, 0.5433699536093627, 0.5521532896575675, None)
'weighted': (0.6148984548011244, 0.5977130977130977, 0.5887866049581948, None)

5) {'C': 0.1}; 2 TF_IDF_KLD_KNN sentence vectors, their sum, their absolute difference, cosine distance, jaccard
'micro': (0.5691268191268192, 0.5691268191268192, 0.5691268191268192, None)
'macro': (0.6274585384582284, 0.5098063967458056, 0.5151494021783734, None)
'weighted': (0.6205205165867679, 0.5691268191268192, 0.552078809047006, None)

6) {'C': 0.01}; 2 TF_KLD_KNN (unk=1) sentence vectors, their sum, their absolute difference, cosine distance, jaccard
'micro': (0.5654885654885655, 0.5654885654885655, 0.5654885654885655, None)
'macro': (0.6125495141634353, 0.49857312268139115, 0.49877805606347225, None)
'weighted': (0.6068771483841698, 0.5654885654885655, 0.5443918375584615, None)
'''

import math
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = 'news_mystem_skipgram_1000_20_2015.bin.gz'

model = KeyedVectors.load_word2vec_format(MODEL_NAME, binary=True)

train_data_file = 'paraphraser/paraphrases_lemmas.txt'
train_labels_file = 'paraphraser/labels.txt'
train_file_pairs = 7227
test_data_file = 'paraphraser_gold/paraphrases_gold_lemmas.txt'
test_labels_file = 'paraphraser_gold/labels.txt'
test_file_pairs = 1924


class SentenceVectorBuilder:
    def __init__(self, filename, labels_file, pairs, weights):
        self.data_file = filename
        self.labels_file = labels_file
        self.pairs = pairs
        self.weights = weights
        self.Y = []

    def get_labels(self):
        return self.Y

    def fit_transform(self):
        res = []
        with open(self.data_file, 'r', encoding='utf8') as inf1, open(self.labels_file, 'r', encoding='utf8') as inf2:
            for i in range(self.pairs):
                s1 = inf1.readline().strip().split()
                s2 = inf1.readline().strip().split()
                v1 = get_sent_vector_tfidf(s1, self.weights, 2*i)
                v2 = get_sent_vector_tfidf(s2, self.weights, 2*i + 1)
                vpositive = v1 + v2
                vnegative = np.absolute(v1-v2)
                c = 1 - cosine(v1, v2)
                jac = get_jaccard_index(s1, s2)
                self.Y.append(int(inf2.readline()))
                res.append(np.hstack((v1, v2, vpositive, vnegative, c, jac)))
        return pd.DataFrame(data=res)


def get_jaccard_index(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


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
    global model, train_TFIDF, FEATURE_NAMES
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


if __name__ == '__main__':
    train = SentenceVectorBuilder(train_data_file, train_labels_file, train_file_pairs, train_TFIDF)
    X_train = train.fit_transform()
    Y_train = train.get_labels()

    param_grid = [
        {'C': [0.03], 'dual': [False],
         'random_state': [42], 'max_iter': [1000]}
    ]

    clf = LinearSVC()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_micro')
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)

    del train, X_train, Y_train

    test = SentenceVectorBuilder(test_data_file, test_labels_file, test_file_pairs, test_TFIDF)
    X_test = test.fit_transform()
    Y_test = test.get_labels()

    Y_pred = grid_search.predict(X_test)
    print(precision_recall_fscore_support(Y_test, Y_pred, average='micro'))
    print(precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
    print(precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
