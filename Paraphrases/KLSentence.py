#!/usr/bin/python

# Spearman correlation with labels and cosines between sentence vectors (KL-div)
import math
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from collections import OrderedDict

# 'ruscorpora_mystem_cbow_300_2_2015.bin.gz', 32739 absent words
# test (correlation=0.5200569713539533, pvalue=8.589539609347886e-134)
# train (correlation=0.65219781441631, pvalue=0.0)

# 'web_mystem_skipgram_500_2_2015.bin.gz', 31881  absent words
# test (correlation=0.524022590586699, pvalue=3.5915018048027524e-136)
# train (correlation=0.6540150571192944, pvalue=0.0)

# 'news_mystem_skipgram_1000_20_2015.bin.gz',  31780 absent words
# test (correlation=0.533170501506163, pvalue=8.833215803914221e-142)
# train (correlation=0.6630961335987, pvalue=0.0)

MODEL_NAME = 'news_mystem_skipgram_1000_20_2015.bin.gz'

model = KeyedVectors.load_word2vec_format(MODEL_NAME, binary=True)


class FreqDict:
    def __init__(self, filename, labels_file, threshold=0):
        self.filename = filename
        self.labels_file = labels_file
        self.threshold = threshold
        self.dict = {}
        self.kl_div_dict = {}

    def create_freq_dict(self):
        if self.dict:
            return self.dict
        with open(self.filename, 'r', encoding='utf8') as lines:
            for line in lines:
                for w in line.strip().split():
                    self.dict[w] = self.dict.setdefault(w, 0) + 1
        self.dict = {i: self.dict[i] for i in self.dict if self.dict[i] >= self.threshold}
        return self.dict

    def create_kl_div_dict(self):
        if self.kl_div_dict:
            return self.kl_div_dict
        total_vol = [0, 0]

        with open(self.filename, 'r', encoding='utf8') as inf1, open(self.labels_file, 'r') as inf2:
            n = 0
            for c in inf2:
                c = c.strip()
                # Classes (0 and 1 for [1]) and (-1 for [0])
                if c == '-1':
                    i = 0
                else:
                    i = 1
                total_vol[i] += 1

                s1 = inf1.readline().split()
                s2 = inf1.readline().split()

                set_new = set(s1).intersection(set(s2))
                has = False
                for w in set_new:
                    if w in model.wv:
                        has = True
                        break

                if len(set_new) != 0 and has:
                    n += 1

                tmp = []
                for w in s1:
                    if w in s2 and w not in tmp:
                        tmp.append(w)
                        self.kl_div_dict[w] = self.kl_div_dict.setdefault(w, [0, 0])
                        self.kl_div_dict[w][i] += 1

        self.kl_div_dict = OrderedDict({i: self.kl_div_dict[i] for i in self.kl_div_dict
                                        if self.kl_div_dict[i][0] >= self.threshold and self.kl_div_dict[i][
                                            1] >= self.threshold})

        for key, value in self.kl_div_dict.items():
            self.kl_div_dict[key] = kl_div([value[0]/sum(value), value[1]/sum(value)])
        print(n)

        return self.kl_div_dict


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


def kl_div(ps):
    arg1 = math.log(ps[0]/ps[1], math.e)
    arg2 = math.log((1 - ps[0])/(1 - ps[1]), math.e)
    return ps[0] * arg1 + (1 - ps[0]) * arg2


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


def get_sent_vector_tfidf_kldiv(s, matrix, i, kl_dict):
    global model, FEATURE_NAMES
    vecs = []
    coefs_tfi_df = []
    coef_kld = []
    res = []
    for w in s:
        if w not in model.wv:
            continue
        if w in FEATURE_NAMES:
            vecs.append(model[w])
            coefs_tfi_df.append(matrix[i, FEATURE_NAMES.index(w)])
            if w in kl_dict.keys():
                coef_kld.append(kl_dict[w])
            else:
                coef_kld.append(0.001)
        else:
            synonyms = get_synonyms(w)
            if len(synonyms) != 0:
                vecs.append(model[w])
                tfidf_syn = []
                kls_syn = []
                for syn in synonyms:
                    tfidf_syn.append(unk_tfidf(w, s, train_TFIDF[:, FEATURE_NAMES.index(syn)]))
                    if syn in kl_dict.keys():
                        kls_syn.append(kl_dict[syn])
                    else:
                        kls_syn.append(0.001)
                coefs_tfi_df.append(sum(tfidf_syn)/len(tfidf_syn))
                coef_kld.append(sum(kls_syn)/len(kls_syn))

    for i in range(len(coefs_tfi_df)):
        res.append(vecs[i] * (coef_kld[i] / sum(coef_kld)))
    return sum(res)


def main():
    print("TF-IDF-KLD")

    pars = FreqDict('paraphraser\paraphrases_lemmas.txt', 'paraphraser\labels.txt', threshold=1)
    kld_dict = pars.create_kl_div_dict()

    # test corpus
    with open('paraphraser_gold\paraphrases_gold_lemmas.txt', 'r', encoding='utf8') as inf1, \
            open('paraphraser_gold\labels.txt', 'r', encoding='utf8') as inf2:
        labels = []
        cosines = []
        for i in range(1924):
            s1 = inf1.readline()
            s2 = inf1.readline()
            v1 = get_sent_vector_tfidf_kldiv(s1.split(), test_TFIDF, 2 * i, kld_dict)
            v2 = get_sent_vector_tfidf_kldiv(s2.split(), test_TFIDF, 2 * i + 1, kld_dict)
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
            v1 = get_sent_vector_tfidf_kldiv(s1.split(), train_TFIDF, 2 * i, kld_dict)
            v2 = get_sent_vector_tfidf_kldiv(s2.split(), train_TFIDF, 2 * i + 1, kld_dict)
            c = 1 - cosine(v1, v2)
            label = int(inf2.readline())
            cosines.append(c)
            labels.append(label)

        p = spearmanr(np.array(labels), np.array(cosines))
        print(p)


if __name__ == '__main__':
    main()
