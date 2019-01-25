#!/usr/bin/python

# Baseline, LinearSVC with only Jaccard Index

# f1-micro=0.5561330561330561
# f1-macro=0.46916754197910326
# f1-weighted=0.5245356730183268

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

train_data_file = 'paraphraser/paraphrases_lemmas.txt'
train_labels_file = 'paraphraser/labels.txt'
train_file_pairs = 7227
test_data_file = 'paraphraser_gold/paraphrases_gold_lemmas.txt'
test_labels_file = 'paraphraser_gold/labels.txt'
test_file_pairs = 1924


class SentenceVectorBuilder:
    def __init__(self, filename, labels_file, pairs):
        self.data_file = filename
        self.labels_file = labels_file
        self.pairs = pairs
        self.Y = []

    def get_labels(self):
        return self.Y

    @staticmethod
    def get_jaccard_index(s1, s2):
        set1 = set(s1)
        set2 = set(s2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def fit_transform(self):
        res = {'Jaccard index': []}
        with open(self.data_file, 'r', encoding='utf8') as inf1, open(self.labels_file, 'r', encoding='utf8') as inf2:
            for i in range(self.pairs):
                s1 = inf1.readline().strip().split()
                s2 = inf1.readline().strip().split()
                jac = self.get_jaccard_index(s1, s2)
                self.Y.append(int(inf2.readline()))
                res['Jaccard index'].append(jac)
        return pd.DataFrame(data=res)


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


if __name__ == '__main__':
    train = SentenceVectorBuilder(train_data_file, train_labels_file, train_file_pairs)
    X_train = train.fit_transform()
    Y_train = train.get_labels()

    test = SentenceVectorBuilder(test_data_file, test_labels_file, test_file_pairs)
    X_test = test.fit_transform()
    Y_test = test.get_labels()

    clf = LinearSVC(dual=False, random_state=42)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    print(precision_recall_fscore_support(Y_test, Y_pred, average='micro'))
    print(precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
    print(precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
