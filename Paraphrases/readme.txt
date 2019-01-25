1) Все 3 модели, используемые в скриптах, были взяты с сайта rusvectores.org:
ruscorpora_mystem_cbow_300_2_2015.bin.gz
web_mystem_skipgram_500_2_2015.bin.gz
news_mystem_skipgram_1000_20_2015.bin.gz

2) mystem opitons: -cdli

3) Ссылки на статьи с TF-KLD-весами для слов (несимметричный KL-divergence для двух распределений Бернулли)
https://www.cc.gatech.edu/~jeisenst/papers/ji-emnlp-2013.pdf
http://www.aclweb.org/anthology/N15-1154

4) Для незнакомых слов используются TF-KLD-KNN веса. Для них берётся средний вес тех k ближайших слов, который есть
в word2vec-модели

5) Для определения схожести векторов используется косинусная мера

6) Ссылки на презентации с основной информацией о задаче и об этой работе, в том числе дальнейшие планы
Проноза Е.В.    https://drive.google.com/open?id=1IRweZyKuaFq6LHVqvPlJx7iqZNcNg6qs
Тимохов В.В.    https://drive.google.com/open?id=1rfjJNszcA3YHXU_JKwdE8SPRk2o5YCfS

7) Для классификации использовался метод опорных векторов (LinearSVC из библиотеки sklearn)

8) Дополнительные метрики из статьи на будущее
1 unigram recall
2 unigram precision
3 bigram recall
4 bigram precision
5 dependency relation recall
6 dependency relation precision
7 BLEU recall
8 BLEU precision
9 Difference of sentence length
10 Tree-editing distance
