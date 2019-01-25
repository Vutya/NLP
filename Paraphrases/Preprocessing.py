#!/usr/bin/python

import re
from bs4 import BeautifulSoup


def xml_to_txt(xmlname, txtname):
    with open(xmlname, 'r', encoding='utf8') as inf:
        xml_doc = inf.read()
    soup = BeautifulSoup(xml_doc, 'html.parser')
    with open(txtname, 'w', encoding='utf8') as ouf:
        for val in soup.find_all('value'):
            if val.get('name') == 'text_1' or val.get('name') == 'text_2':
                ouf.write(val.string + '\n')


def get_labels(xmlname, labelsname):
    with open(xmlname, 'r', encoding='utf8') as inf:
        xml_doc = inf.read()
    soup = BeautifulSoup(xml_doc, 'html.parser')
    with open(labelsname, 'w', encoding='utf8') as ouf:
        for val in soup.find_all('value'):
            if val.get('name') == 'class':
                ouf.write(val.string + '\n')


def parse_mystem(filein, fileout):
    with open(filein, 'r', encoding='utf8') as lines, open(fileout, 'w', encoding='utf8') as ouf:
        for line in lines:
            sent = line.split()
            out = []
            for word in sent:
                try:
                    word = re.split(r'[=,]', word)
                    word = re.sub('[{}?,":»«.]', '', word[0] + '_' + word[1])
                    out.append(word)
                except IndexError:
                    continue
            ouf.write(' '.join(out) + '\n')


if __name__ == '__main__':
    xml_to_txt('paraphraser/paraphrases.xml', 'paraphraser/paraphrases_texts.txt')
    get_labels('paraphraser/paraphrases.xml', 'paraphraser/labels.txt')
    # parse_mystem(mystem file with -cdli options, file with lemmas and POS tags)
