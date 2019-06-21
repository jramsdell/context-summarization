import json
from collections import Counter
from typing import *

from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from textblob import TextBlob

from utilities import iterutils


class Blobber(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.stop_list = set(stopwords.words("english"))
        self.dv = DictVectorizer()
        self.vocab_index = {}
        self.index_counter = 0
        self.doc_index = {}
        self.doc_vectors = {}
        self.vocab_counter = Counter()
        self.loaded_vocab = False

    def parse_string(self, doc: str):
        words = TextBlob(doc.lower()).words
        if self.loaded_vocab:
            return [self.stemmer.stem(word) for word in words if word not in self.stop_list and word in self.vocab_index]
        else:
            return [self.stemmer.stem(word) for word in words if word not in self.stop_list]

    def get_freq_map(self, doc: str):
        parsed = self.parse_string(doc)
        return Counter(parsed)


    def store_label(self, label, id, freqs):
        if label not in self.doc_index:
            self.doc_index[label] = {}
        d_index = self.doc_index[label]
        d_index[id] = freqs


    def learn_vocabulary(self, text, label, id):
        if label not in self.doc_index:
            self.doc_index[label] = {}
        d_index = self.doc_index[label]
        if id in d_index:
            return

        parsed = self.get_freq_map(text)
        d_index[id] = parsed

        for word in parsed:
            if word.isalpha:
                self.vocab_counter[word] += 1
            if word.isalpha()  and word not in self.vocab_index:
                self.vocab_index[word] = self.index_counter
                self.index_counter += 1


    def filter_most_frequent(self):
        self.vocab_index = {}
        for idx, (k,v) in enumerate(self.vocab_counter.most_common(200)):
            self.vocab_index[k] = idx


    def vectorize(self):
        # if not self.loaded_vocab:
        #     self.filter_most_frequent()
        self.dv.fit([self.vocab_index])

        for (label, c_docs) in self.doc_index.items():
            if label not in self.doc_vectors:
                self.doc_vectors[label] = {}
            doc_vecs = self.doc_vectors[label]
            for (id, freqs) in c_docs.items():
                doc_vecs[id] = self.dv.transform(freqs)


    def save_vocabulary_json(self, out_loc):
        with open(out_loc, "w") as w:
            w.write(json.dumps(self.vocab_index))

    def load_vocabulary_json(self, in_loc):
        self.loaded_vocab = True
        with open(in_loc) as f:
            self.vocab_index = json.loads(f.read())



class BigramBlobber(Blobber):

    def parse_string(self, doc: str):
        words = TextBlob(doc.lower()).words
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_list]
        bigrams = []
        for (w1, w2) in iterutils.window(words):
            bigrams.append(w1 + w2)
        return bigrams




