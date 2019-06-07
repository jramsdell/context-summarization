import json

from parsing.blobbing.blobber import Blobber, BigramBlobber


class TQAEnwikiAnalyzer(object):
    def __init__(self, loc, n_limit = -1):
        self.data = []
        self.blobber = Blobber()
        self.bigram_blobber = BigramBlobber()
        self.n_limit = n_limit

        with open (loc) as f:
            self.retrieve_data(f)

    def retrieve_data(self, f):
        counter = 0
        for line in f:
            if self.n_limit > 0:
                counter += 1
                if counter > self.n_limit:
                    break

            self.data.append(json.loads(line))


    def blob_by_label(self, label, array):
        for paragraph in array:
            self.blobber.learn_vocabulary(paragraph["text"], label, paragraph["pid"])
            if self.bigram_blobber is not None:
                self.bigram_blobber.learn_vocabulary(paragraph["text"], label, paragraph["pid"])


    def blobbify(self, vocab_index=None, bigram_index=None):
        for example in self.data:
            self.blob_by_label("tqa", example["tqa"])
            self.blob_by_label("enwiki", example["enwiki"])
            self.blob_by_label("negatives", example["negatives"])

        if vocab_index is not None:
            self.blobber.load_vocabulary_json(vocab_index)

        if bigram_index is not None:
            self.bigram_blobber.load_vocabulary_json(bigram_index)

        self.blobber.vectorize()
        if self.bigram_blobber is not None:
            self.bigram_blobber.vectorize()





