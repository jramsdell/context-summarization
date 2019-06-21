import json

from parsing.blobbing.blobber import Blobber, BigramBlobber


class ComparisonAnalyzer(object):
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

    def get_mappings(self, example):
        all = {}

        out = self.retrieve_by_label("p1", example["p1"])
        all[out[0]] = out[1]
        out = self.retrieve_by_label("positives", example["positives"])
        all[out[0]] = out[1]
        out = self.retrieve_by_label("negatives", example["negatives"])
        all[out[0]] = out[1]
        return all

    def retrieve_by_label(self, label, array):
        all = []
        for paragraph in array:
            freqs = self.blobber.get_freq_map(paragraph["text"])
            all.append([paragraph["pid"], freqs])
        return [label, all]


    def blobbify(self, vocab_index=None, bigram_index=None):
        if vocab_index is not None:
            self.blobber.load_vocabulary_json(vocab_index)

        t = len(self.data)
        counter = 0
        # p = Pool(40)
        # m = p.map(self.get_mappings, self.data, chunksize=4000)

        for example in self.data:
            counter += 1
            print("{} out of {}".format(counter, t))
            self.blob_by_label("p1", example["p1"])
            self.blob_by_label("positives", example["positives"])
            self.blob_by_label("negatives", example["negatives"])
            if counter > 100:
                break

        # for example in m:
        #     for (label, f_index) in example.items():
        #         for (pid, freqs) in f_index:
        #             self.blobber.store_label(label, pid, freqs)


        # wee = [i for i in m]


        # if bigram_index is not None:
        #     self.bigram_blobber.load_vocabulary_json(bigram_index)

        self.blobber.vectorize()
        # if self.bigram_blobber is not None:
        #     self.bigram_blobber.vectorize()





