from random import shuffle
import numpy as np
import json
from random import seed
from scipy import sparse
from sklearn.preprocessing import normalize

from parsing.blobbing.blobber import Blobber, BigramBlobber


class ParagraphCorpusBlobber(object):
    def __init__(self, loc, n_picks=5000, n_dims=10):
        self.n_picks = n_picks
        self.paragraphs = []
        self.blobber = Blobber()
        self.bigram_blobber = BigramBlobber()
        self.n_dims = n_dims

        with open(loc) as f:
            self.retrieve_paragraph_jsons(f)

    def retrieve_paragraph_jsons(self, f):
        seed(11)
        random_picks = list(range(5000000))
        shuffle(random_picks)
        random_picks = set(random_picks[0:self.n_picks])

        counter = -1
        for line in f:
            counter += 1
            if counter not in random_picks:
                continue


            random_picks.remove(counter)

            # We've found everything we need
            if len(random_picks) == 0:
                break

            self.paragraphs.append(json.loads(line))


    def blobbify_paragraphs(self):
        for paragraph in self.paragraphs:
            text = paragraph["text"]
            pid = paragraph["pid"]
            self.blobber.learn_vocabulary(text, "text", pid)
            self.bigram_blobber.learn_vocabulary(text, "text", pid)

        self.blobber.vectorize()
        self.bigram_blobber.vectorize()
        self.paragraphs = self.blobber.doc_vectors["text"]
        self.bigram_paragraphs = self.bigram_blobber.doc_vectors["text"]




    def do_embedding(self, paragraphs, blobber, out_name, out_dims=40):
        embedder_indices = list(range(self.n_picks))
        shuffle(embedder_indices)
        embedder_indices = embedder_indices[0:self.n_dims]

        embedders = []
        plist = list(paragraphs.values())
        for i in embedder_indices:
            embedders.append(plist[i])

        embedders = sparse.vstack(embedders)
        normalize(embedders, axis=1, norm='l2', copy=False)
        ps = sparse.vstack(plist)
        normalize(ps, axis=1, norm='l2', copy=False)
        dproduct = embedders.dot(ps.T)

        rows = []
        for i in range(dproduct.shape[0]):
            rows.append(dproduct[i].sum(0))

        fmatrix = np.squeeze(np.asarray(rows))

        cov = fmatrix @ fmatrix.T
        lhv, eig, rhv = np.linalg.svd(cov)
        best = lhv.T[0:out_dims]

        new_sparse = []
        for b in best:
            first = b[0] * embedders[0]
            for idx, weight in enumerate(b[1:]):
                first += embedders[idx + 1] * weight
            new_sparse.append(first)

        new_sparse = sparse.vstack(new_sparse)
        wubba = new_sparse @ ps.T
        rows = []
        for i in range(wubba.shape[0]):
            rows.append(wubba[i].sum(0))

        fmatrix2 = np.squeeze(np.asarray(rows))
        print(fmatrix2.shape)
        sparse.save_npz(out_name + "_sparse_embeddings", new_sparse)
        blobber.save_vocabulary_json(out_name + "_vocab_index.json")

        np.save(out_name + "_eigens", eig[0:out_dims])

    # def embedding(self):
    #     embedder_indices = list(range(self.n_picks))
    #     shuffle(embedder_indices)
    #     embedder_indices = embedder_indices[0:self.n_dims]
    #
    #     embedders = []
    #     plist = list(self.paragraphs.values())
    #     for i in embedder_indices:
    #         embedders.append(plist[i])
    #
    #     embedders = sparse.vstack(embedders)
    #     normalize(embedders, axis=1, norm='l2', copy=False)
    #     ps = sparse.vstack(plist)
    #     normalize(ps, axis=1, norm='l2', copy=False)
    #     dproduct = embedders.dot(ps.T)
    #
    #     rows = []
    #     for i in range(dproduct.shape[0]):
    #         rows.append(dproduct[i].sum(0))
    #
    #     fmatrix = np.squeeze(np.asarray(rows))
    #
    #     cov = fmatrix @ fmatrix.T
    #     lhv, eig, rhv = np.linalg.svd(cov)
    #     best = lhv.T[0:40]
    #
    #     new_sparse = []
    #     for b in best:
    #         first = b[0] * embedders[0]
    #         for idx, weight in enumerate(b[1:]):
    #             first += embedders[idx + 1] * weight
    #         new_sparse.append(first)
    #
    #     new_sparse = sparse.vstack(new_sparse)
    #     wubba = new_sparse @ ps.T
    #     rows = []
    #     for i in range(wubba.shape[0]):
    #         rows.append(wubba[i].sum(0))
    #
    #     fmatrix2 = np.squeeze(np.asarray(rows))
    #     print(fmatrix2.shape)
    #     sparse.save_npz("sparse_embeddings", new_sparse)
    #     self.blobber.save_vocabulary_json("vocab_index.json")
    #
    #     np.save("eigens", eig)




















if __name__ == '__main__':
    loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
    # pblobber = ParagraphCorpusBlobber(loc, n_picks=50000, n_dims=100)
    pblobber = ParagraphCorpusBlobber(loc, n_picks=80000, n_dims=100)
    # pblobber = ParagraphCorpusBlobber(loc, n_picks=500, n_dims=10)
    pblobber.blobbify_paragraphs()
    pblobber.do_embedding(pblobber.paragraphs, pblobber.blobber, "unigram", out_dims=40)
    # pblobber.do_embedding(pblobber.bigram_paragraphs, pblobber.bigram_blobber, "bigram", out_dims=20)
