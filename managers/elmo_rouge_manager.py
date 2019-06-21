import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder
from analyzers.elmo_tqa_enwiki_analyzer import ElmoTqaEnwikiAnalyzer
from learning.models.bert_lstm_model import BertLSTMModel
from scipy.special import expit
from sklearn.model_selection import train_test_split
from learning.trainers.base_trainer import BaseTrainer
from learning.trainers.data_handler import DataHandler, DataPolicy


class ElmoManager(object):
    def __init__(self, loc, load_bert=True):
        if load_bert:
            self.max_words = 200
            self.analyzer = ElmoTqaEnwikiAnalyzer(loc, self.max_words)
            self.analyzer.elmotize()
            self.model = ElmoEmbedder()
            self.embeddings = {}
            self.max_sentences = 4
            self.embedding_size = 1024



    def get_embedding(self, p1):
        # return [self.model.embed_sentence(i) for i in p1["tokens"]]
        sentences =  [i[0] for i in self.model.embed_sentences(p1["tokens"][0:self.max_sentences])]
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            # if sentence.shape[0] < self.max_words:
            #     word_diff = self.max_words - sentence.shape[0]
            #     zshape = (word_diff, sentence.shape[1])
            #     sentence = np.concatenate([sentence, np.zeros(zshape)], 0)
            sentences[idx] = sentence.mean(0)

        sentences = np.asarray(sentences)

        if sentences.shape[0] < self.max_sentences:
            sentence_diff = self.max_sentences - sentences.shape[0]
            # zshape = (sentence_diff, self.max_words, self.embedding_size)
            zshape = (sentence_diff, self.embedding_size)
            sentences = np.concatenate([sentences, np.zeros(zshape)], 0)


        # return np.asarray(sentences)
        return sentences



    def run_test(self):
        t = len(self.analyzer.data)
        tqa_matrix = []
        sample_matrix = []
        label_matrix = []
        for idx, example in enumerate(self.analyzer.data):
            print("{} for {}".format(idx, t))
            # if idx > 1:
            #     break
            try:
                qid = example["qid"]
                qd = self.analyzer.bert_data[qid]
                tqa = qd["tqa"][0]
                tqa_embedding = self.get_embedding(tqa)

                for p in qd["enwiki"]:
                    embedding = self.get_embedding(p)
                    tqa_matrix.append(tqa_embedding)
                    label_matrix.append(1)
                    sample_matrix.append(embedding)

                for p in qd["negatives"]:
                    embedding = self.get_embedding(p)
                    tqa_matrix.append(tqa_embedding)
                    label_matrix.append(0)
                    sample_matrix.append(embedding)
            except RuntimeError:
                print("Error")
            except KeyError:
                print("Key Error")



        tqa_matrix = np.asarray(tqa_matrix)
        sample_matrix = np.asarray(sample_matrix)
        label_matrix = np.asarray(label_matrix)

        np.save("elmo_tqa.npy", tqa_matrix)
        np.save("elmo_sample.npy", sample_matrix)
        np.save("elmo_label.npy", label_matrix)



    def load_test(self):
        tqa_matrix = np.load("elmo_tqa.npy")
        sample_matrix = np.load("elmo_sample.npy")
        label_matrix = np.load("elmo_label.npy")

        tqas_train, tqas_test, \
        samples_train, samples_test, \
        labels_train, labels_test = train_test_split(tqa_matrix, sample_matrix, label_matrix,
                                                               test_size = 0.05, random_state = 422)


        # labels_train = np.where(labels_train == 0, -1, 1)
        # labels_test = np.where(labels_test == 0, -1, 1)


        model = BertLSTMModel
        loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.train_handler = DataHandler(predictors={'p1' : tqas_train, 'p2' : samples_train}, response=labels_train, policy=DataPolicy.ALL_DATA)
        self.trainer = BaseTrainer(data_handler=self.train_handler, model=model, loss_function=loss_function, lr=0.001)

        for i in range(10):
            self.trainer.model.is_train = True
            self.trainer.train(weight_decay=0.0000, n=5)
            self.trainer.model.is_train = False



            self.train_handler2 = DataHandler(predictors={'p1' : tqas_test, 'p2' : samples_test}, response=labels_test, policy=DataPolicy.ALL_DATA)


            results = self.trainer.model(self.train_handler2).detach().numpy()
            results = expit(results)
            results = np.where(results >= 0.5, 1, 0)
            results = np.squeeze(results)

            acc = (np.squeeze(results) == np.squeeze(labels_test)).sum() / labels_test.shape[0]
            print("Acc Test: {}".format(acc))

            self.train_handler2 = DataHandler(predictors={'p1' : tqas_train, 'p2' : samples_train}, response=labels_train, policy=DataPolicy.ALL_DATA)


            tp = (((results == 1) * (labels_test == 1)).sum())
            tn = ((results == 0) * (labels_test == 0)).sum()
            fn = ((results == 0) * (labels_test == 1)).sum()
            fp = ((results == 1) * (labels_test == 0)).sum()
            print("tp: {}, tn: {}, fn: {}, fp: {}, recall: {}, precision: {}".format(tp, tn, fn, fp, tp / (tp + fn), tp / (tp + fp)))


            results = self.trainer.model(self.train_handler2).detach().numpy()
            results = expit(results)
            results = np.where(results >= 0.5, 1, 0)

            acc = (np.squeeze(results) == np.squeeze(labels_train)).sum() / labels_train.shape[0]
            print("Acc Train: {}".format(acc))




            # results = np.where(results > 0.5, 1, 1)
            #
            # acc = (np.squeeze(results) == np.squeeze(labels_test)).sum() / labels_test.shape[0]
            # print(acc)
            #
            # results = np.where(results > 0.5, 0, 0)
            #
            # acc = (np.squeeze(results) == np.squeeze(labels_test)).sum() / labels_test.shape[0]
            # print(acc)

        torch.save(self.trainer.model, "elmo_model")












if __name__ == '__main__':
    loc = "/home/jsc57/projects/context_summarization/managers/training_data.jsonl"
    # manager = ElmoManager(loc, load_bert=True)
    manager = ElmoManager(loc, load_bert=False)
    # manager.run_test()
    manager.load_test()

