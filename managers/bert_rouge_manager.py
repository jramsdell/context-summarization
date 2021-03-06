import math

import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
from scipy.special import expit
from sklearn.model_selection import train_test_split

from analyzers.bert_tqa_enwiki_analyzer import BertTqaEnwikiAnalyzer
from learning.models.bert_lstm_model import BertLSTMModel
from learning.trainers.base_trainer import BaseTrainer
from learning.trainers.data_handler import DataHandler, DataPolicy


class BertManager(object):
    def __init__(self, loc, load_bert=True):
        if load_bert:
            self.analyzer = BertTqaEnwikiAnalyzer(loc)
            self.analyzer.bertanize()
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.model.eval()
            self.embeddings = {}







    def get_bert_logits(self, p1, p2):
        segments = np.concatenate([np.zeros(len(p1), dtype=np.long), np.ones(len(p2), dtype=np.long)])
        segments = torch.tensor([segments])
        token_tensor = torch.tensor([p1 + p2])
        with torch.no_grad():
            pred = self.model(token_tensor, segments)
        return pred

    def get_embedding(self, p1):
        # print(p1)
        # print(self.analyzer.tokenizer.convert_ids_to_tokens(p1))
        token_tensors = [i for i in p1["tokens"]]
        token_tensors = torch.tensor(token_tensors)



        with torch.no_grad():
            # return self.model(torch.tensor([p1]))[1].shape
            t =  self.model(token_tensors)[1].detach().numpy()
            sdiff = 4 - len(p1["tokens"])
            if sdiff > 0:
                t = np.concatenate([t, np.zeros((sdiff, t.shape[1]))])
            return t



    def get_label(self, p1, p2):
        try:
            logits = self.get_bert_logits(p1, p2)
            return 1 if logits[0][0] > logits[0][1] else 0
        except:
            print("Error")
            return -1


    def get_accuracy(self):
        total_acc = 0.0
        counter = 0
        t = len(self.analyzer.data)
        for idx, example in enumerate(self.analyzer.data):
            print("{} for {}".format(idx, t))
            qid = example["qid"]
            qd = self.analyzer.bert_data[qid]
            tqa = qd["tqa"][0]

            for p in qd["enwiki"]:
                prediction = self.get_label(tqa, p)
                if prediction == -1:
                    continue
                elif prediction == 1:
                    total_acc += 1
                counter += 1.0


            for p in qd["negatives"]:
                prediction = self.get_label(tqa, p)
                if prediction == -1:
                    continue
                elif prediction == 0:
                    total_acc += 1
                counter += 1.0

            if counter != 0:
                print("Final ACC: {}".format(total_acc / counter))


        print("Final ACC: {}".format(total_acc / counter))


    def run_test(self):
        t = len(self.analyzer.data)
        tqa_matrix = []
        sample_matrix = []
        label_matrix = []
        for idx, example in enumerate(self.analyzer.data):
            print("{} for {}".format(idx, t))
            try:
                qid = example["qid"]
                qd = self.analyzer.bert_data[qid]
                tqa = qd["tqa"][0]
                tqa_embedding = self.get_embedding(tqa)

                for p in qd["enwiki"]:
                    embedding = self.get_embedding(p)
                    tqa_matrix.append(tqa_embedding)

                    # rel = p["rel"]

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

        np.save("bert_tqa.npy", tqa_matrix)
        np.save("bert_sample.npy", sample_matrix)
        np.save("bert_label.npy", label_matrix)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def load_test(self):
        tqa_matrix = np.load("bert_tqa.npy")
        sample_matrix = np.load("bert_sample.npy")
        label_matrix = np.load("bert_label.npy")

        tqas_train, tqas_test, \
        samples_train, samples_test, \
        labels_train, labels_test = train_test_split(tqa_matrix, sample_matrix, label_matrix,
                                                               test_size = 0.20, random_state = 422)


        # labels_train = np.where(labels_train == 0, -1, 1)
        # labels_test = np.where(labels_test == 0, -1, 1)


        model = BertLSTMModel
        loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.train_handler = DataHandler(predictors={'p1' : tqas_train, 'p2' : samples_train}, response=labels_train, policy=DataPolicy.ALL_DATA)
        self.trainer = BaseTrainer(data_handler=self.train_handler, model=model, loss_function=loss_function, lr=0.001)

        for i in range(200):
            self.trainer.model.is_train = True
            self.trainer.train(weight_decay=0.0000, n=100)
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












if __name__ == '__main__':
    loc = "/home/jsc57/projects/context_summarization/managers/training_data.jsonl"
    manager = BertManager(loc, load_bert=False)
    # manager = BertManager(loc, load_bert=True)
    # manager.get_accuracy()
    # manager.run_test()
    manager.load_test()
