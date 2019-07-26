import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder
from analyzers.elmo_tqa_enwiki_analyzer import ElmoTqaEnwikiAnalyzer
from learning.models.bert_lstm_model import BertLSTMModel
from scipy.special import expit
from sklearn.model_selection import train_test_split
from learning.trainers.base_trainer import BaseTrainer
from learning.trainers.data_handler import DataHandler, DataPolicy
from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from parsing.qrel_reader import QrelReader
import numpy as np


class ElmoRedux(object):
    def __init__(self, loc, array_loc: str, qrel_loc: str):
        self.query_parser = QrelReader(qrel_loc)
        print("Retrieving vectors")
        input = { PipeEnum.IN_LOC.value : array_loc }
        self.vector_reader = NumpyArrayPipelineReader(**input)
        self.vector_reader.run()

        self.labels = []
        self.positives = []
        self.negatives = []











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
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs-elmo_embedding-ndarray.npy"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    # manager = ElmoManager(loc, load_bert=True)
    manager = ElmoRedux(loc,
                        array_loc=array_loc,
                        qrel_loc=qrel_loc)
    # manager.run_test()
    manager.load_test()

