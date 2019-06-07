import json

from analyzers.tqa_enwiki_analyzer import TQAEnwikiAnalyzer
import numpy as np
import os
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
from random import shuffle

from learning.models.simple_text_logistic_model import SimpleTextLogisticModel
from learning.models.simple_text_model import SimpleTextModel
from learning.models.weight_estimator import WeightEstimatorModel
from learning.trainers.base_trainer import BaseTrainer
from learning.trainers.data_handler import DataHandler, DataPolicy
from learning.trainers.layered_trainer import LayeredLearner
from learning.trainers.simple_trainer import SimpleTrainer
from sklearn.preprocessing import normalize


class RougeParser(object):
    def __init__(self, data_loc):

        analyzer = TQAEnwikiAnalyzer(data_loc)
        # vocab_index = "/home/jsc57/projects/context_summarization/parsing/unigram_vocab_index.json"
        vocab_index = "/home/jsc57/projects/context_summarization/parsing/unigram_vocab_index.json"
        bigram_vocab_index = "/home/jsc57/projects/context_summarization/parsing/bigram_vocab_index.json"

        # analyzer.blobbify(vocab_index=vocab_index, bigram_index=bigram_vocab_index)
        analyzer.blobbify(vocab_index=vocab_index, bigram_index=None)
        # embedders = list(analyzer.blobber.doc_vectors['enwiki'].values())[90:110] + list(analyzer.blobber.doc_vectors['negatives'].values())[40:50]

        # embedders = list(analyzer.blobber.doc_vectors['enwiki'].values())[90:110]
        unigram_embedders = sparse.load_npz("/home/jsc57/projects/context_summarization/parsing/unigram_sparse_embeddings.npz")
        # bigram_embedders = sparse.load_npz("/home/jsc57/projects/context_summarization/parsing/bigram_sparse_embeddings.npz")



        print("Parsing unigrams")
        unigram_contexts, unigram_tqas, unigram_samples, unigram_labels = self.samples_to_matrices(analyzer, unigram_embedders, analyzer.blobber)

        # print("Parsing bigrams")
        # bigram_contexts, bigram_tqas, bigram_samples, bigram_labels = self.samples_to_matrices(analyzer, bigram_embedders, analyzer.bigram_blobber)

        # tqas = np.concatenate([unigram_tqas, bigram_tqas], 1)
        # samples = np.concatenate([unigram_samples, bigram_samples], 1)

        tqas = unigram_tqas
        contexts = unigram_contexts
        samples = unigram_samples




        # self.train(tqas, samples, labels)

        self.contexts_train, self.contexts_test, \
        self.tqas_train, self.tqas_test, \
        self.samples_train, self.samples_test, \
        self.labels_train, self.labels_test = train_test_split(contexts, tqas, samples, unigram_labels,
                                                              test_size = 0.33, random_state = 42)



        if not os.path.exists("save"):
            os.mkdir("save")
        np.save("save/contexts_train.npy", self.contexts_train)
        np.save("save/tqas_train.npy", self.tqas_train)
        np.save("save/labels_train.npy", self.labels_train)
        np.save("save/samples_train.npy", self.samples_train)

        np.save("save/contexts_test.npy", self.contexts_test)
        np.save("save/tqas_test.npy", self.tqas_test)
        np.save("save/labels_test.npy", self.labels_test)
        np.save("save/samples_test.npy", self.samples_test)



    def samples_to_matrices(self, analyzer, embedders, blobber):
        all_tqas = []
        all_samples = []
        all_labels = []
        all_contexts = []
        dv = blobber.doc_vectors


        for i in range(len(analyzer.data)):
            data = analyzer.data[i]
            if not data['tqa'] or len(data['negatives']) < 2 or  len(data['enwiki']) < 2:
                continue

            tqa_embedding, samples, labels = self.construct_training_example(data, embedders, dv)
            tqas = np.tile(tqa_embedding, (samples.shape[1], 1)).T

            context = " ".join(data['context'])
            context = blobber.get_freq_map(context)
            context_vector = blobber.dv.transform(context)
            context_embedding = self.embedding(embedders, context_vector)
            context = np.tile(context_embedding, (samples.shape[1], 1)).T


            all_tqas.append(tqas)
            all_samples.append(samples)
            all_labels.append(labels)
            all_contexts.append(context)



        new_contexts = [self.scale(i) for i in all_contexts ]
        new_samples = [self.scale(i) for i in all_samples ]
        new_tqas = [self.scale(i) for i in all_tqas ]


        divided_by_query = np.asarray([new_contexts, new_tqas, new_samples, all_labels])
        np.save("save/divided_by_query", divided_by_query)


        samples = np.concatenate(all_samples, axis=1)
        samples = self.scale(samples)

        # scale = StandardScaler().fit(samples)
        # samples = scale.transform(samples)
        # samples = samples * self.unigram_eigens


        tqas = np.concatenate(all_tqas, axis=1)
        tqas = self.scale(tqas)

        # scale2 = StandardScaler().fit(tqas)
        # tqas = scale2.transform(tqas)

        contexts = np.concatenate(all_contexts, axis=1)
        contexts = self.scale(contexts)

        # scale3 = StandardScaler().fit(contexts)
        # contexts = scale3.transform(contexts)

        labels = np.concatenate(all_labels)
        labels = np.expand_dims(labels, 1)

        print(labels.shape, tqas.shape, samples.shape)
        return contexts.T, tqas.T, samples.T, labels


    def scale(self, xs):
        scaler = StandardScaler().fit(xs)
        return scaler.transform(xs)





    def construct_training_example(self, data, embedders, dv):
        tqa_pid = data['tqa'][0]['pid']
        enwiki_pids = [i['pid'] for i in data['enwiki']]
        negative_pids = [i['pid'] for i in data['negatives']]

        tqa_vec = dv['tqa'][tqa_pid]
        enwiki_vecs = [dv['enwiki'][i] for i in enwiki_pids ]
        negative_vecs = [dv['negatives'][i] for i in negative_pids ]

        tqa_embedding = self.embedding(embedders, tqa_vec)


        # enwiki_embeddings = np.asarray([self.embedding(embedders, i) for i in enwiki_vecs])
        enwiki_embeddings = self.get_product(embedders, enwiki_vecs)
        negative_embeddings = self.get_product(embedders, negative_vecs)

        # negative_embeddings = np.asarray([self.embedding(embedders, i) for i in negative_vecs])

        # print("Enwiki: {}, negative: {}".format(enwiki_embeddings.shape, negative_embeddings.shape))

        samples = np.concatenate([enwiki_embeddings, negative_embeddings], axis=1)

        labels = np.concatenate([np.ones(enwiki_embeddings.shape[1]), np.zeros(negative_embeddings.shape[1])])

        return tqa_embedding, samples, labels


    def get_product(self, embedders, vecs):
        vecs = sparse.vstack(vecs)
        # print(vecs.shape)
        # print(embedders.shape)
        dproduct = embedders @ vecs.T
        # dproduct = embedders.dot(vecs.T)


        rows = []
        for i in range(dproduct.shape[0]):
            rows.append(dproduct[i].sum(0))


        fmatrix = np.squeeze(np.asarray(rows))
        # fmatrix = np.concatenate([fmatrix, self.unigram_eigens.T @ (fmatrix * fmatrix)])
        return fmatrix



    def embedding(self, embedders, vec):
        arr = np.asarray([(i @ vec.T).sum() for i in embedders])
        return arr



class RougeManager(object):
    def __init__(self):
        torch.manual_seed(10)
        self.contexts_train = np.load("save/contexts_train.npy")
        self.tqas_train = np.load("save/tqas_train.npy")
        self.labels_train = np.load("save/labels_train.npy")
        self.samples_train = np.load("save/samples_train.npy")

        self.contexts_test = np.load("save/contexts_test.npy")
        self.tqas_test = np.load("save/tqas_test.npy")
        self.labels_test = np.load("save/labels_test.npy")
        self.samples_test = np.load("save/samples_test.npy")
        self.prepare_test_train2()
        


    def prepare_test_train2(self):
        np.random.seed(10)
        dq = np.load("save/divided_by_query.npy")
        contexts, tqas, samples, labels = dq

        n_queries = contexts.shape[0]
        q_indices = list(range(n_queries))

        shuffle(q_indices)
        train_i = q_indices[10:]
        test_i = q_indices[:10]

        self.test_contexts, self.test_tqas, self.test_samples, self.test_labels = \
            contexts[test_i], tqas[test_i], samples[test_i], labels[test_i]
        self.train_contexts, self.train_tqas, self.train_samples, self.train_labels = \
            contexts[train_i], tqas[train_i], samples[train_i], labels[train_i]


        samples = []
        tqas = []
        labels = []
        contexts = []
        for i in range(self.train_samples.shape[0]):
            samples.append(self.train_samples[i].T)
            tqas.append(self.train_tqas[i].T)
            labels.append(self.train_labels[i])
            contexts.append(self.train_contexts[i].T)

        samples = np.concatenate(samples)
        labels = np.concatenate(labels)
        tqas = np.concatenate(tqas)
        contexts = np.concatenate(contexts)

        print("Training")
        torch.manual_seed(10)
        self.model = SimpleTextModel

        loss_function = torch.nn.BCELoss(reduction='mean')
        predictors = {"p1" : tqas, "p2" : samples, "c" : contexts}
        self.train_handler = DataHandler(predictors=predictors, response=labels, policy=DataPolicy.ALL_DATA)
        self.trainer = BaseTrainer(data_handler=self.train_handler, model=SimpleTextModel, loss_function=loss_function)
        # self.trainer = SimpleTrainer(self.samples_train, ys.view(-1, 1), self.model, 0.001)
        # self.trainer.train_text(self.tqas_train, self.contexts_train)
        self.trainer.train()


        samples = []
        tqas = []
        labels = []
        contexts = []
        for i in range(self.test_samples.shape[0]):
            samples.append(self.test_samples[i].T)
            tqas.append(self.test_tqas[i].T)
            labels.append(self.test_labels[i])
            contexts.append(self.test_contexts[i].T)

        samples = np.concatenate(samples)
        labels = np.concatenate(labels)
        tqas = np.concatenate(tqas)
        contexts = np.concatenate(contexts)

        predictors = {"p1" : tqas, "p2" : samples, "c" : contexts}
        handler = DataHandler(predictors=predictors, response=labels, policy=DataPolicy.ALL_DATA)

        results = self.trainer.model(handler)
        results = np.where(results > 0.5, 1.0, 0.0)
        acc = (np.squeeze(results) == labels).sum() / labels.shape[0]
        print(acc)








    def prepare_test_train(self):
        np.random.seed(10)
        dq = np.load("save/divided_by_query.npy")
        contexts, tqas, samples, labels = dq

        n_queries = contexts.shape[0]
        q_indices = list(range(n_queries))

        shuffle(q_indices)
        train_i = q_indices[20:]
        test_i = q_indices[:20]

        self.test_contexts, self.test_tqas, self.test_samples, self.test_labels = \
            contexts[test_i], tqas[test_i], samples[test_i], labels[test_i]
        self.train_contexts, self.train_tqas, self.train_samples, self.train_labels = \
            contexts[train_i], tqas[train_i], samples[train_i], labels[train_i]


        loss_function = torch.nn.BCELoss(reduction='mean')

        data_handlers = []


        for i in range(self.train_contexts.shape[0]):
            predictors = {"xs" : self.train_samples[i] }
            response = self.train_labels[i]
            data_handler = DataHandler(predictors=predictors, response=response, policy=DataPolicy.ALL_DATA)
            data_handlers.append(data_handler)


            # trainer = BaseTrainer(data_handler=data_handler, model=SimpleTextLogisticModel, loss_function=loss_function)
        lt = LayeredLearner(data_handlers=data_handlers, model=SimpleTextLogisticModel, loss_function=loss_function, trainer_class=BaseTrainer, weight_decay=0.05)


        weights = []
        for layer in lt.base_layers:
            w = layer.trainer.model.linear.weight.detach().numpy()
            # bias = layer.trainer.model.linear.bias.detach().numpy()
            # weights.append(np.concatenate([w, np.expand_dims(bias, 1)], 1))
            weights.append(w)

        # weights = [i.trainer.model.linear.weight.detach().numpy() for i in lt.base_layers]
        weights = np.concatenate(weights, 0)
        # ss = StandardScaler(with_mean=False).fit(weights)
        # weights = ss.transform(weights)

        contexts = [np.expand_dims(i.T[0], 1) for i in self.train_contexts]
        tqas = [np.expand_dims(i.T[0], 1) for i in self.train_tqas]
        squished_samples = [i.mean(1) for i in self.train_samples]

        # print(contexts[0].shape)
        mse = torch.nn.MSELoss(reduction='mean')


        # def my_loss(y_pred, y_actual):
        #     print(y_pred.shape)
        #     print(y_actual.shape)
        #     return (((y_pred - y_actual) ** 2).sum(1) ** 0.5).mean()

        predictors = {"p1" : np.hstack(tqas), "c" : np.hstack(contexts), "squished_samples" : np.vstack(squished_samples)}
        data_handler = DataHandler(predictors=predictors, response=weights, model_parameters={"n_weights" : weights.shape[1]}, response_shape=(weights.shape))

        trainer = BaseTrainer(data_handler=data_handler, model=WeightEstimatorModel, loss_function=mse, lr=0.01)
        trainer.train(n=2000)

        total_acc = 0.0
        total_counter = 0


        # predictors = {"p1" : tqas, "c" : contexts[0], "xs" : samples}
        # data_handler = DataHandler(predictors=predictors, response=weights, model_parameters={"n_weights" : weights.shape[1]})

        # m = WeightEstimatorModel(data_handler)
        # best_weights = trainer.model(data_handler).detach().numpy().T

        # print(((best_weights - weights.T) ** 2).sum(0) ** 0.5)

        for i in range(self.test_contexts.shape[0]):
        # for i in range(self.train_contexts.shape[0]):
            context = np.expand_dims(self.test_contexts[i].T[0], 1)
            tqa = np.expand_dims(self.test_tqas[i].T[0], 1)
            samples = self.test_samples[i]
            labels = self.test_labels[i]

            # context = np.expand_dims(self.train_contexts[i].T[0], 1)
            # tqa = np.expand_dims(self.train_tqas[i].T[0], 1)
            # samples = self.train_samples[i]
            # labels = self.train_labels[i]

            squished_samples = samples.mean(1)
            predictors = {"p1" : tqa, "c" : context, "xs" : samples, "squished_samples" : squished_samples}
            data_handler = DataHandler(predictors=predictors, response=weights, model_parameters={"n_weights" : weights.shape[1]})
            # m = WeightEstimatorModel(data_handler)
            m = trainer.model
            best_weights = m(data_handler).detach().numpy().T

            # bias = best_weights[-1]
            # best_weights = best_weights[0:best_weights.shape[0] - 1]

            lm = SimpleTextLogisticModel(data_handler)
            # lm.linear.bias = Parameter(Tensor(bias.T))

            lm.linear.weight = Parameter(Tensor(best_weights.T))

            results = lm(data_handler)
            results = np.where(results > 0.5, 1.0, 0.0)
            acc = (labels == np.squeeze(results)).sum() / (labels.shape[0])
            total_acc += acc
            total_counter += 1

        print("Final ACC: {}".format(total_acc / total_counter))













    def train(self):
        print("Training")
        torch.manual_seed(10)
        self.model = SimpleTextModel
        ys = Tensor(self.labels_train)

        loss_function = torch.nn.BCELoss(reduction='mean')
        predictors = {"p1" : self.tqas_train, "p2" : self.samples_train, "c" : self.contexts_train}
        self.train_handler = DataHandler(predictors=predictors, response=self.labels_train, policy=DataPolicy.ALL_DATA)
        self.trainer = BaseTrainer(data_handler=self.train_handler, model=SimpleTextModel, loss_function=loss_function)
        # self.trainer = SimpleTrainer(self.samples_train, ys.view(-1, 1), self.model, 0.001)
        # self.trainer.train_text(self.tqas_train, self.contexts_train)
        self.trainer.train()



    def test(self):
        predictors = {"p1" : self.tqas_test, "p2" : self.samples_test, "c" : self.contexts_test}
        self.test_handler = DataHandler(predictors=predictors, response=self.labels_test, policy=DataPolicy.ALL_DATA)
        # results = self.trainer.model(Tensor(self.tqas_test), Tensor(self.contexts_test), Tensor(self.samples_test)).detach().numpy()
        results = self.trainer.model(self.test_handler).detach().numpy()
        results = np.where(results > 0.5, 1.0, 0.0)
        acc = (results == self.labels_test).sum() / self.labels_test.shape[0]
        print(acc)


if __name__ == '__main__':
    loc = "/home/jsc57/projects/context_summarization/managers/training_data.jsonl"
    # parser = RougeParser(loc)

    # manager = RougeManager(loc, from_save=0)
    manager = RougeManager()
    # manager.train()
    # manager.test()
