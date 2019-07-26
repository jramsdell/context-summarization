from learning.trainers.data_handler import DataHandler
import random
import os
from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from random import shuffle

from managers.scoring.trec_eval_parameters import TrecEvalParameters
from parsing.qrel_reader import QrelReader
from parsing.run_ranking_parser import RunDirParser
from collections import defaultdict
import numpy as np
from typing import *
import torch
import json
import matplotlib.pyplot as plt


class TrecRunPreparer(object):
    def __init__(self, run_dir, qrel_loc, array_loc, model_loc, n_limit = 10):
        self.param_evaluators = TrecEvalParameters.generate_parameters()
        random.seed(213)
        self.evals = {}
        run_dir_parser = RunDirParser(run_dir)
        self.n_limit = n_limit
        print("Reading runs")
        self.qmap = run_dir_parser.run()
        print("Reading qrels")
        self.query_parser = QrelReader(qrel_loc)
        print("Retrieving run ids")
        self.pmap = self.retrieve_run_pids()
        input = { PipeEnum.IN_LOC.value : array_loc }
        print("Retrieving vectors")
        self.vector_reader = NumpyArrayPipelineReader(**input)
        self.vector_reader.run()
        print("Loading model")
        self.model = torch.load(model_loc)
        print("Done")
        self.score_map = {}
        # self.allowed = defaultdict(dict)
        # self.scorer = scorer

        # all_positives = []
        # for (_, rels) in self.query_parser.qrels.items():
        #     all_positives.extend(rels)
        #
        # all_negatives = []
        # for (_, rels) in self.query_parser.negative_qrels.items():
        #     all_negatives.extend(rels)
        #
        # shuffle(all_positives)
        # shuffle(all_negatives)
        # all_positives = all_positives[0:int((len(all_positives) / 2))]
        # all_negatives = all_negatives[0:int((len(all_negatives) / 2))]
        # self.all_negatives = set(all_negatives)
        # self.all_positives = set(all_positives)




    def retrieve_run_pids(self):
        pids = set()
        pmap = defaultdict(lambda: defaultdict(list))

        for (run_name,run) in self.qmap.items():
            for (query, run_lines) in run.items():
                counter = 0
                for run_line in run_lines:
                    pids.add(run_line.pid)
                    pmap[run_name][query].append(run_line.pid)
                    counter += 1
                    if counter >= self.n_limit:
                        break
        return pmap


    def build_score_map(self, run_pids):
        pid_map = {}
        counter = 0
        p1s = []
        p2s = []
        for (_, run) in self.pmap.items():

            for (query, retrieved_pids) in run.items():
                if query.split(":")[0] == "enwiki" or query not in self.query_parser.qrels:
                    continue
                gold_standard_pids = [i.pid for i in self.query_parser.qrels[query]]
                c2 = 0
                print(counter)

                for pid in [i for i in retrieved_pids]:
                    for gold_standard_pid in gold_standard_pids:
                        key = (gold_standard_pid, pid)
                        if key in pid_map:
                            continue

                        p1s.append(self.vector_reader.get_vector(gold_standard_pid))
                        p2s.append(self.vector_reader.get_vector(pid))
                        pid_map[key] = counter
                        counter += 1
                    c2 += 1
                    if c2 >= self.n_limit:
                        break


        p1s = np.asarray(p1s)
        p2s = np.asarray(p2s)
        print("Built!")

        predictors = {
            "p1" : p1s,
            "p2" : p2s
        }

        data_handler = DataHandler(predictors)

        scores = (torch.nn.Sigmoid()(self.model(data_handler))).detach().numpy()
        # plt.hist(scores, bins='auto')
        # plt.show()

        return pid_map, scores


    # def score_runs(self, pid_map, scores):
    def score_runs(self):
        # run_scores = {}
        # run_query_scores = {}
        for (run_name, run) in self.pmap.items():
            run_name = run_name.split("/")[-1]
            # run_score, query_scores = self.score_run(run_name, run, pid_map, scores)
            # run_score, query_scores = self.score_runs(run_name, run)
            self.score_run(run_name, run)
            # run_scores[run_name] = run_score
            # run_query_scores[run_name] = query_scores

        # return run_scores, run_query_scores

    # def score_run(self, run_name, run, pid_map, scores):
    def score_run(self, run_name, run):
        total = 0.0
        query_scores = {}
        for (query, retrieved_pids) in run.items():
            if query.split(":")[0] == "enwiki" or query not in self.query_parser.qrels:
                continue
            # query_score = self.score_query(run_name, query, retrieved_pids, pid_map, scores)
            query_score = self.score_query(run_name, query, retrieved_pids)
            total += query_score
            # query_scores[query] = float(query_score)
        # return total, query_scores

    # def score_query(self, run_name, query, retrieved_pids, pid_map, scores):
    def score_query(self, run_name, query, retrieved_pids):
        gold_standard_pids = [i.pid for i in self.query_parser.qrels[query]]
        negatives = [i.pid for i in self.query_parser.negative_qrels[query]]
        total = 0.0
        for gs in gold_standard_pids:
            seen = 0

            highest = 0.0
            for retrieved_pid in retrieved_pids:
                # in_positive = gs == retrieved_pid and gs in self.all_positives
                # in_negative = retrieved_pid in negatives and retrieved_pid in self.all_negatives
                in_positive = retrieved_pid in gold_standard_pids
                in_negative = retrieved_pid in negatives
                seen += 1
                if seen >= 100:
                    break


                key = (gs, retrieved_pid)
                if key not in self.score_map:
                    v1 = self.vector_reader.get_vector(gs)
                    v2 = self.vector_reader.get_vector(retrieved_pid)
                    predictors = {
                        "p1" : np.asarray([v1]),
                        "p2" : np.asarray([v2])
                    }

                    data_handler = DataHandler(predictors)

                    score = (torch.nn.Sigmoid()(self.model(data_handler))).detach().numpy()[0]
                    self.score_map[key] = score




                # if key in pid_map:
                    # highest = max(highest, scores[pid_map[key]])
                    # score = scores[pid_map[key]][0]
                score = self.score_map[key]
                if score > 0.5 and in_negative:
                    print("WOA")
                for param_evaluator in self.param_evaluators:
                    param_evaluator.update_map(seen, query, retrieved_pid, in_positive, in_negative, score)
                    param_evaluator.update_score(seen, run_name, query, retrieved_pid, in_positive, in_negative, score)
                    param_evaluator.update_score_max(seen, run_name, query, retrieved_pid, in_positive, in_negative, score)



                # highest += score
            # total += highest


        # return total / seen
        # return total / self.n_limit
        # return total / max(seen, 1)
        # return total / 10
        return total


        # return self.scorer.score(query, gold_standard_pids, retrieved_pids)


    def write_qrels(self):
        if not os.path.exists("qrels"):
            os.mkdir("qrels")
        with open("qrels/all_qrels.qrel", "w") as out:
            for (query, pid_map) in self.evals.items():
                for (pid, relevance) in pid_map.items():
                    out.write("{} 0 {} {}\n".format(query, pid, relevance))














if __name__ == '__main__':
    run_dir = "/home/jsc57/fixed_psg_runs"
    model_loc = "/home/jsc57/projects/context_summarization/managers/elmo_model"
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_new_paragraphs-elmo_embedding-ndarray.npy"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    # scorer = RetrievalScorer(
    #     model_loc = model_loc,
    #     array_loc = array_loc
    # )

    tqa = "/home/jsc57/data/benchmark/y2-psg-manual-only-tqa.qrels"

    scorer = TrecRunPreparer(
        run_dir=run_dir,
        # qrel_loc=qrel_loc,
        qrel_loc=tqa,
        model_loc=model_loc,
        array_loc=array_loc,
        n_limit=60

    )

    run_pids = scorer.retrieve_run_pids()
    print("Retrieved")

    # pid_map, scores = scorer.build_score_map(run_pids)
    print("Built Score Map")
    # run_scores, run_query_scores = scorer.score_runs(pid_map, scores)
    # run_scores, run_query_scores = scorer.score_runs()
    scorer.score_runs()
    # with open("standard_paragraph_run_scores.json", "w") as f:
    #     f.write(json.dumps(run_query_scores))

    # for (run_name, score) in sorted(list(run_scores.items()), key=lambda x: x[1])[::-1]:
    #     print(run_name.split("/")[-1], score)
    # scorer.write_qrels()
    for param in scorer.param_evaluators:
        param.write_neural_qrels()
        param.write_evals()
        param.write_max_evals()
        param.write_filtered_qrels()

    # print(len(prep.retrieve_pids()))

