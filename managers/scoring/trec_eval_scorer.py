from learning.trainers.data_handler import DataHandler
import random
import os
from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from random import shuffle

from managers.scoring.trec_eval_bigram_rouge import TrecEvalBigramRouge
from managers.scoring.trec_eval_parameters import TrecEvalParameters
from parsing.qrel_reader import QrelReader
from parsing.run_ranking_parser import RunDirParser
from collections import defaultdict
import numpy as np
from typing import *
import torch
import json
import matplotlib.pyplot as plt


class TrecEvalScorer(object):
    def __init__(self, run_pmap_loc, score_map_loc, qrel_loc, n_limit = 10, bigram_loc=None):


        if bigram_loc is not None:
            self.rouge = TrecEvalBigramRouge(bigram_loc)
        self.param_evaluators = TrecEvalParameters.generate_parameters()
        self.neural_qrels = defaultdict(dict)
        random.seed(213)
        self.n_limit = n_limit
        with open(score_map_loc) as f:
            self.score_map = json.loads(f.read())
        print("LOADED")

        with open(run_pmap_loc) as f:
            self.pmap = json.loads(f.read())

        print("LOADED")
        self.query_parser = QrelReader(qrel_loc)
        self.same = [0.0, 0.0]
        self.diff = [0.0, 0.0]
        self.neg = [0.0, 0.0]






    # def score_runs(self, pid_map, scores):
    def score_runs(self):
        for (run_name, run) in sorted(self.pmap.items(), key=lambda x: x[0]):
            run_name = run_name.split("/")[-1]
            self.score_run(run_name, run)


    def score_run(self, run_name, run):
        for (query, retrieved_pids) in run.items():
            if query.split(":")[0] == "enwiki" or query not in self.query_parser.qrels:
                continue
            self.score_query(run_name, query, retrieved_pids)



    def score_query(self, run_name, query, retrieved_pids):
        gold_standard_pids = set([i.pid for i in self.query_parser.qrels[query]])
        negatives = set([i.pid for i in self.query_parser.negative_qrels[query]])
        for gs in gold_standard_pids:
            seen = 0

            for retrieved_pid in retrieved_pids:
                # in_positive = gs == retrieved_pid and gs in self.all_positives
                # in_negative = retrieved_pid in negatives and retrieved_pid in self.all_negatives
                in_positive = retrieved_pid in gold_standard_pids
                in_negative = retrieved_pid in negatives
                seen += 1
                if seen >= 100:
                    break


                key = gs + "_" + retrieved_pid
                if key not in self.score_map:
                    continue

                score = self.score_map[key]
                if in_positive and retrieved_pid != gs:
                    self.diff = [self.diff[0] + score, self.diff[1] + 1.0]
                elif retrieved_pid == gs:
                    self.same = [self.same[0] + score, self.same[1] + 1.0]
                elif in_negative:
                    self.neg = [self.neg[0] + score, self.neg[1] + 1.0]

                if score > 0.5:
                    self.neural_qrels[query][retrieved_pid] = 1
                # if score > 0.5 and in_negative:
                #     print("WOA")

                f1 = self.rouge.f1(gs, retrieved_pid)

                for param_evaluator in self.param_evaluators:
                    param_evaluator.update_map(seen, query, retrieved_pid, in_positive, in_negative, score)
                    param_evaluator.update_score(seen, run_name, query, retrieved_pid, in_positive, in_negative, score)
                    param_evaluator.update_score_max(seen, run_name, query, retrieved_pid, in_positive, in_negative, f1)
                    # param_evaluator.update_score_max(seen, run_name, query, retrieved_pid, in_positive, in_negative, score)



    def write_qrels(self):
        if not os.path.exists("qrels"):
            os.mkdir("qrels")
        with open("qrels/all_qrels.qrel", "w") as out:
            for (query, pid_map) in self.neural_qrels.items():
                for (pid, relevance) in pid_map.items():
                    out.write("{} 0 {} {}\n".format(query, pid, relevance))














if __name__ == '__main__':
    run_dir = "/home/jsc57/fixed_psg_runs"
    model_loc = "/home/jsc57/projects/context_summarization/managers/elmo_model"
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs-elmo_embedding-ndarray.npy"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"

    run_pmap_loc = "/home/jsc57/projects/context_summarization/storage/pmap.json"
    score_map_loc = "/home/jsc57/projects/context_summarization/storage/score_map.json"
    tqa = "/home/jsc57/data/benchmark/y2-psg-manual-only-tqa.qrels"
    bigram_loc = "/home/jsc57/projects/context_summarization/managers/scoring/bigrams.json"

    scorer = TrecEvalScorer(
        run_pmap_loc=run_pmap_loc,
        score_map_loc=score_map_loc,
        qrel_loc=tqa,
        n_limit=80,
        bigram_loc=bigram_loc
    )

    scorer.score_runs()
    print("Same: {}", scorer.same[0] / scorer.same[1])
    print("Diff: {}", scorer.diff[0] / scorer.diff[1])
    print("Neg: {}", scorer.neg[0] / scorer.neg[1])
    for param in scorer.param_evaluators:
        param.write_neural_qrels()
        param.write_evals()
        param.write_max_evals()
        # param.write_filtered_qrels()


