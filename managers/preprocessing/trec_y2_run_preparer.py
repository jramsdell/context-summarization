from numpy.core.multiarray import ndarray
from torch.jit.annotations import Module

from learning.trainers.data_handler import DataHandler
from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from parsing.qrel_reader import QrelReader
from parsing.run_ranking_parser import RunDirParser
from collections import defaultdict
import numpy as np
from typing import *
import torch


class TrecRunPreparer(object):
    def __init__(self, run_dir, qrel_loc, array_loc, model_loc):
        run_dir_parser = RunDirParser(run_dir)
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
        # self.scorer = scorer


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
                    if counter >= 10:
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
                    if c2 >= 10:
                        break


        p1s = np.asarray(p1s)
        p2s = np.asarray(p2s)

        predictors = {
            "p1" : p1s,
            "p2" : p2s
        }

        data_handler = DataHandler(predictors)
        scores = (torch.nn.Sigmoid()(self.model(data_handler))).detach().numpy()

        return pid_map, scores


    def score_runs(self, pid_map, scores):
        run_scores = {}
        for (run_name, run) in self.pmap.items():
            run_score = self.score_run(run, pid_map, scores)
            run_scores[run_name] = run_score

        return run_scores

    def score_run(self, run, pid_map, scores):
        total = 0.0
        for (query, retrieved_pids) in run.items():
            if query.split(":")[0] == "enwiki" or query not in self.query_parser.qrels:
                continue
            query_score = self.score_query(query, retrieved_pids, pid_map, scores)
            total += query_score
        return total

    def score_query(self, query, retrieved_pids, pid_map, scores):
        gold_standard_pids = [i.pid for i in self.query_parser.qrels[query]]
        total = 0.0
        seen = 0
        for gs in gold_standard_pids:
            for retrieved_pid in retrieved_pids[0:10]: # todo: don't hardcode top N paragraphs to be 10
                if gs == retrieved_pid:
                    continue
                seen += 1

                key = (gs, retrieved_pid)
                if key in pid_map:
                    total += scores[pid_map[key]]


        return total / seen


        # return self.scorer.score(query, gold_standard_pids, retrieved_pids)


class RetrievalScorer(object):
    model: torch.nn.Module

    def __init__(self, array_loc, model_loc):
        input = { PipeEnum.IN_LOC.value : array_loc }
        self.vector_reader = NumpyArrayPipelineReader(**input)
        self.vector_reader.run()
        self.pid_map = self.vector_reader.index_map
        self.model = torch.load(model_loc)

    def score(self, query: str, gold_standard_pids: List[str], retrieved_pids: List[str]) -> float:
        total = 0.0
        for gold_passage in gold_standard_pids:
            # gold_passage = self.pid_map[gold_standard_pids[0]]
            for pid in retrieved_pids:
                passage = self.pid_map[pid]
                total += self.model(gold_passage, passage)
        return total




class TrecY2RunEvaluator(object):
    def __init__(self, run_dir):
        pass



if __name__ == '__main__':
    run_dir = "/home/jsc57/fixed_psg_runs"
    model_loc = "/home/jsc57/projects/context_summarization/managers/elmo_model"
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/paragraphs-elmo_embedding-ndarray.npy"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    # scorer = RetrievalScorer(
    #     model_loc = model_loc,
    #     array_loc = array_loc
    # )

    scorer = TrecRunPreparer(
        run_dir=run_dir,
        qrel_loc=qrel_loc,
        model_loc=model_loc,
        array_loc=array_loc

    )

    run_pids = scorer.retrieve_run_pids()
    pid_map, scores = scorer.build_score_map(run_pids)
    run_scores = scorer.score_runs(pid_map, scores)
    for (run_name, score) in sorted(list(run_scores.items()), key=lambda x: x[1])[::-1]:
        print(run_name.split("/")[-1], score)
    # print(len(prep.retrieve_pids()))

