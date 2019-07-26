import random
from collections import defaultdict
import numpy as np
import json
import os

import torch

from learning.trainers.data_handler import DataHandler
from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from parsing.qrel_reader import QrelReader
from parsing.run_ranking_parser import RunDirParser


class TrecEvalScoreMap(object):
    def __init__(self, run_dir, qrel_loc, array_loc, model_loc, n_limit = 10):
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
        print(len(self.vector_reader.index_map.keys()))
        # print((self.vector_reader.data[55000].shape))
        print("Loading model")
        self.model = torch.load(model_loc)
        print("Done")
        self.score_map = {}

        if not os.path.exists("storage/"):
            os.mkdir("storage/")




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


    def build_score_map(self):
        pid_map = {}
        counter = 0
        p1s = []
        p2s = []
        keys = []
        score_map = {}
        chunk_size = 100000

        def update_scores():
            predictors = {
                "p1" : np.asarray(p1s),
                "p2" : np.asarray(p2s)
            }
            data_handler = DataHandler(predictors)
            scores = (torch.nn.Sigmoid()(self.model(data_handler))).detach().numpy()
            # scores = ((self.model(data_handler))).detach().numpy()
            p1s.clear()
            p2s.clear()

            for (key, score) in zip(keys, scores):
                score_map[key] = float(score[0])
            keys.clear()



        for (_, run) in self.pmap.items():

            for (query, retrieved_pids) in run.items():
                if query.split(":")[0] == "enwiki" or query not in self.query_parser.qrels:
                    continue
                gold_standard_pids = [i.pid for i in self.query_parser.qrels[query]]
                c2 = 0

                for pid in [i for i in retrieved_pids]:
                    for gold_standard_pid in gold_standard_pids:
                        key = gold_standard_pid + "_" + pid
                        if key in pid_map:
                            continue

                        p1s.append(self.vector_reader.get_vector(gold_standard_pid))
                        p2s.append(self.vector_reader.get_vector(pid))
                        keys.append(key)
                        pid_map[key] = counter
                        counter += 1
                        if counter % chunk_size == chunk_size - 1:
                            print(counter)
                            update_scores()


                    c2 += 1
                    if c2 >= self.n_limit:
                        break



        # plt.hist(scores, bins='auto')
        # plt.show()

        if keys:
            update_scores()

        return score_map





if __name__ == '__main__':
    run_dir = "/home/jsc57/fixed_psg_runs"
    model_loc = "/home/jsc57/projects/context_summarization/managers/elmo_model"
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs-elmo_embedding-ndarray.npy"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    # scorer = RetrievalScorer(
    #     model_loc = model_loc,
    #     array_loc = array_loc
    # )

    tqa = "/home/jsc57/data/benchmark/y2-psg-manual-only-tqa.qrels"

    scorer = TrecEvalScoreMap(
        run_dir=run_dir,
        qrel_loc=qrel_loc,
        model_loc=model_loc,
        array_loc=array_loc,
        n_limit=80

    )

    run_pids = scorer.retrieve_run_pids()
    score_map = scorer.build_score_map()
    with open("storage/score_map.json", "w") as f:
        f.write(json.dumps(score_map))

    with open("storage/pmap.json", "w") as f:
        f.write(json.dumps(scorer.pmap))



