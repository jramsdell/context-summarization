import os
from typing import List, Any

from managers.scoring.trec_eval_reader import TrecEvalReader


class TrecEvalCollection(object):
    eval_readers: List[TrecEvalReader]

    def __init__(self, eval_directory: str, metric: str):
        self.eval_readers = []
        self.metric = metric
        for eval_loc in os.listdir(eval_directory):
            self.eval_readers.append(TrecEvalReader(eval_directory + "/" + eval_loc, eval_loc))


    def get_rankings(self):
        scores = []
        for reader in self.eval_readers:
            scores.append(
                [reader.run_name, reader.get_stats(self.metric)[0]]
            )

        scores.sort(key=lambda x: x[1], reverse=True)
        rankings = dict([[run, idx + 1] for idx, (run, _) in enumerate(scores)])
        return rankings


if __name__ == '__main__':
    eval_directory = "/home/jsc57/projects/context_summarization/run_evals"
    manager = TrecEvalCollection(eval_directory, "map")
    rankings = manager.get_rankings()


