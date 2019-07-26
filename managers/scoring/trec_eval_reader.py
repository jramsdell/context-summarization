from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np


# class TrecEvalLine(object):
#     method: str
#     query: str
#     score: float
#
#     def __init__(self, method, query, score):
#         self.method = method
#         self.query = query
#         self.score = score


class TrecEvalReader(object):
    evals: Dict[str, Dict[str, float]]  # metric -> query -> score

    def __init__(self, eval_loc, run_name):
        self.evals = defaultdict(dict)
        self.run_name = run_name
        with open(eval_loc) as f:
            self.parse(f)

    def parse(self, f):
        for line in f:
            if line.startswith("runid"):
                continue
            metric, query, score = line.split()
            score_method_map = self.evals[metric]
            score_method_map[query] = float(score)

    def get_stats(self, metric: str) -> Tuple[float, float]:
        """
        :param metric: The metric to retrieve run stats for
        :return: across all queres: mean score (under metric), standard deviation (under metric)
        """
        score_method_map = self.evals[metric]
        results = list(score_method_map.values())
        results = np.asarray(results)
        # return results.mean(), results.std()
        return results.mean(), results.std()




if __name__ == '__main__':
    loc = "/home/jsc57/projects/run_comparison/final_eval_results/manual_tqa/guir"
    teval = TrecEvalReader(loc, "guir")
    print(teval.evals)
    print(teval.get_stats("RougeF1"))
