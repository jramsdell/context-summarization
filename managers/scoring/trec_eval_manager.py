import os
from scipy.stats import kendalltau

from managers.scoring.trec_eval_collection import TrecEvalCollection


class TrecEvalManager(object):
    def __init__(self, baseline_loc: str, run_dirs_loc: str, metric: str):
        self.metric = metric
        self.baseline_collection = TrecEvalCollection(baseline_loc, "map").get_rankings()
        self.run_collections = {}

        for run_dir_loc in os.listdir(run_dirs_loc):
            collection = TrecEvalCollection(run_dirs_loc + "/" + run_dir_loc, self.metric)
            key = "{}_{}".format(run_dirs_loc.split("/")[-1], run_dir_loc)
            self.run_collections[key] = collection.get_rankings()



    def return_correlations(self):
        def vs(x):
            offset = 1
            e = x.split("_")
            cutoff = e[offset + 1][5:]
            pos = e[offset + 2][3:]
            neg = e[offset + 3][3:].replace(".qrel", "") # bug
            return (int(cutoff), int(pos), int(neg))

        # for (method, rankings) in sorted(self.run_collections.items(), key=lambda x: x[0]):
        for (method, rankings) in sorted(self.run_collections.items(), key=lambda x: vs(x[0])):
            correlation = self.get_spearman(self.baseline_collection, rankings)
            kendall_tau = self.get_kendall_tau(self.baseline_collection, rankings)

            cutoff, pos, neg = vs(method)

            # e = method.split("_")
            # cutoff = e[2][5:]
            # pos = e[3][3:]
            # neg = e[4][3:].replace(".qrel", "") # bug

            print("{} & {}\\% & {}\\% & {:.4f} & {:.4f} & {:.4f}\\\\".format(
                cutoff, pos, neg, correlation, kendall_tau[0], kendall_tau[1]
            ))


            # print("{}: {} / {}".format(method.replace("depth", "cutoff"),
            #                            correlation, kendall_tau))


    def get_spearman(self, ranks1, ranks2):
        d = 0.0
        for key in ranks1:
            ranking1 = ranks1[key]
            ranking2 = ranks2[key]
            d += (ranking1 - ranking2) ** 2

        d *= 6.0
        n = len(ranks1)
        denom = n * (n ** 2 - 1)
        return 1 - d / denom


    def get_kendall_tau(self, ranks1, ranks2):
        rankings1 = []
        rankings2 = []
        for key in ranks1:
            rankings1.append(ranks1[key])
            rankings2.append(ranks2[key])

        corr, pvalue = kendalltau(rankings1, rankings2)
        return [corr, pvalue]








if __name__ == '__main__':
    baseline_loc = "/home/jsc57/projects/context_summarization/baseline_evals"
    run_dirs_loc = "/home/jsc57/projects/context_summarization/map_evals"
    # run_dirs_loc = "/home/jsc57/projects/context_summarization/filtered_evals"
    # run_dirs_loc = "/home/jsc57/projects/context_summarization/max_evals"
    # run_dirs_loc = "/home/jsc57/projects/context_summarization/evals"
    manager = TrecEvalManager(baseline_loc, run_dirs_loc, "map")
    # manager = TrecEvalManager(baseline_loc, run_dirs_loc, "neural_score")
    manager.return_correlations()


