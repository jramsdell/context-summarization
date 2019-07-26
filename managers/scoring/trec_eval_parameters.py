import math
from collections import defaultdict, Counter
import random
import os

class TrecEvalParameters(object):
    def __init__(self, max_depth, negative_chance, positive_chance, name):
        self.name = name
        self.qrels = defaultdict(dict)
        self.evals = defaultdict(lambda: defaultdict(list))
        self.max_evals = defaultdict(lambda: defaultdict(Counter))
        self.max_depth = max_depth
        self.negative_chance = negative_chance
        self.positive_chance = positive_chance
        self.allowed = defaultdict(dict)

    def update_map(self, cur_depth, query, pid, in_positive, in_negative, score):
        if cur_depth > self.max_depth:
            return

        allowed_pids = self.allowed[query]

        if in_positive and pid not in allowed_pids:
            result = 0 if random.random() < self.positive_chance else 1
            allowed_pids[pid] = result

        if in_negative and pid not in allowed_pids:
            result = 0 if random.random() < self.negative_chance else 2
            allowed_pids[pid] = result

        # Ignore pid if not allowed
        if (in_negative or in_positive) and (allowed_pids[pid] == 0):
            return

        if score > 0.5:
            query_map = self.qrels[query]
            query_map[pid] = 1



    def update_score(self, cur_depth, run_name, query, pid, in_positive, in_negative, score):
        if cur_depth > self.max_depth:
            return

        allowed_pids = self.allowed[query]

        if in_positive and pid not in allowed_pids:
            result = 0 if random.random() < self.positive_chance else 1
            allowed_pids[pid] = result

        if in_negative and pid not in allowed_pids:
            result = 0 if random.random() < self.negative_chance else 2
            allowed_pids[pid] = result

        # Ignore pid if not allowed
        if (in_negative or in_positive) and (allowed_pids[pid] == 0):
            return


        # self.evals[run_name][query] += math.log(score)
        # self.evals[run_name][query] += (score / (cur_depth + 1)**2)
        score_list = self.evals[run_name][query]
        if len(score_list) < self.max_depth:
            score_list.append(score)
        # self.evals[run_name][query].append(score)

    def update_score_max(self, cur_depth, run_name, query, pid, in_positive, in_negative, score):
        if cur_depth > self.max_depth:
            return

        allowed_pids = self.allowed[query]

        if in_positive and pid not in allowed_pids:
            result = 0 if random.random() < self.positive_chance else 1
            allowed_pids[pid] = result

        if in_negative and pid not in allowed_pids:
            result = 0 if random.random() < self.negative_chance else 2
            allowed_pids[pid] = result

        # Ignore pid if not allowed
        if (in_negative or in_positive) and (allowed_pids[pid] == 0):
            return

        cur_score = self.max_evals[run_name][query][pid]

        self.max_evals[run_name][query][pid] = max(cur_score, score)




    def write_filtered_qrels(self):
        if not os.path.exists("filtered_qrels"):
            os.mkdir("filtered_qrels")

        with open("filtered_qrels/{}".format(self.name), "w") as out:
            for (query, pid_map) in self.allowed.items():
                for (pid, is_allowed) in pid_map.items():
                    rel = 1 if is_allowed == 1 else 0
                    out.write("{} 0 {} {}\n".format(query, pid, rel))

    def write_neural_qrels(self):
        if not os.path.exists("qrels"):
            os.mkdir("qrels")
        with open("qrels/{}".format(self.name), "w") as out:
            for (query, pid_map) in self.qrels.items():
                for (pid, relevance) in pid_map.items():
                    out.write("{} 0 {} {}\n".format(query, pid, relevance))

    def write_evals(self):
        if not os.path.exists("evals"):
            os.mkdir("evals")

        path = "evals/{}".format(self.name.split(".")[0])

        if not os.path.exists(path):
            os.mkdir(path)

        for (run_name, query_map) in self.evals.items():
            with open("{}/{}".format(path, run_name), "w") as out:
                for (query, query_score) in query_map.items():
                    out.write("{}\t{}\t{}\n".format(
                        # "neural_score", query, query_score / self.max_depth
                    "neural_score", query, sum(query_score) / len(query_score)
                    ))

    def write_max_evals(self):
        if not os.path.exists("max_evals"):
            os.mkdir("max_evals")

        path = "max_evals/{}".format(self.name.split(".")[0])

        if not os.path.exists(path):
            os.mkdir(path)

        for (run_name, query_map) in self.max_evals.items():
            with open("{}/{}".format(path, run_name), "w") as out:
                # for (query, pid_scores) in query_map.items():
                for (query, pid_scores) in sorted(query_map.items(), key=lambda x: x[0]):
                    # best = sum(sorted(pid_scores.values())[::-1][0:3])
                    out.write("{}\t{}\t{}\n".format(
                        "neural_score", query, sum(pid_scores.values()) / len(pid_scores)
                    ))



    @staticmethod
    def generate_parameters():
        parameters = []
        chances = [(0.0, "0"), (0.5, "50"), (1.0, "100")]
        # chances = [(0.0, "0"), (0.5, "50"), (0.75, "75"), (1.0, "100")]
        # chances = [(0.5, "50"), (1.0, "100")]
        for (neg_chance, neg_name) in  chances:
            for (pos_chance, pos_name) in chances:
                for max_depth in [10, 20, 30]:
                    name = "depth{}_pos{}_neg{}.qrel".format(max_depth, pos_name, neg_name)
                    parameters.append(
                        TrecEvalParameters(
                            max_depth=max_depth,
                            positive_chance=pos_chance,
                            negative_chance=neg_chance,
                            name=name
                        )
                    )


        return parameters
