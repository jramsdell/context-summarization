import json
import numpy as np
from scipy.stats import ttest_rel
import os


def get_array(stats):
    array = []
    for run in sorted(stats.keys()):
        temp_array = []
        run_results = stats[run]
        for query in sorted(run_results.keys()):
            query_result = run_results[query]
            temp_array.append(query_result)
        total = sum(temp_array)
        array.append(total)
        # for i in temp_array:
        #     array.append(i / total)
    total = sum(array)
    # array = [i / total for i in array]
    return np.asarray(array)

def to_run_array(run):
    array = []
    for (query, query_score) in sorted(run.items(), key=lambda x: x[0]):
        array.append(query_score)
    return np.asarray(array)



def paired_stuff(stats):
    best = get_rankings(stats)
    # best_array = to_run_array(stats[best[0]])
    arrays = {}
    for (run, run_result) in stats.items():
        arrays[run] = to_run_array(run_result)

    best_array = arrays[best[0]]




def the_whole_shebang(stats):
    if not os.path.exists("run_stuff"):
        os.mkdir("run_stuff")

    for run, run_scores in stats.items():
        run_name = run.split("/")[-1]
        run_path = "run_stuff/{}".format(run_name)
        with open(run_path, "w") as f:
            for query, query_score in run_scores.items():
                f.write("{}\t{}\t{}\n".format(
                    "similarity", query, query_score
                ))




def do_the_thing(stats1, stats2):
    arr1 = get_array(stats1)
    arr2 = get_array(stats2)
    print(arr1)
    print(arr2)
    print(ttest_rel(arr1, arr2))

def get_rankings(stats):
    run_scores = {}
    for run in sorted(stats.keys()):
        total = 0.0
        for (_, query_score) in stats[run].items():
            total += query_score
        # run_scores[run.split("/")[-1]] = total
        run_scores[run] = total

    best = list(sorted(run_scores.items(), key=lambda x: x[1]))[::-1]
    return best[0]



if __name__ == '__main__':
    loc1 = "/home/jsc57/projects/context_summarization/managers/preprocessing/standard_paragraph_run_scores.json"
    loc2 = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_standard_paragraph_run_scores.json"

    with open(loc1) as f1:
        stats1 = json.loads(f1.read())

    with open(loc2) as f2:
        stats2 = json.loads(f2.read())


    do_the_thing(stats1, stats2)

    # paired_stuff(stats1)
    the_whole_shebang(stats1)


