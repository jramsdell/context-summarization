import json

from parsing.blobbing.blobber import Blobber
from parsing.run_ranking_parser import RunRankingParser

from parsing.jsonl_paragraph_reader import JSONLParagraphReader
from parsing.outline_reader import OutlineReader
from parsing.qrel_reader import QrelReader


class QrelTextManager(object):
    def __init__(self,
                 manual_qrel_loc,
                 tqa_qrel_loc,
                 y1_json_loc,
                 y1_json_pmap_loc,
                 y2_json_loc,
                 y2_json_pmap_loc,
                 outline_loc,
                 run_dir
                 ):
        qrel_reader = QrelReader(manual_qrel_loc)
        run_parser = RunRankingParser(run_dir)
        # tqa_qrel_reader = QrelReader(tqa_qrel_loc)
        # tqa_ids = tqa_qrel_reader.retrieve_unique_ids()
        ids = qrel_reader.retrieve_unique_ids()
        id_set = set()
        # tqa_id_set = set()

        for i in ids.values():
            id_set = id_set.union(i)

        for (query, pids) in run_parser.qmap.items():
            if query in qrel_reader.qid_set:
                id_set = id_set.union(list(pids)[0:10])

        # for i in tqa_ids.values():
        #     tqa_id_set = tqa_id_set.union(i)


        y2_preader = JSONLParagraphReader(y2_json_loc, y2_json_pmap_loc)
        y2_pid_map = y2_preader.retrieve_by_ids(id_set)

        y1_preader = JSONLParagraphReader(y1_json_loc, y1_json_pmap_loc)
        y1_pid_map = y1_preader.retrieve_by_ids(id_set)
        # y1_pid_map = {}

        outline_reader = OutlineReader(outline_loc)
        query_id_map = outline_reader.retrieve_query_map()

        out = open("training_data.jsonl", "w")



        first = 1
        for k,v in qrel_reader.qrels.items():
            if k.split(":")[0] != "tqa":
                continue


            tqa_paragraphs = list()
            enwiki_paragraphs = list()
            negatives = list()
            seen = set()

            # qmap = qrel_reader.qrels[k]
            # rel_map = dict([(i.pid, i.rel) for i in qmap])

            for i in qrel_reader.negative_qrels[k]:
                seen.add(i)
                if i.pid in y1_pid_map:
                    result = y1_pid_map[i.pid]
                    c = dict(result)
                    c["rel"] = i.rel
                    negatives.append(c)


            for qline in v:
                seen.add(qline.pid)
                if qline.pid in y2_pid_map:

                    result = y2_pid_map[qline.pid]
                    c = dict(result)
                    c["rel"] = qline.rel
                    tqa_paragraphs.append(c)

                    # tqa_paragraphs.append(y2_pid_map[qline.pid])
                else:
                    result = y1_pid_map[qline.pid]
                    c = dict(result)
                    c["rel"] = qline.rel
                    enwiki_paragraphs.append(c)
                    # enwiki_paragraphs.append(y1_pid_map[qline.pid])


            # Also add negative examples from run results with this query
            # for negative_example in run_parser.qmap[k]:
            #     if negative_example not in seen and negative_example in y1_pid_map:
            #         negatives.append(y1_pid_map[negative_example])




            results = {}
            results["qid"] = k
            results["context"] = query_id_map[k]
            results["tqa"] = self.filter_by_size(tqa_paragraphs, 30)
            results["enwiki"] = self.filter_by_size(enwiki_paragraphs, 30)
            results["negatives"] = self.filter_by_size(negatives, 30)
            if results["tqa"] and len(results["enwiki"]) > 2 and len(results["negatives"]) > 2:
                if first:
                    out.write(json.dumps(results))
                else:
                    out.write("\n" + json.dumps(results))
                first = 0


        out.close()

    def filter_by_size(self, l, size):
        return [i for i in l if len(i["text"]) >= size]

class QrelKeywordStatistics(object):
    def __init__(self,
                 manual_qrel_loc,
                 tqa_qrel_loc,
                 y1_json_loc,
                 y1_json_pmap_loc,
                 y2_json_loc,
                 y2_json_pmap_loc,
                 outline_loc,
                 keyword_loc
                 ):
        qrel_reader = QrelReader(manual_qrel_loc)
        # tqa_qrel_reader = QrelReader(tqa_qrel_loc)
        # tqa_ids = tqa_qrel_reader.retrieve_unique_ids()
        ids = qrel_reader.retrieve_unique_ids()
        id_set = set()
        self.blobber = Blobber()
        keyword_map = self.get_keyword_map(keyword_loc)
        print(keyword_map)

        for i in ids.values():
            id_set = id_set.union(i)

        # for i in tqa_ids.values():
        #     tqa_id_set = tqa_id_set.union(i)


        y2_preader = JSONLParagraphReader(y2_json_loc, y2_json_pmap_loc)
        y2_pid_map = y2_preader.retrieve_by_ids(id_set)

        y1_preader = JSONLParagraphReader(y1_json_loc, y1_json_pmap_loc)
        y1_pid_map = y1_preader.retrieve_by_ids(id_set)

        outline_reader = OutlineReader(outline_loc)
        # query_id_map = outline_reader.retrieve_query_map()


        first = 1

        def update_stats(smap, rel, tokens, keyword_set):
            if rel not in smap:
                smap[rel] = [0.0, 0.0]

            stats = smap[rel]
            hits = 0.0
            for token in tokens:
                if token in keyword_set:
                    hits += 1.0

            # Normalize by length
            # hits /= len(tokens)

            # stats[0] += hits / len(keyword_set)
            stats[0] += len(set(tokens).intersection(keyword_set)) / len(keyword_set)
            stats[1] += 1.0




        tqa_key_stats = {}
        context_key_stats = {}
        n_examples = len(keyword_map)

        for (query, (tqa_keys, context_keys)) in keyword_map.items():
            rel_map = qrel_reader.rel_map[query]
            context_keys = set(context_keys)
            tqa_keys = set(tqa_keys)


            for (rel_value, pids) in rel_map.items():
                for pid in pids:
                    text = ""
                    if pid in y1_pid_map:
                        text = y1_pid_map[pid]["text"]
                    else:
                        text = y2_pid_map[pid]["text"]

                    tokens = self.blobber.parse_string(text)
                    update_stats(tqa_key_stats, rel_value, tokens, tqa_keys)
                    update_stats(context_key_stats, rel_value, tokens, context_keys)




        def print_stats(smap, name):
            keys = sorted(smap.keys())
            print("{} stats:".format(name))
            for key in keys:
                score, num = smap[key]
                print("{}: {}".format(key, score / num))


        print_stats(tqa_key_stats, "TQA Keywords")
        print_stats(context_key_stats, "Related Keywords")




    def get_keyword_map(self, loc):
        with open(loc) as f:
            return self._get_ketword_map(f)

    def _get_ketword_map(self, f):
        keyword_map = {}

        counter = 0

        qid = ""
        cur_example = []
        for idx, line in enumerate(f.readlines()):
            line = line.rstrip()
            phase = counter % 4
            if phase == 0: # qid comes first
                qid = line
            elif phase == 1: # then keywords based on tqa passage
                tqa_keywords = self.blobber.parse_string(line.replace(",", ""))
                cur_example.append(tqa_keywords)
            elif phase == 2: # then keywords added based on context / Wikipedia
                context_keywords = self.blobber.parse_string(line.replace(",", ""))
                cur_example.append(context_keywords)
            else: # a newline separate the next example
                keyword_map[qid] = cur_example
                cur_example = []
            counter += 1

        return keyword_map







if __name__ == '__main__':
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-goldpassages.qrels"
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-psg-manual.qrels"
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-psg-lenient.qrels"
    # manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual/all-paragraph-manual.qrels"
    manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    tqa_qrel_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2/benchmarkY2.cbor.tree.qrels"
    qrel_reader = QrelReader(manual_loc)
    ids = qrel_reader.retrieve_unique_ids()
    keyword_loc = "/home/jsc57/plotting/keywords.text"

    y2_json_loc = "/home/jsc57/projects/context_summarization/y2_test.jsonl"
    y2_pmap_loc = "/home/jsc57/projects/context_summarization/y2_test_pmap.txt"
    y1_json_loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
    y1_pmap_loc = "/home/jsc57/projects/context_summarization/y1_corpus_pmap.txt"
    outline_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2.public/benchmarkY2.cbor-outlines.cbor"
    run_dir = "/home/jsc57/fixed_psg_runs"

    manager = QrelTextManager(
        manual_qrel_loc=manual_loc,
        tqa_qrel_loc=tqa_qrel_loc,
        y1_json_loc=y1_json_loc,
        y1_json_pmap_loc=y1_pmap_loc,
        y2_json_loc=y2_json_loc,
        y2_json_pmap_loc=y2_pmap_loc,
        outline_loc=outline_loc,
        run_dir=run_dir
    )

    # manager = QrelKeywordStatistics(
    #     manual_qrel_loc=manual_loc,
    #     tqa_qrel_loc=tqa_qrel_loc,
    #     y1_json_loc=y1_json_loc,
    #     y1_json_pmap_loc=y1_pmap_loc,
    #     y2_json_loc=y2_json_loc,
    #     y2_json_pmap_loc=y2_pmap_loc,
    #     outline_loc=outline_loc,
    #     keyword_loc=keyword_loc
    # )





