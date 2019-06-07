from parsing.jsonl_paragraph_reader import JSONLParagraphReader
import json
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
                 outline_loc
                 ):
        qrel_reader = QrelReader(manual_qrel_loc)
        # tqa_qrel_reader = QrelReader(tqa_qrel_loc)
        # tqa_ids = tqa_qrel_reader.retrieve_unique_ids()
        ids = qrel_reader.retrieve_unique_ids()
        id_set = set()
        # tqa_id_set = set()

        for i in ids.values():
            id_set = id_set.union(i)

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


        print(list(y2_pid_map.keys()))

        first = 1
        for k,v in qrel_reader.qrels.items():
            if k.split(":")[0] != "tqa":
                continue

            tqa_paragraphs = []
            enwiki_paragraphs = []
            negatives = []

            for i in qrel_reader.negative_qrels[k]:
                if i.pid in y1_pid_map:
                    negatives.append(y1_pid_map[i.pid])

            for qline in v:
                if qline.pid in y2_pid_map:
                    tqa_paragraphs.append(y2_pid_map[qline.pid])
                else:
                    enwiki_paragraphs.append(y1_pid_map[qline.pid])
                    # pass # todo


            results = {}
            results["qid"] = k
            results["context"] = query_id_map[k]
            results["tqa"] = tqa_paragraphs
            results["enwiki"] = enwiki_paragraphs
            results["negatives"] = negatives
            if tqa_paragraphs:
                if first:
                    out.write(json.dumps(results))
                else:
                    out.write("\n" + json.dumps(results))
                first = 0


        out.close()



if __name__ == '__main__':
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-goldpassages.qrels"
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-psg-manual.qrels"
    # manual_loc = "/home/jsc57/plotting/pack/results-for-participants/UNH/benchmarkY2test-psg-lenient.qrels"
    # manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual/all-paragraph-manual.qrels"
    manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    tqa_qrel_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2/benchmarkY2.cbor.tree.qrels"
    qrel_reader = QrelReader(manual_loc)
    ids = qrel_reader.retrieve_unique_ids()

    y2_json_loc = "/home/jsc57/projects/context_summarization/y2_test.jsonl"
    y2_pmap_loc = "/home/jsc57/projects/context_summarization/y2_test_pmap.txt"
    y1_json_loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
    y1_pmap_loc = "/home/jsc57/projects/context_summarization/y1_corpus_pmap.txt"
    outline_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2.public/benchmarkY2.cbor-outlines.cbor"

    manager = QrelTextManager(
        manual_qrel_loc=manual_loc,
        tqa_qrel_loc=tqa_qrel_loc,
        y1_json_loc=y1_json_loc,
        y1_json_pmap_loc=y1_pmap_loc,
        y2_json_loc=y2_json_loc,
        y2_json_pmap_loc=y2_pmap_loc,
        outline_loc=outline_loc
    )



