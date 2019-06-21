from collections import defaultdict
import json

from parsing.jsonl_paragraph_reader import JSONLParagraphReader
from parsing.outline_reader import OutlineReader
from parsing.qrel_reader import QrelReader
from parsing.run_ranking_parser import RunDirParser


class Y1Y2PassageRetriever(object):
    def __init__(self, pids, y1_json_loc, y1_json_pmap_loc,
                 y2_json_loc, y2_json_pmap_loc):

        self.y1_preader = JSONLParagraphReader(y1_json_loc, y1_json_pmap_loc)
        self.y1_pid_map = self.y1_preader.retrieve_by_ids(pids)

        self.y2_preader = JSONLParagraphReader(y2_json_loc, y2_json_pmap_loc)
        self.y2_pid_map = self.y2_preader.retrieve_by_ids(pids)


    def get_text(self, pid):
        text = None
        if pid in self.y1_pid_map:
            text = self.y1_pid_map[pid]["text"]
        elif pid in self.y2_pid_map:
            text = self.y2_pid_map[pid]["text"]
        return text


class BatchTextEmbedder(object):
    def __init__(self,
                 passages
                 ):
        pass





class RunPreparer(object):
    def __init__(self, run_dir, qrel_file, outline_loc):
        parser = RunDirParser(run_dir)
        self.qrel_reader = QrelReader(qrel_file)
        outline_reader = OutlineReader(outline_loc)
        self.query_id_map = outline_reader.retrieve_query_map()
        self.qmap = parser.run()



    def retrieve_pids(self):
        pids = set()
        pmap = defaultdict(lambda: defaultdict(list))

        for (run_name,run) in self.qmap.items():
            for (query, run_lines) in run.items():
                counter = 0
                for run_line in run_lines:
                    pids.add(run_line.pid)
                    pmap[run_name][query].append(run_line.pid)
                    counter += 1
                    if counter >= 20:
                        break
        return pids



    def run(self):
        pids = self.retrieve_pids()

        for (query, more_pids) in self.qrel_reader.retrieve_unique_ids().items():
            pids = pids.union(more_pids)


        y2_json_loc = "/home/jsc57/projects/context_summarization/y2_test.jsonl"
        y2_json_pmap_loc = "/home/jsc57/projects/context_summarization/y2_test_pmap.txt"
        y1_json_loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
        y1_json_pmap_loc = "/home/jsc57/projects/context_summarization/y1_corpus_pmap.txt"

        self.retriever = Y1Y2PassageRetriever(
            pids=pids,
            y1_json_loc=y1_json_loc,
            y1_json_pmap_loc=y1_json_pmap_loc,
            y2_json_loc=y2_json_loc,
            y2_json_pmap_loc=y2_json_pmap_loc
        )

        paragraph_index_map = []
        context_index_map = []
        paragraphs = []
        contexts = []

        # Read paragraphs from Trec Y1
        for (pid, paragraph) in [i for i in self.retriever.y1_pid_map.items() if i is not None]:
            base_dict = {"text": paragraph["text"]}
            paragraphs.append(json.dumps(base_dict))
            paragraph_index_map.append(pid)


        # Read Paragraphs from Trec Y2
        for (pid, paragraph) in [i for i in self.retriever.y2_pid_map.items() if i is not None]:
            base_dict = {"text": paragraph["text"]}
            paragraphs.append(json.dumps(base_dict))
            paragraph_index_map.append(pid)


        # Read Outlines (to get section paths) from Y2 outline
        for query in self.qrel_reader.qrels.keys():
            base_dict = {"text" : " ".join(self.query_id_map[query])}
            contexts.append(json.dumps(base_dict))
            context_index_map.append(query)


        with open("new_paragraphs.index", "w") as out:
            out.write("\n".join(paragraph_index_map))

        with open("new_paragraphs-base-jsonl", "w") as out:
            out.write("\n".join(paragraphs))

        with open("new_contexts.index", "w") as out:
            out.write("\n".join(context_index_map))

        with open("new_contexts-base-jsonl", "w") as out:
            out.write("\n".join(contexts))




if __name__ == '__main__':
    run_dir = "/home/jsc57/fixed_psg_runs"
    outline_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2.public/benchmarkY2.cbor-outlines.cbor"
    manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"

    preparer = RunPreparer(run_dir=run_dir, outline_loc=outline_loc, qrel_file=manual_loc)
    preparer.run()


