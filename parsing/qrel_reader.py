from typing import *
from collections import defaultdict


class QrelLine(object):
    pid = ...  # type: str
    qid = ...  # type: str
    rel = ...  # type: int

    def __init__(self, line: str):
        qid, _, pid, rel = line.split(" ")
        rel = int(rel)

        self.qid = qid
        self.pid = pid
        self.rel = rel


class QrelReader(object):
    qrels = ...  # type: Dict[str, List[QrelLine]]
    negative_qrels = ...  # type: Dict[str, List[QrelLine]]

    def __init__(self, loc):
        self.c = Counter()
        self.qid_set = set()
        self.qrels = {}
        self.rel_map = {}
        self.negative_qrels = {}
        with open(loc, 'r') as f:
            self.parse_qrels(f)

    def _add_qrel_line(self, query: str, qrel_line: QrelLine):
        if query not in self.qrels:
            self.qrels[query] = []

        self.qrels[query].append(qrel_line)

    def _add_negative_qrel_line(self, query: str, qrel_line: QrelLine):
        if query not in self.negative_qrels:
            self.negative_qrels[query] = []

        self.negative_qrels[query].append(qrel_line)



    def parse_qrels(self, f):
        for line in f:
            qrel_line = QrelLine(line)
            self.qid_set.add(qrel_line.qid)

            if qrel_line.qid not in self.rel_map:
                self.rel_map[qrel_line.qid] = {}

            qmap = self.rel_map[qrel_line.qid]
            if qrel_line.rel not in qmap:
                qmap[qrel_line.rel] = []
            qmap[qrel_line.rel].append(qrel_line.pid)

            # self.c[qrel_line.rel] += 1
            # print(self.c)

            # if qrel_line.rel <= 2 and qrel_line.rel > 0:
            if qrel_line.rel <= 0:
                self._add_negative_qrel_line(qrel_line.qid, qrel_line)
            elif qrel_line:
                self._add_qrel_line(qrel_line.qid, qrel_line)




    def retrieve_unique_ids(self) -> DefaultDict[str, Set[str]]:
        query_id_map = defaultdict(set)  # type: DefaultDict[str, Set[str]]

        for (query, qrel_lines) in self.qrels.items():
            for qrel_line in qrel_lines:
                query_id_map[query].add(qrel_line.pid)

        for (query, qrel_lines) in self.negative_qrels.items():
            for qrel_line in qrel_lines:
                query_id_map[query].add(qrel_line.pid)

        return query_id_map




if __name__ == '__main__':
    # loc = "/home/hcgs/data_science/data/qrels/benchmarkY2test-psg-manual.qrels"
    manual_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    keyword_loc = "/home/jsc57/plotting/keywords.text"
    qreader = QrelReader(manual_loc)

    qmap = qreader.retrieve_unique_ids()
    print(qmap)





