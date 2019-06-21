from managers.pipeline.abstract_text_pipeline import PipeEnum
from managers.pipeline.pipeline_readers import NumpyArrayPipelineReader
from parsing.jsonl_paragraph_reader import JSONLParagraphReader
from parsing.outline_reader import OutlineReader
from parsing.qrel_reader import QrelReader
import numpy as np


class Y2TrainingConstructor(object):
    def __init__(self, array_loc, qrel_loc, y2_json_pmap_loc):

        self.qrel_reader = QrelReader(qrel_loc)
        # outline_reader = OutlineReader(outline_loc)
        # query_id_map = outline_reader.retrieve_query_map()


        self.y2_pids = self._get_y2_pids(y2_json_pmap_loc)


        input = { PipeEnum.IN_LOC.value : array_loc }

        self.vector_reader = NumpyArrayPipelineReader(**input)
        self.vector_reader.run()


    def _get_y2_pids(self, pmap_loc):
        pids = set()
        with open(pmap_loc) as f:
            for line in f:
                pids.add(line.split()[0])
        return pids

    def create_examples(self):
        total_tqa_paragraphs = []
        total_samples = []
        total_labels = []

        for (query, qrels) in self.qrel_reader.qrels.items():

            # Skip enwiki queries
            if query.split(":")[0] != "tqa":
                continue

            tqa_paragraphs = []
            positive_paragraphs = []
            negative_paragraphs = []

            negative_qrels = self.qrel_reader.negative_qrels[query]
            positive_qrels = self.qrel_reader.qrels[query]

            for positive_qrel in positive_qrels:
                if positive_qrel.pid in self.y2_pids:
                    tqa_paragraphs.append(self.vector_reader.get_vector(positive_qrel.pid))
                else:
                    positive_paragraphs.append(self.vector_reader.get_vector(positive_qrel.pid))

            for negative_qrel in negative_qrels:
                if negative_qrel.pid in self.y2_pids:
                    # print("What?!?!!: {}".format(query))
                    # print(negative_qrel.pid)
                    tqa_paragraphs.append(self.vector_reader.get_vector(negative_qrel.pid))
                else:
                    negative_paragraphs.append(self.vector_reader.get_vector(negative_qrel.pid))

            if not tqa_paragraphs or len(positive_paragraphs) < 2 or len(negative_paragraphs) < 2:
                continue


            tqa_paragraph = sum(tqa_paragraphs)
            tqa_paragraphs = [tqa_paragraph for _ in range(len(positive_paragraphs) + len(negative_paragraphs))]
            labels = [1 for _ in positive_paragraphs] + [0 for _ in negative_paragraphs]
            labels = np.asarray(labels)

            total_samples.extend(positive_paragraphs)
            total_samples.extend(negative_paragraphs)
            total_tqa_paragraphs.extend(tqa_paragraphs)
            total_labels.extend(labels)

        tqa_matrix = np.asarray(total_tqa_paragraphs)
        sample_matrix = np.asarray(total_samples)
        label_matrix = np.asarray(total_labels)

        base = "/home/jsc57/projects/context_summarization/managers/"

        np.save(base + "elmo_tqa.npy", tqa_matrix)
        np.save(base + "elmo_sample.npy", sample_matrix)
        np.save(base + "elmo_label.npy", label_matrix)


















if __name__ == '__main__':
    array_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/paragraphs-elmo_embedding-ndarray.npy"
    outline_loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2.public/benchmarkY2.cbor-outlines.cbor"
    qrel_loc = "/mnt/grapes/share/trec-car-allruns-2018/all-judgments/manual-tqa/all-paragraph-rouge-manual.qrels"
    y2_pmap_loc = "/home/jsc57/projects/context_summarization/y2_test_pmap.txt"
    constructor = Y2TrainingConstructor(
        array_loc=array_loc,
        qrel_loc=qrel_loc,
        y2_json_pmap_loc=y2_pmap_loc
    )
    constructor.create_examples()



