import json

import numpy as np

from managers.pipeline.abstract_text_pipeline import AbstractPipelineReader


class JsonPipelineReader(AbstractPipelineReader):
    def run(self):
        self.data = []
        with open(self.file_loc) as f:
            for line in f:
                self.data.append(json.loads(line))


    @staticmethod
    def expected_input():
        return "jsonl"



class NumpyArrayPipelineReader(AbstractPipelineReader):
    @staticmethod
    def expected_input():
        return "ndarray.npy"


    def run(self):
        self.data = np.load(self.file_loc)

    def get_vector(self, id):
        return self.data[self.index_map[id]]
