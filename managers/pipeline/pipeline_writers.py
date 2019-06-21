import json

import numpy as np

from managers.pipeline.abstract_text_pipeline import AbstractPipelineWriter


class JsonPipelineWriter(AbstractPipelineWriter):

    @staticmethod
    def get_modifier_name():
        return "jsonl"

    def run(self):
        with open(self.get_out_name(), "w") as out:
            counter = 0
            for datum in self.runner.output:
                suffix = "\n" if counter != len(self.runner.output) - 1 else ""
                out.write(json.dumps(datum) + suffix)
                counter += 1


class NumpyArrayPipelineWriter(AbstractPipelineWriter):

    @staticmethod
    def get_modifier_name():
        return "ndarray"

    def run(self):
        ndarray = np.asarray(self.runner.output)
        np.save(self.get_out_name(), ndarray)



