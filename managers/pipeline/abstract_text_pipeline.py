from abc import ABC
from enum import Enum
from typing import Dict, Any, List, Type

from multiprocess.pool import Pool


class PipelineInputException(Exception):
    pass

class PipelineMethodException(Exception):
    pass



class PipeEnum(Enum):
    IN_LOC = "in_loc"
    OUT_LOC = "out_loc"
    PARALLEL_N_PROCS = "parallel_n_processes"
    PARALLEL_N_CHUNKS = "parallel_n_chunks"
    MAX_SENTENCES = "max_sentences"
    MAX_WORDS = "max_words"


class AbstractPipelineReader(ABC):
    def __init__(self, **kwargs):
        """
        kwargs: Expects IN_LOC pointing to location for file
        """
        file_loc = kwargs[PipeEnum.IN_LOC.value]
        path_elements = file_loc.split("/")
        self.base_path = "/".join(path_elements[0:len(path_elements) - 1])
        self.base_name, self.method_name, self.input_name = path_elements[-1].split("-")
        self.index_loc = self.base_path + "/" + self.base_name + ".index"
        self.index_map = self._get_indices(self.index_loc)
        self.data: List[Any] = []
        self.file_loc = file_loc
        self.kwargs = kwargs

        if self.expected_input() != self.input_name:
            raise PipelineInputException("Expected input of type {} but got type {}".format(
                self.expected_input(), self.input_name
            ))

    def run(self):
        self.data = []

    @staticmethod
    def expected_input() -> str:
        pass

    def _get_indices(self, index_loc):
        index_map: Dict[str, int] = {}
        with open(index_loc) as f:
            for idx, i in enumerate(f.readlines()):
                index_map[i.rstrip()] = idx
        return index_map


class AbstractPipelineRunner(ABC):
    def __init__(self, reader: AbstractPipelineReader, **kwargs):
        self.index_map = reader.index_map
        self.reader = reader
        self.output = []
        self.kwargs = kwargs

        if self.expected_method() != self.reader.method_name:
            raise PipelineMethodException("Expected input method of type {} but got type {}".format(
                self.expected_method(), self.reader.method_name
            ))



    @staticmethod
    def get_modifier_name() -> str:
        pass

    @staticmethod
    def required_writer_type() -> Type['AbstractPipelineWriter']:
        pass

    @staticmethod
    def expected_method() -> str:
        pass


    def run(self):
        nproc = PipeEnum.PARALLEL_N_PROCS.value
        nchunks = PipeEnum.PARALLEL_N_CHUNKS.value
        if nproc in self.kwargs:
            n_processes = self.kwargs[nproc]
            chunks = self.kwargs.get(nchunks, 1)
            pool = Pool(n_processes)
            self.output = [i for i in pool.map(self.map_function, self.reader.data, chunksize=chunks)]
        else:
            self.output = self.map_function(self.reader.data)


    def map_function(self, x):
        return None



class AbstractPipelineWriter(ABC):
    def __init__(self, reader: AbstractPipelineReader, runner: AbstractPipelineRunner, **kwargs):
        self.reader = reader
        self.runner = runner
        self.kwargs = kwargs

    @staticmethod
    def get_modifier_name() -> str:
        return ""

    def run(self):
        pass

    def get_out_name(self):
        out = self.reader.base_path + "/" + self.reader.base_name
        out += "-" + self.runner.get_modifier_name()
        out += "-" + self.get_modifier_name()
        return out




class AbstractTextPipeline(ABC):
    kwargs: Dict[str, Any]
    pipeline_reader: AbstractPipelineReader
    pipeline_runner: AbstractPipelineRunner
    pipeline_writer: AbstractPipelineWriter
    pipeline_id_map: Dict[str, int]
    pipeline_data: List[Any]

    def __init__(self, **kwargs):
        self.kwargs = kwargs


    def set_reader(self, reader_type: Type[AbstractPipelineReader]):
        self.pipeline_reader = reader_type(**self.kwargs)
        self.pipeline_reader.run()

    def set_runner(self, runner_type: Type[AbstractPipelineRunner]):
        self.pipeline_runner = runner_type(self.pipeline_reader, **self.kwargs)
        self.pipeline_runner.run()

    def set_writer(self):
        writer_type = self.pipeline_runner.required_writer_type()
        self.pipeline_writer = writer_type(self.pipeline_reader, self.pipeline_runner, **self.kwargs)
        self.pipeline_writer.run()







def _get_index_loc(file_loc):
    path_elements = file_loc.split("/")
    base_path = "/".join(path_elements[0:len(path_elements) - 1])
    base_name = path_elements[-1].split("-")[0]
    index_loc = base_path + "/" + base_name + ".index"
    return base_path, base_name, index_loc


if __name__ == '__main__':
    pass


