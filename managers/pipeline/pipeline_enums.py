from enum import Enum

from managers.pipeline.pipeline_runners import SentenceTokensPipelineRunner
from managers.pipeline.pipeline_writers import JsonPipelineWriter, NumpyArrayPipelineWriter


class RunnerMethodEnum(Enum):
    BASIC_SENTENCE_TOKENS = SentenceTokensPipelineRunner

class WriterMethodEnum(Enum):
    JSONL = JsonPipelineWriter
    NUMPY_ARRAY = NumpyArrayPipelineWriter
