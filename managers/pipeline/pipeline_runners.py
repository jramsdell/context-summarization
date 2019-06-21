from typing import *

from managers.pipeline.abstract_text_pipeline import AbstractPipelineRunner, AbstractPipelineWriter, \
    AbstractPipelineReader, PipeEnum
from managers.pipeline.pipeline_writers import JsonPipelineWriter
from utilities.token_utils import tokenize_sentences


class SentenceTokensPipelineRunner(AbstractPipelineRunner):

    def __init__(self, reader: AbstractPipelineReader, **kwargs):
        super().__init__(reader, **kwargs)
        self.max_sentences = self.kwargs.setdefault(PipeEnum.MAX_SENTENCES.value, 4)
        self.max_words = self.kwargs.setdefault(PipeEnum.MAX_WORDS.value, 20)
        self.kwargs.setdefault(PipeEnum.PARALLEL_N_PROCS.value, 30)
        self.kwargs.setdefault(PipeEnum.PARALLEL_N_CHUNKS.value, 4000)

    @staticmethod
    def get_modifier_name():
        return "stoken"

    @staticmethod
    def expected_method():
        return "base"

    @staticmethod
    def required_writer_type() -> Type['AbstractPipelineWriter']:
        return JsonPipelineWriter

    # def run(self):
    #     for text in self.reader.data:
    #         self.output.append(tokenize_sentences(text, max_sentences=max_sentences, max_words=max_words))


    def map_function(self, x):
        return tokenize_sentences(x["text"], max_sentences=self.max_sentences, max_words=self.max_words)



