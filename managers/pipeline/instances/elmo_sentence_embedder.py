from typing import *

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

from managers.pipeline.abstract_text_pipeline import AbstractTextPipeline, PipeEnum, AbstractPipelineRunner, \
    AbstractPipelineReader, AbstractPipelineWriter
from managers.pipeline.pipeline_readers import JsonPipelineReader
from managers.pipeline.pipeline_writers import NumpyArrayPipelineWriter


class ElmoVectorEmbedderRunner(AbstractPipelineRunner):

    def __init__(self, reader: AbstractPipelineReader, **kwargs):
        super().__init__(reader, **kwargs)
        self.max_sentences = self.kwargs.setdefault(PipeEnum.MAX_SENTENCES.value, 4)
        self.max_words = self.kwargs.setdefault(PipeEnum.MAX_WORDS.value, 20)
        self.embedding_size = 1024
        self.model = ElmoEmbedder()

        self.null_vector = np.zeros((self.max_sentences, 1024))




    @staticmethod
    def get_modifier_name():
        return "elmo_embedding"

    @staticmethod
    def expected_method():
        return "stoken"

    @staticmethod
    def required_writer_type() -> Type['AbstractPipelineWriter']:
        return NumpyArrayPipelineWriter


    def get_embedding(self, tokens):
        # return [self.model.embed_sentence(i) for i in p1["tokens"]]
        sentences =  [i[0] for i in self.model.embed_sentences(tokens[0:self.max_sentences])]
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            # if sentence.shape[0] < self.max_words:
            #     word_diff = self.max_words - sentence.shape[0]
            #     zshape = (word_diff, sentence.shape[1])
            #     sentence = np.concatenate([sentence, np.zeros(zshape)], 0)
            sentences[idx] = sentence.mean(0)

        sentences = np.asarray(sentences)


        try:
            if sentences.shape[0] < self.max_sentences:
                sentence_diff = self.max_sentences - sentences.shape[0]
                # zshape = (sentence_diff, self.max_words, self.embedding_size)
                zshape = (sentence_diff, self.embedding_size)
                sentences = np.concatenate([sentences, np.zeros(zshape)], 0)
        except ValueError:
            return None


        return sentences




    def map_function(self, text_tokens):
        results = []
        mlength = len(text_tokens)
        for idx, tokens in enumerate(text_tokens):
            embedded = self.get_embedding(tokens)
            if embedded is not None:
                results.append(embedded)
            else:
                results.append(self.null_vector)
                print("Problem with: {}".format(idx))
            if idx % 100 == 0:
                print("{} out of {}".format(idx, mlength))
        return results


class ElmoSentenceEmbeddingPipeline(AbstractTextPipeline):

    @staticmethod
    def construct(**kwargs):
        pipeline = ElmoSentenceEmbeddingPipeline(**kwargs)
        pipeline.set_reader(reader_type=JsonPipelineReader)


        pipeline.set_runner(runner_type=ElmoVectorEmbedderRunner)
        pipeline.set_writer()



if __name__ == '__main__':
    # import os
    # for (k,v) in os.environ.items():
    #     print("{}: {}".format(k, v))

    # file_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/paragraphs-elmo_embedding-ndarray.npy"
    #
    #
    # nreader = NumpyArrayPipelineReader(**in_dict)
    #
    # nreader.run()

    file_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs-stoken-jsonl"
    in_dict = {
        PipeEnum.IN_LOC.value : file_loc
    }
    pipeline = ElmoSentenceEmbeddingPipeline.construct(**in_dict)
