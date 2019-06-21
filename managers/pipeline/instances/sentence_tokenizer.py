from managers.pipeline.abstract_text_pipeline import AbstractTextPipeline, PipeEnum
from managers.pipeline.pipeline_readers import JsonPipelineReader
from managers.pipeline.pipeline_runners import SentenceTokensPipelineRunner


class TextBlobSentenceTokenizerPipeline(AbstractTextPipeline):

    @staticmethod
    def construct(**kwargs):
        pipeline = TextBlobSentenceTokenizerPipeline(**kwargs)
        pipeline.set_reader(reader_type=JsonPipelineReader)
        pipeline.set_runner(runner_type=SentenceTokensPipelineRunner)
        pipeline.set_writer()



if __name__ == '__main__':
    file_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/new_paragraphs-base-jsonl"

    in_dict = {
        PipeEnum.IN_LOC.value : file_loc
        # PipeEnum.PARALLEL_N_PROCS.value : 30,
        # PipeEnum.PARALLEL_N_CHUNKS.value : 100
    }

    pipeline = TextBlobSentenceTokenizerPipeline.construct(**in_dict)





