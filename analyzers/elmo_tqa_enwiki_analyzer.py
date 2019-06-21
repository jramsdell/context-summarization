import json

from textblob import TextBlob


class ElmoTqaEnwikiAnalyzer(object):
    def __init__(self, loc, max_words = 20):
        self.data = []
        self.max_words = max_words
        self.bert_data = {}

        with open (loc) as f:
            self.retrieve_data(f)

    def retrieve_data(self, f):
        for line in f:
            self.data.append(json.loads(line))



    def get_indexed_sentences(self, text):
        sentence_tokens = []
        for sentence in TextBlob(text).sentences:
            # sentence = "[CLS] " + str(sentence) + " [SEP]"
            tokens = sentence.tokens[0:self.max_words]
            sentence_tokens.append(tokens)

        return sentence_tokens



    def blob_by_label(self, qid, label, passages):
        if qid not in self.bert_data:
            self.bert_data[qid] = {}
        q = self.bert_data[qid]

        if label not in q:
            q[label] = []

        l = q[label]

        for passage in passages:
            result = {}
            tokens = self.get_indexed_sentences(passage["text"])
            result["pid"] = passage["pid"]
            result["rel"] = passage["rel"]
            result["tokens"] = tokens
            l.append(result)

    def tqa_blob_by_label(self, qid, label, passages):
        if qid not in self.bert_data:
            self.bert_data[qid] = {}
        q = self.bert_data[qid]

        if label not in q:
            q[label] = []

        l = q[label]
        text = " ".join([passage["text"] for passage in passages])
        tokens = self.get_indexed_sentences(text)
        result = {"pid" : "wee", "tokens" : tokens}
        l.append(result)




    def elmotize(self):
        for example in self.data:
            if len(example["tqa"]) < 1 or len(example["enwiki"]) < 2 or len(example["negatives"]) < 2:
                continue
            self.tqa_blob_by_label(example["qid"], "tqa", example["tqa"])
            # self.blob_by_label(example["qid"], "tqa", example["tqa"])
            self.blob_by_label(example["qid"], "enwiki", example["enwiki"])
            self.blob_by_label(example["qid"], "negatives", example["negatives"])




