import json

from pytorch_pretrained_bert import BertTokenizer
from textblob import TextBlob


class BertTqaEnwikiAnalyzer(object):
    def __init__(self, loc, n_limit = -1):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.n_limit = n_limit
        self.bert_data = {}
        self.pad_token = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

        with open (loc) as f:
            self.retrieve_data(f)

    def retrieve_data(self, f):
        counter = 0
        for line in f:
            if self.n_limit > 0:
                counter += 1
                if counter > self.n_limit:
                    break

            self.data.append(json.loads(line))

    def get_indexed_tokens(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[0:511] + [tokenized_text[-1]]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return token_ids


    def get_indexed_sentences(self, text):
        token_ids = []
        highest = 0
        for sentence in TextBlob(text).sentences:
            sentence = "[CLS] " + str(sentence) + " [SEP]"
            tokens = self.get_indexed_tokens(sentence)
            token_ids.append(tokens)
            highest = max(highest, len(tokens))

        for tlist in token_ids:
            hdiff = highest - len(tlist)
            if hdiff > 0:
                tlist.extend([self.pad_token for i in range(hdiff)])

        token_ids = token_ids[0:4]
        sdiff = 4 - len(token_ids)
        # for i in range(sdiff):
        #     token_ids.append([self.pad_token for _ in range(highest)])

        return token_ids



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

            # text = "[CLS] " + passage["text"]
            # tokens = self.get_indexed_tokens(text + " [SEP]")
            # l.append(tokens)
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




    def bertanize(self):
        for example in self.data:
            if len(example["tqa"]) < 1 or len(example["enwiki"]) < 2 or len(example["negatives"]) < 2:
                continue
            self.tqa_blob_by_label(example["qid"], "tqa", example["tqa"])
            # self.blob_by_label(example["qid"], "tqa", example["tqa"])
            self.blob_by_label(example["qid"], "enwiki", example["enwiki"])
            self.blob_by_label(example["qid"], "negatives", example["negatives"])


if __name__ == '__main__':
    loc = "/home/jsc57/projects/context_summarization/managers/training_data.jsonl"
    b = BertTqaEnwikiAnalyzer(loc)
    b.bertanize()


