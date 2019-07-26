import json

from nltk import SnowballStemmer
from nltk.corpus import stopwords
from textblob import TextBlob

class TrecEvalBigramConverter(object):
    def __init__(self, gram_json_loc, gram_index_loc):
        self.stemmer = SnowballStemmer("english")
        self.stop_list = set(stopwords.words("english"))
        self.bigrams = {}


        with open(gram_json_loc) as gram_file:
            with open(gram_index_loc) as index_file:
                for (line, pid) in zip(gram_file.readlines(), index_file.readlines()):
                    text = json.loads(line)["text"]
                    self.bigrams[pid.rstrip()] = self.to_bigrams(text)




    def to_bigrams(self, text):
        words = [self.stemmer.stem(word) for word in TextBlob(text.lower()).words if word not in self.stop_list]
        bigrams = []
        counter = 0
        while counter < len(words) - 1:
            bigrams.append(words[counter] + "_" + words[counter + 1])
            counter += 1
        return bigrams


class TrecEvalBigramRouge(object):
    def __init__(self, bigram_json_loc):
        with open(bigram_json_loc) as f:
            self.bigrams = json.loads(f.read())

    def f1(self, source, target):
        s = set(self.bigrams[source])
        t = set(self.bigrams[target])

        try:
            recall = float(len(s.intersection(t))) / len(s)
            precision = float(len(s.intersection(t))) / len(t)
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0.0

        return f1









if __name__ == '__main__':
    base_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs-base-jsonl"
    index_loc = "/home/jsc57/projects/context_summarization/managers/preprocessing/100_redux_paragraphs.index"

    # wee = TrecEvalBigramConverter(base_loc, index_loc)
    # with open("bigrams.json", "w") as f:
    #     f.write(json.dumps(wee.bigrams))

    bigram_loc = "/home/jsc57/projects/context_summarization/managers/scoring/bigrams.json"
    rouge = TrecEvalBigramRouge(bigram_loc)
    f1, f2 = list(rouge.bigrams.keys())[0:2]
    rouge.f1(f1, f1)


    # print(wee.to_bigrams(wee.grams[0]["text"]))

