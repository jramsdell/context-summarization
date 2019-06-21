
import json
import random
from random import shuffle

from trec_car.read_data import iter_annotations, Para


class Y1TrainingExampleMaker(object):
    def __init__(self, cbor_loc: str, json_dump_name: str):
        self.cbor_loc = cbor_loc
        self.json_dump_name = json_dump_name
        random.seed(10)

    def dump_cbor(self):
        # out = open(self.json_dump_name + ".jsonl", 'w')
        # index_dir = self.json_dump_name + "_index"
        # pmap_out = open(self.json_dump_name + "_pmap.txt", 'w')
        # os.mkdir(index_dir)


        progress = 0
        offset = 0
        counter = 0

        # random_pages = list(range(300000))
        random_pages = list(range(300000))
        shuffle(random_pages)
        random_pages = set(random_pages[0:100000])
        # random_pages = set(random_pages[0:10])

        out = "y1_comparison_training_examples.json"
        out = open(out, "w")
        success = 0

        with open(self.cbor_loc, 'rb') as f:
            for page in iter_annotations(f):
                counter += 1
                if counter in random_pages:
                    random_pages.remove(counter)

                    sections = self.get_proper_sections(page)
                    if len(sections) >= 4:
                        success += 1
                        self.create_training_examples(sections, page, out)

                    print("{} / {}".format(success, len(random_pages)))
                    if not random_pages:
                        break

        out.close()



    def create_training_examples(self, sections, page, out):
        page_name = page.page_name
        paragraphs = []
        for sec in sections:
            paragraphs.extend(sec.children)

        for sec in sections:
            example = {}
            first = sec.children[0]
            child_set = set(sec.children)
            positives = sec.children[1:]
            negatives = []
            shuffle(paragraphs)
            section_name = sec.heading

            for p in paragraphs:
                if p not in child_set:
                    negatives.append(p)
                    if len(negatives) >= len(positives):
                        break


            example['p1'] = [self.create_pdict(first)]
            example['positives'] = [self.create_pdict(i) for i in positives]
            example['negatives'] = [self.create_pdict(i) for i in negatives]
            example['query'] = page_name + " " + section_name


            out.write(json.dumps(example) + "\n")






    def create_pdict(self, para: Para):
        pdict = {}
        pdict["pid"] = para.paragraph.para_id
        pdict["text"] = para.paragraph.get_text()

        # Todo: Figure out what some of these passages are empty!
        if pdict["text"] is None or len(pdict["text"]) < 20:
            return None
        return pdict







    def get_proper_sections(self, page):
        proper_sections = []
        for section in page.child_sections:
            proper_children = []
            for child in section.children:
                if isinstance(child, Para):
                    proper_children.append(child)
            section.children = proper_children
            if len(section.children) > 3:
                proper_sections.append(section)

        return proper_sections





if __name__ == '__main__':
    loc = "/home/jsc57/data/unprocessedAllButBenchmark.cbor/unprocessedAllButBenchmark.cbor"
    maker = Y1TrainingExampleMaker(loc, "wee")
    maker.dump_cbor()
