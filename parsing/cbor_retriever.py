import json
import sys
from collections import OrderedDict
from typing import *

from trec_car.read_data import iter_paragraphs, Paragraph, ParaLink, ParaText


class CborRetriever(object):
    def __init__(self, cbor_loc: str, json_dump_name: str):
        self.cbor_loc = cbor_loc
        self.json_dump_name = json_dump_name

    def retrieve_text_matching_ids(self, ids: Set[str]):
        jsons = OrderedDict()  # type: OrderedDict[str, str]

        out = open(self.json_dump_name + ".jsonl", 'w')

        counter = 0

        with open(self.cbor_loc, 'rb') as f:
            for paragraph in iter_paragraphs(f):
                counter += 1
                if paragraph.para_id in ids:
                    jsons[paragraph.para_id] = self.create_json(paragraph)

                    # stop once we've retrieved all of the paragraphs
                    ids.remove(paragraph.para_id)
                    if not ids:
                        break

        for _, json in jsons.items():
            out.write(json + "\n")
        out.close()

    def dump_cbor(self):
        out = open(self.json_dump_name + ".jsonl", 'w')
        # index_dir = self.json_dump_name + "_index"
        pmap_out = open(self.json_dump_name + "_pmap.txt", 'w')
        # os.mkdir(index_dir)

        progress = 0
        offset = 0

        with open(self.cbor_loc, 'rb') as f:
            for paragraph in iter_paragraphs(f):
                to_json = self.create_json(paragraph) + "\n"
                out.write(to_json)

                pmap_out.write("{} {} {}\n".format(
                    paragraph.para_id,
                    offset,
                    offset + len(to_json) - 1
                ))

                offset += len(to_json)


                progress += 1
                if progress % 10000 == 0:
                    print(progress)
        out.close()
        pmap_out.close()



    def quick_ids(self):
        counter = 0
        myset = set()
        with open(self.cbor_loc, 'rb') as f:
            for paragraph in iter_paragraphs(f):
                counter += 1
                if counter > 10:
                    break
                myset.add(paragraph.para_id)
        return myset


    def create_json(self, paragraph: Paragraph) -> str:
        pmap = {}
        pmap["text"] = paragraph.get_text()
        # links = {}
        # pmap["links"] = links
        pmap["pid"] = paragraph.para_id

        # Iterate over the entity links inside of the paragraph
        entities = self.get_entities(paragraph)
        pmap["entities"] = entities
        # for child in paragraph.bodies:
        #
        #     # Add contents of entity link to links (used when we dump to json)
        #     if isinstance(child, ParaLink):
        #         link = {}
        #         link["anchor_text"] = child.anchor_text
        #         link["page"] = child.page
        #         link["page_id"] = child.pageid
        #         link["link_section"] = child.link_section
        #         links[child.pageid] = link

        return json.dumps(pmap)




    def explore(self):

        progress = 0
        offset = 0

        with open(self.cbor_loc, 'rb') as f:
            for paragraph in iter_paragraphs(f):
                wee = self.get_entities(paragraph)
                self.extract_from_text(paragraph, wee)
                progress += 1
                if progress % 10 == 0:
                    break


    # def extract_from_text(self, p: Paragraph, entities):
    #     text = p.get_text()
    #     for entity in entities:
    #         start, stop = entity["start"], entity["stop"]
    #         print("Extract: " + text[start:stop])
    #         print("Anchor: " + entity["anchor_text"])



    def get_entities(self, p: Paragraph):
        position = 0
        entities = []

        for body in p.bodies:
            if isinstance(body, ParaText):
                position += len(body.text)
            elif isinstance(body, ParaLink):
                start = position
                stop = start + len(body.anchor_text)

                entity = {}
                entity["name"] = body.page
                entity["id"] = body.pageid
                entity["anchor_text"] = body.anchor_text
                entity["section_link"] = body.link_section
                entity["start"] = start
                entity["stop"] = stop

                entities.append(entity)
                position = stop


        return entities



if __name__ == '__main__':

    # loc = "/home/hcgs/data_science/data/corpus/dedup.articles-paragraphs.cbor"
    # loc = "/home/jsc57/data/corpus/paragraphCorpus/dedup.articles-paragraphs.cbor"


    # retriever = CborRetriever(loc, "")
    # retriever.explore()

    loc = sys.argv[1]
    label = sys.argv[2]
    retriever = CborRetriever(loc, label)
    retriever.dump_cbor()



