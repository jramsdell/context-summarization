from typing import List, Dict

from trec_car.read_data import iter_outlines, iter_paragraphs, iter_annotations, ParaLink


class OutlineReader(object):
    page_title_map = ...  # type: Dict[str, str]
    section_heading_map = ...  # type: Dict[str, List[str]]
    section_id_map = ...  # type: Dict[str, str]

    def __init__(self, f):
        self.page_title_map = {}
        self.section_heading_map = {}
        self.section_id_map = {}
        out = ""

        for outline in iter_outlines(f):
            # print(outline.page_name)
            self.page_title_map[outline.page_id] = outline.page_name
            hmap = []
            imap = []

            for s in outline.child_sections:
                hmap.append(s.heading)
                id = outline.page_id + "/" + s.headingId
                out += id + "\n"
                imap.append(id)
                self.page_title_map[id] = s.heading


            self.section_heading_map[outline.page_id] = hmap
            self.section_id_map[outline.page_id] = imap

        with open("y2_outlines.txt", "w") as f:
            f.write(out)



class CborReader(object):
    def __init__(self, loc):
        with open (loc, 'rb') as f:
            self.parse(f)

    def parse(self, f):
        for paragraph in iter_paragraphs(f):
            for body in paragraph.bodies:
                if isinstance(body, ParaLink):
                    # print(body)
                    if body.link_section is not None:
                        print("Link Section: {}".format(body.link_section))
                        print("Link Text: {}".format(body.get_text()))
                        print("Link Anchor: {}".format(body.anchor_text))
                        print("Link Page: {}".format(body.page))
            # print(paragraph.get_text())

class ArticleReader(object):
    def __init__(self, loc):
        with open (loc, 'rb') as f:
            self.parse(f)

    def parse(self, f):
        for page in iter_annotations(f):
            print(page.p)

if __name__ == '__main__':
    # path = "/home/hcgs/PycharmProjects/run_parsing/benchmarkY2.cbor-outlines.cbor"
    # with open(path, 'rb') as f:
    #     oreader = OutlineReader(f)
    tqa_path = "/home/hcgs/PycharmProjects/run_parsing/benchmarkY2.cbor-articles.cbor"
    cbor_reader = ArticleReader(tqa_path)









