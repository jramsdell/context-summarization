import json
from typing import *
from typing import Dict, List, Any

from trec_car.read_data import iter_outlines



class OutlineReader(object):
    def __init__(self, outline_loc):
        self.outline_loc = outline_loc


    def retrieve_query_map(self) -> Dict[str, Tuple[str, str]]:
        with open(self.outline_loc, 'rb') as f:
            return self._parse(f)

    def _parse(self, f) -> Dict[str, Tuple[str, str]]:
        heading_map: Dict[str, Tuple[str, str]] = {}

        for page in iter_outlines(f):
            for nested in page.nested_headings():
                top_level = nested[0]
                key = page.page_id + "/" + top_level.headingId
                heading_map[key] = [page.page_name, top_level.heading]

        return heading_map



if __name__ == '__main__':
    loc = "/home/ben/trec-car/data/benchmarkY2/benchmarkY2.public/benchmarkY2.cbor-outlines.cbor"
    outline_reader = OutlineReader(loc)
    retrieved = outline_reader.retrieve_query_map()


