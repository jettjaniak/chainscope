import re

from chainscope.typing import *


def get_entity_type(props: Properties) -> str:
    # Extract the entity type from the question using regex
    match = re.match(r"about ([^:]+):", props.gt_question)
    if match:
        entity_type_map = {
            "movies": "movie",
            "songs": "song",
            "books": "book",
            "historical figures": "historical figure",
            "NYT articles": "NYT article",
            "places in NYC": "place in NYC",
            "US cities": "US city",
            "US colleges": "US college",
            "US counties": "US county",
            "US natural places": "US natural place",
            "US structures": "US structure",
            "US zip codes": "US zip code",
            "world natural places": "world natural place",
            "world places": "world place",
            "world structures": "world structure",
        }
        entity_type = match.group(1)
        if entity_type in entity_type_map:
            return entity_type_map[entity_type]
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")
    else:
        raise ValueError(f"Unknown entity type: {props.gt_question}")
    
def get_value(props: Properties) -> str:
    value_map = {
        "work is longer": "Length",
        "work was released": "Release date",
        "located more to the north": "Latitude",
        "located more to the east": "Longitude",
        "published": "Publication date",
        "lived longer": "Lifespan in years",
        "born": "Birth year",
        "died": "Death year",
        "densely populated": "Population density",
        "populous": "Population",
        "area": "Area",
        "longer total runtime": "Total runtime",
        "has more pages": "Number of pages",
    }
    for key, value in value_map.items():
        if key in props.gt_open_ended_question:
            return value
    raise ValueError(f"Unknown value for {props.gt_open_ended_question}")
