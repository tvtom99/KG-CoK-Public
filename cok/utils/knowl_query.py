from cok.utils.retrieval.wikidata_embedding import retrieve_wikidata_knowledge
from cok.utils.retrieval.wikidata_embedding import retrieve_dummy_knowledge
from collections.abc import Callable
from cok.typing import Knowledge

# domain and knowledge sources mapping
domain_mapping: dict[str, dict[str, Callable[[str, dict, int, int, int], str]]] = {
    "factual": {
        "wikidata": retrieve_wikidata_knowledge,
        # "wikitable": retrieve_wikitable_knowledge,
        # "dpr": retrieve_dpr_knowledge,
        "wikipedia": retrieve_dummy_knowledge,
    },
    "medical": {
        # "flashcard": retrieve_flashcard_knowledge,
        "uptodate": retrieve_dummy_knowledge,
    },
    "biology": {
        "scienceqa_bio": retrieve_dummy_knowledge,
        "ck12": retrieve_dummy_knowledge,
    },
    "physical": {
        "scienceqa_phy": retrieve_dummy_knowledge,
        "physicsclassroom": retrieve_dummy_knowledge,
    },
}


def retrieve_knowledge(domain_list: list[str], input: str, data_point: dict, k: int, width: int, depth: int) -> dict[str, dict[str, str]]:
    # input is a string
    knowl: dict[str, dict] = {}
    # If not in mapping, automatically use "factual"
    domain_list = [x if x in domain_mapping else "factual" for x in domain_list]
    # Remove duplicates through using fancy dictionarys
    domain_list = list(dict.fromkeys(domain_list));
    for domain in domain_list:
        knowl[domain] = {}
        domain_sources = domain_mapping[domain]
        for source in domain_sources:
            print("--- Retrieving knowledge from", domain, source)
            # Get the retrieval function for the current domain and source
            # retrieval_function = domain_sources[source];
            retrieval_function = domain_sources.get(source, retrieve_dummy_knowledge);
            if retrieval_function == retrieve_dummy_knowledge:
                import logging
                logging.info(f"Unable to find rerieval function for domain and source: {domain}, {source}.\nUsing dummy knowledge.")
                print("Dummy retrieval function... no data will be collected")
            # Get the knowledge
            tmp_knowl: str = retrieval_function(input, data_point, k, width, depth);

            knowl[domain][source] = tmp_knowl

    return knowl


def knowl_is_empty(knowl: Knowledge) -> bool:
    """
    Identify if the given dictionary of knowledge is empty.

    **Arguments**:
        knowl: dict[str, dict[str, str]] - The dictionary of knowledge.

    **Returns**:
        boolean - True or false depending on if there is knowledge in the dictionary.
    """
    
    for domain in knowl:
        for source in knowl[domain]:
            if knowl[domain][source] != '':
                return False
    return True