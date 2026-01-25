import json
import logging
import time
import requests
from requests import HTTPError, Response
from sentence_transformers import SentenceTransformer
from numpy import ndarray
from typing import Any



def retrieve_wikidata_knowledge(
        rational: str, 
        data_point: dict,
        k: int = 2, 
        width: int = 3, 
        depth: int = 2
    # ) -> list[str]:
    ) -> str:
    """
    Get KG-RAG knowledge from the wikidata embedding project.

    **Arguments**:
        rational: str - The rational to recieve information on.
        data_point: dict - A dictionary representing the json datapoint that will holds the current question's information.
        k: int - How many nodes to select at the initial query.
        width: int - How many properties to select per node for further checking.
        depth: int - How many times to explore further nodes.

    **Returns**:
        A list of strings representing the gathered knowledge from the nodes.

    """

    knowledge: list[str] = [];  # List of found knowledge.

    # Create the model now so it doesn't have to be re-created throughout the runtime
    model: str = "all-MiniLM-L6-v2"
    vectorising_model: SentenceTransformer = SentenceTransformer(model);   # Vectorising model used for finding relevant properties.

    # Vectorise the rational
    rational_embedding: ndarray = vectorising_model.encode([rational], normalize_embeddings=True)[0];

    # Get the response from the Wikidata Embedding Project
    wikidata_embedded_response: list[dict[str, Any]] = get_wikidata_embedded_response(rational, k);

    # Initialise variables for tracking exploration of knowledge graph
    cur_depth: int = 0;
    cur_explore_qids: list[str] = [];   # The list of QIDs that are currently being explored at this depth.
    next_explore_qids: list[str] = [];  # The list of QIDs that have been found at this depth, and will be explored next loop.
    explored_qids: list[str] = []; # The list of explored QIDs. Used to prevent re-exploring nodes, speed up program.

    # Pull all the QIDs for initial exploration
    for item in wikidata_embedded_response:
        cur_explore_qids.append(item["QID"]);

    # Ensure only explore to a maximum of 'depth'
    while cur_depth < depth:
        # Debug logging information
        logging.info(f"Cur depth: {cur_depth}");
        logging.info("QIDs to explore:")
        for qid in cur_explore_qids:
            logging.info(f"{qid}");

        # Create a final 'exploration list' to pass to the getter function
        qid_explore_list: list[str] = [];
        for qid in cur_explore_qids:
            if qid not in explored_qids:
                qid_explore_list.append(qid);
            else:
                logging.info(f"Found QID to remove from exploration: {qid}");
        
        items: dict[str, list[dict[str, dict[str, str]]]] = get_multiple_wikidata_properties(qid_explore_list);

        for qid, properties in items.items():
            # Update the explored QIDs to have this one.
            if qid not in explored_qids:
                explored_qids.append(qid);  # Should not get here if QID is explored, but just incase
            else:
                logging.error("ERROR: Unexpected QID in for loop. Already explored but still reached for loop.")

            # Get the names of the current property
            properties_names: list[str] = [];
            for property in properties:
                prop_label: dict[str, str] | None = property.get("propertyLabel", None);   # Get the property name
                if prop_label is not None:
                    prop_label_value: str | None = prop_label.get("value", None);
                    if prop_label_value is not None:
                        properties_names.append(prop_label_value);
                    else:
                        # If the value cannot be found
                        logging.info(f"ERROR: Unable to find value on property label: {prop_label}");
                else:
                    # If the property label itself cannot be found (this is bad and shouldn't happen)
                    logging.info(f"ERROR: Unable to find property label on property: {property}");
            
            # Get the top properties from the list of properties
            top_properties: list[str] = get_top_properties(properties_names, width, rational_embedding, vectorising_model);

            # Read current node's properties.
            current_node: list[dict[str, dict[str, str]]] = properties;

            for property in current_node:
                # Get the current property label
                property_label: str = property["propertyLabel"]["value"];   # The property name.
                
                # Only add the top properties to the knowledge
                if property_label in top_properties:

                    # Check if there is a wikipedia page for the current item
                    item_wikipedia: dict[str, str] | None = property.get("article", None);
                    item_label: str = "";
                    if item_wikipedia is not None: # If there was a value for the wikipedia page...
                        item_wikipedia_value: str = item_wikipedia["value"];  
                        # Set the name of the item to the wikipedia page's description so that it contains extra info on the specific version of the thing being looked at.
                        item_label = item_wikipedia_value.split("/")[-1].replace("_", " ");
                    else:
                        # Just grab the name from the itemLabel. Will not have as much info, but will always be there.
                        item_label = property["itemLabel"]["value"];   # The QID item's name.

                    value_label: str = property["valueLabel"]["value"]; # The value of the item's property's name!

                    # Check if there is a QID for this properties value.
                    value_qid = property.get("valueQID", None);
                    if value_qid is not None:
                        # If there is, add it to the list of QIDs to explore next.
                        value_qid_label: str = value_qid["value"];
                        # Ensure a blank QID isn't explored.
                        if value_qid_label:
                            next_explore_qids.append(value_qid_label);
                        
                    # Construct a string containing this information
                    if f"{item_label} has {property_label}" not in knowledge:
                        new_knowledge: str = f"{item_label} has {property_label} of {value_label}";

                    # Add this QID to the list of explored QIDs

                    knowledge.append(new_knowledge);     
        else:
            logging.info(f"Explored QID found: {qid}");               

        # Clear and update the list of QIDs to explore.
        cur_explore_qids.clear();
        cur_explore_qids = next_explore_qids.copy();    # Must use ".copy()" otherwise both variables point to the same place in memory!

        cur_depth += 1; # Increment the current depth.

    logging.info("Found the following knowledge:")
    for n in knowledge:
        logging.info(n);

    # Debug dump
    if logging.getLogger().isEnabledFor(logging.INFO):
        with open(f"knowledge_dump_{rational.split(' ')[0]}.txt", "w", encoding="utf-8") as f:
            for s in knowledge:
                f.write(f"{s}\n");
    

    # Convert output to a single string.
    out_s: str = "";
    
    for s in knowledge:
        out_s += s;
        out_s += ", ";
    
    # Clear memory to stop memory leaks
    import torch, gc
    del vectorising_model;
    gc.collect();
    torch.cuda.empty_cache();

    # Return the gathered knowledge
    # return knowledge;
    return out_s;



def retrieve_dummy_knowledge(rational: str, data_point, k: int = 0, width: int = 0, height: int = 0) -> str:
    """
    Dummy function for knowledge retrieval for unimplemented sections.

    **Returns**:
        ""
    """

    return "";




# ---------------------------------------------------------------------
# -                                                                   -
# -                        HELPER FUNCTIONS                           -
# -                                                                   -
# ---------------------------------------------------------------------

def get_multiple_wikidata_properties(qids: list[str], max_attempts: int = 5) -> dict[str, list[dict[str, dict[str, str]]]]:
    
    # Retrieve page from wikidata relating to the found QID
    wikidata_url = "https://query.wikidata.org/sparql";
    wikidata_headers = {
        "Accept": "application/sparql-results+json"
    }

    try:
        wikidata_response: Response = requests.get(wikidata_url, params={"query": build_sparql_query_multiple(qids=qids)}, headers=wikidata_headers);
        wikidata_response.raise_for_status();
    except Exception as e:
        print(f"Error while contacting wikidata: {e}\n----------\nWaiting and trying again...");
        time.sleep(10);
        wikidata_response: Response = requests.get(wikidata_url, params={"query": build_sparql_query_multiple(qids=qids)}, headers=wikidata_headers);
        wikidata_response.raise_for_status();


    items: dict[str, list[dict[str, dict[str, str]]]] = {};

    try:
        response: list[dict[str, dict[str, str]]] = wikidata_response.json()["results"]["bindings"];
    except Exception as e:
        raise Exception(f"something went wrong... e:{e}")


    for item in response:
        qid: str = item["item"]["value"].split("/")[-1];
        try:
            if item not in items[qid]:
                items[qid].append(item);
        except KeyError:
            items[qid] = [item]

    with open(f"request_debug_output_dict.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2);
            
    return items;


def get_wikidata_embedded_response(rational: str, num_matches: int) -> list[dict[str, Any]]:
    """
    Calls the Wikidata Embedding Project API to get the *num_matches* most similar Wikidata items to the given rational.

    **Arguments**:
        rational: str - The rational to find matches on
        num_matches: int - How many matches to find

    **Returns**:
        The GET response converted to json.  
    """
    from dotenv import load_dotenv
    import os

    # Load the .env file to get the Wikidata embedding project's API key
    load_dotenv();

    # Get the API key
    api_key: str | None = os.getenv("WIKIDATA_EMBEDDING_PROJECT_API_KEY");

    # Check that API key was successfully created
    if api_key is None:
        raise ValueError("Wikidata Embedding API not found.");

    # Build request to send to db
    url: str = "https://wd-vectordb.wmcloud.org/item/query/"; # <-- currently only searching on items, but can search on properties
    params: dict = {
        "query": rational,
        "K": num_matches
    }
    headers: dict = {
        "x-api-secret": api_key,
        "User-Agent": "tmc-honours (thomas.childs@curtin.edu.au)"
    }


    # Response validation values
    retry_time: int = 5;
    max_retry: int = 20;
    cur_retry: int = 0;
    
    # Loop until the response is a pass (or raise error)

    status_check: int = 0;

    while(status_check != 200):
        # Get response
        response: Response = requests.get(url, params=params, headers=headers, timeout=50);
        status_check = response.status_code;

        # Based on response...
        if status_check == 401:
            raise ValueError("Invalid Wikidata Embedding API code.");
        elif status_check == 500:
            print(f"Response: {response.text}")
            print(f"Wikidata ERROR 500: Waiting {retry_time} before retrying...");
            time.sleep(retry_time);
        elif status_check != 200:   # Check it IS NOT 200 before raising the error... Because the response is gotten within the while loop.
            print(f"An error occured while contacing thee Wikidata Embedding Project:\nQuery: {rational}\nK:{num_matches}")
            print(f"Response: {response.text}");
            raise ValueError(f"Invalid construction of request to Wikidata Embedding API\n Status: {status_check}\nResponse:{response.text}");

        cur_retry += 1;
        if cur_retry >= max_retry:
            raise ValueError(f"Unable to contact Wikidata after {cur_retry} attempts!")

    # All validated, pull information.
    return response.json();


def get_wikidata_properties(qid: str) -> list[dict]:
    """
    Returns the properties/relations that exist on the wikidata object found at the given QID.

    **Arguments**:
        qid: str - The QID of the wikipedia object to pull the properties from.

    **Returns**:
        properties: list[dict] - The list of properties fonund for the object.

    **Errors**:
        ValueError - If the response to Wikidata does not succeed.
        KeyError - If there are no bindings on the Wikidata object (should be very rare)
    """
    
    # Retrieve page from wikidata relating to the found QID
    wikidata_url = "https://query.wikidata.org/sparql";
    wikidata_headers = {
        "Accept": "application/sparql-results+json"
    }
    wikidata_response: Response = requests.get(wikidata_url, params={"query": build_sparql_query(qid=qid)}, headers=wikidata_headers);
    wikidata_response.raise_for_status();

    # UNCOMMENT THE BELLOW TO SEE DEBUG DUMPS
    # print(f"{i} response: {wikidata_response.json()}");
    # with open(f"request_debug_output_{i}.txt", "w", encoding="utf-8") as f:
        # json.dump(wikidata_response.json(), f, ensure_ascii=False, indent=2);
    
    # Validate response
    if wikidata_response.status_code != 200:
        raise ValueError(f"Response did not succeed: {wikidata_response.status_code}");

    # Put the wikidata response into a json object and pull the properties of the item from it
    wikidata_json: dict[str, Any] = wikidata_response.json();
    
        # NOTE: In this GET request, the resulting request object will have the following format:
        # {
        #   "head": {
        #     "vars": List of variables being grabbed by the SQL search
        #   },
        #   "results": {
        #     "bindings": A list of dictionaries for each property of the SPARQL search.
        #   }
        # }
        #
        # I pull the 'bindings' section out into the 'properties' variable to make more semantic sense
        # and easily access it for the next step.

    # Pull the properties of the item out from the 'bindings' section of the json. 
    properties: list[dict[str, Any]] = wikidata_json["results"]["bindings"];

    return properties

def get_top_properties(
        properties: list[str], 
        num_top: int, 
        rational_embedding: ndarray, 
        vectorising_model: SentenceTransformer) -> list[str]:
    """
    Returns a list of *num_top* properties that are most similar to the given rational.

    **Arguments**:
        properties: list[dict] - Should be a list of properties gotten from a SPARQL query to wikidata.
        num_top: int - How many properties to return. I.e. the length of the list of *top_properties*.
        rational: str - The given rational to compute similarity between.
        model: str - Which SentenceTransformer model to use. Default of "all-MiniLM-L6-v2".

    **Returns**:
        top_properties: list[str] - A list of the most similar properties, with a length of *num_top*.
    """
    import numpy as np

    top_properties: list[str] = [];

    # Vectorise the property labels
    property_embeddings = vectorising_model.encode(properties, normalize_embeddings=True);
    # Compute similarity of properties and select 'num_top' properties to explore further.
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b):   # Function used to calculate the similarity (the encoded property names are vectors)
        return dot(a, b) / (norm(a) * norm(b));
    # Similarity scores
    scores = [cosine_sim(rational_embedding, pe) for pe in property_embeddings];

    # The list of 'num_top' properties with the highest scores.
    # top_properties_numbers: list[int] = np.argsort(scores)[::-1][:num_top]; <--- if the below breaks, go back to this
    top_properties_numbers: list[int] = np.argsort(scores)[::-1][:num_top].tolist();
    for num in top_properties_numbers:
        # Log the names and their scores
        logging.info(f"{properties[num]} (score: {scores[num]:.3f})");
        top_properties.append(properties[num]);
    
    return top_properties;

def build_sparql_query_multiple(qids: list[str]) -> str:
    """
    Constructs a SPARQL query for multiple QIDs.
    """

    # Turn ["Q42", "Q1", "Q27037218"] into: wd:Q42 wd:Q1 wd:Q27037218
    qid_values = " ".join(f"wd:{qid}" for qid in qids)

    return f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>

    SELECT ?item ?itemLabel ?property ?propertyLabel ?value ?valueLabel ?valueQID ?article
    WHERE {{
        VALUES ?item {{ {qid_values} }}
        ?item ?p ?value .
        ?property wikibase:directClaim ?p .

        BIND(
            IF(STRSTARTS(STR(?value), "http://www.wikidata.org/entity/"),
                REPLACE(STR(?value), "http://www.wikidata.org/entity/", ""),
                ""
            ) AS ?valueQID
        )

        OPTIONAL {{
            ?article schema:about ?item .
            ?article schema:inLanguage "en" .
            ?article schema:isPartOf <https://en.wikipedia.org/> .
        }}

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

def build_sparql_query(qid: str) -> str:
    """
    Constructs a sparql query including the provided qid.
    """
    return f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>

    
    SELECT ?itemLabel ?property ?propertyLabel ?value ?valueLabel ?valueQID ?article
    WHERE {{
        BIND(wd:{qid} AS ?item)
        ?item ?p ?value .
        ?property wikibase:directClaim ?p .

        BIND(
            IF(STRSTARTS(STR(?value), "http://www.wikidata.org/entity/"),
                REPLACE(STR(?value), "http://www.wikidata.org/entity/",""),
                ""
            ) AS ?valueQID
        )

        OPTIONAL {{
            ?article schema:about ?item .
            ?article schema:inLanguage "en" .
            ?article schema:isPartOf <https://en.wikipedia.org/> .
        }}

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """



# TESTING OF THE GET CALLS
if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,  # Show debug messages while testing!
        format="[%(levelname)s] %(message)s"
    );



    # Retrieve page from wikidata relating to the found QID
    # wikidata_url = "https://query.wikidata.org/sparql";
    # wikidata_headers = {
    #     "Accept": "application/sparql-results+json"
    # }

    # qid = "Q106615149"
    # r = get_wikidata_properties(qid);

    # r = requests.get(wikidata_url, params={"query": build_sparql_query(qid=qid)})
    # r.raise_for_status();

    

    # qids = ["Q20", "Q131569", "Q35"];

    # wikidata_response: Response = requests.get(wikidata_url, params={"query": build_sparql_query_multiple(qids=qids)}, headers=wikidata_headers);
    # wikidata_response.raise_for_status();

    # get_multiple_wikidata_properties(qids);

    # # UNCOMMENT THE BELLOW TO SEE DEBUG DUMPS
    # print(f"response: {r.json()}");
    # with open(f"request_debug_output_Q106615149.json", "w", encoding="utf-8") as f:
        # json.dump(r, f, ensure_ascii=False, indent=2);


    # Test the knowledge retrieval

    # prompt = "Polar bears are not endangered."
    prompt = "Who wrote \"Here comes the sun\"?"
    # prompt = "Chumlee is a nickname given to Austin Russell on the show, but there is no information indicating that Chumlee is his real name."
    # prompt = "Austin Russell's full name is Austin Lee Russell, and he is known for his role in the television show \"Pawn Stars.\""

    retrieve_wikidata_knowledge(prompt, {}, 5, 5, 3);