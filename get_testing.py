import requests;
from requests import Response;
from dotenv import load_dotenv
import os

rational: str = "The sky is blue"

# Load the .env file to get the Wikidata embedding project's API key
load_dotenv();
# Get the API key
api_key: str | None = os.getenv("WIKIDATA_EMBEDDING_PROJECT_API_KEY");
# Check that API key was successfully created
if api_key is None:
    raise ValueError("Wikidata Embedding API not found.");

print("API KEY: ", api_key)
# Query Wikidata through the embedding project with the rational.
updated_rational: str = rational; # Set to the original rational incase updating fails - falls back on the original thoughts. 
# Build request to send to db
url: str = "https://wd-vectordb.toolforge.org/property/query/";
params: dict = {
    "query": rational,
    "K": 10
}
headers: dict = {
    "x-api-secret": api_key
    
}
response: Response = requests.get(url, params=params, headers=headers);
print("response code: ", response.status_code);
print(response.json());
# raise ValueError("no error here i just wanna see the response");
# Return the updated rational.
# return updated_rational;