# 
#   C U S T O M _ Q U E R Y . P Y
#   ------------------------------
# 
#   This file is intended to be used to get a CoK response from a local LLM for a given question.
# 

def get_cot_query(query: str) -> str:
    llm_cot_query: str = f"""
    Create a chain of thought of two steps to answer the final question.
    
    Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in 
    what year
    A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second,
    Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991.
    
    Q: This band released a song Wonderwall from an album called what?
    A: First, the song Wonderwall was released by the band Oasis. Second, the Oasis album with 
    Wonderwall on it was called What's the Story Morning Glory.
    The answer is What's the Story Morning Glory.
    
    Q: This liquid is essential for human life, is drunk by everyone alive and has what molecular makeup
    A: First, water is an essential for human life and is drunk by everyone. Second, water's molecular makeup is H2O. 
    The answer is H20.
    
    Q: {query}
    A:

    """

    return llm_cot_query;
def get_next_rational_query(query: str, rational_1: str) -> str:
    llm_next_rational_q: str = f"""
    Create a chain of thought of two steps to answer the final question.
    
    Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in 
    what year
    A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second,
    Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991.
    
    Q: This band released a song Wonderwall from an album called what?
    A: First, the song Wonderwall was released by the band Oasis. Second, the Oasis album with 
    Wonderwall on it was called What's the Story Morning Glory.
    The answer is What's the Story Morning Glory.
    
    Q: This liquid is essential for human life, is drunk by everyone alive and has what molecular makeup
    A: First, water is an essential for human life and is drunk by everyone. Second, water's molecular makeup is H2O. 
    The answer is H20.
    
    Q: {query}
    A: {rational_1}. Second,

    """

    return llm_next_rational_q;

def get_rational_correct_query(rational: str, knowledge: str) -> str:
    llm_rational_correct_query: str = f"""

    Strictly follow the format of the below examples. The given sentence may have factual errors,
    please correct them based on the given external knowledge.
    Sentence: the Alpher-Bethe-Gamow paper was invented by Ralph Alpher.
    Knowledge: discoverer or inventor of Alpher-Bethe-Famow paper is Ralph Alpher.
    Edited sentence: the Alpher-Bethe-Gamow paper was invented by Ralph Alpher.
    Sentence: Ralph Alpher was advised by Hans Bethe.
    Knowledge: doctoral advisor of Ralph Alpher is George Gamow.
    Edited sentence: Ralph Alpher was advised by George Gamow.
    Sentence: {rational}
    Knowledge: {knowledge}
    Edited sentence:

    """

    return llm_rational_correct_query;


#######################################################################################################


from ollama import chat
from cok.utils.retrieval.wikidata_embedding import retrieve_wikidata_knowledge


llm: str = input("Enter LLM to be used: ");
query: str = input("Enter the query to be parsed: ");

k: int = 1;
w: int = 1;
d: int = 1;

if "gemm3:27b" in llm:
    k = 1;
    w = 1;
    d = 2;
elif "deepseek-r1:70b" in llm:
    k = 2;
    w = 3;
    d = 3;
elif "gpt-oss:120b" in llm:
    k = 2;
    w = 3;
    d = 1;

print(f"Query: {get_cot_query(query)}");

messages: dict[str, str] = [
    {
    'role': 'user',
    'content': get_cot_query(query),
    },
]

llm_cot: str | None = chat(llm, messages=messages, think=False, options={"temperature": 0, "thinking": "low"}).message.content;

print(f"RESPONSE: {llm_cot}");

try:
    rational_1: str = llm_cot.split("Second,")[0];
    print(f"Rational 1: {rational_1}");
    rational_2: str = llm_cot.split("Second,")[1];
    print(f"Rational 2: {rational_2}");
except Exception as e:
    print("failed to split string");

knowledge: str = retrieve_wikidata_knowledge(rational_1, None, k, w, d);

rational_corret_q:str  = get_rational_correct_query(rational_1, knowledge);
messages: dict[str, str] = [
    {
    'role': 'user',
    'content': rational_corret_q,
    },
]
llm_rational_1_correct: str | None = chat(llm, messages=messages, think=False, options={"temperature": 0, "thinking": "low"}).message.content;

print(f"Rational 1 correct: {llm_rational_1_correct}");

messages: dict[str, str] = [
    {
    'role': 'user',
    'content': get_next_rational_query(query, llm_rational_1_correct),
    },
]
llm_second_rational_gen: str | None = chat(llm, messages=messages, think=False, options={"temperature": 0, "thinking": "low"}).message.content;

print(f"Final explanation: {llm_second_rational_gen}");


final_prompt: str = f"""
Using only the knowledge provided here, please explain in detail the answer to this question.
Do your best, the information may be incomplete.

Question: {query}
Knowledge: {llm_second_rational_gen}
"""

print(f"final_prompt: {final_prompt}");

messages: dict[str, str] = [
    {
    'role': 'user',
    'content': final_prompt,
    },
]
llm_final_answer: str | None = chat(llm, messages=messages, think=False, options={"temperature": 0, "thinking": "low"}).message.content;

print(f"Final KG-CoK answer: {llm_final_answer}")


kg_rag_prompt: str = f"""
Using only the knowledge provided here, please explain in detail the answer to this question.
Do your best, the information may be incomplete.

Question: {query}
Knowledge: {knowledge}
"""

print(f"Final KG-RAG prompt: {kg_rag_prompt}")

messages: dict[str, str] = [
    {
    'role': 'user',
    'content': kg_rag_prompt,
    },
]
llm_final_answer_kg_rag: str | None = chat(llm, messages=messages, think=False, options={"temperature": 0, "thinking": "low"}).message.content;

print(f"Final KG-RAG answer: {llm_final_answer_kg_rag}");