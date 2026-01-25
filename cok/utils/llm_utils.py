


from typing import List

import requests

from ollama import chat



def call_model(model: str, input_text: str, max_tokens: int = 256, temperature: float = 0.8, n: int = 1, local: bool = True) -> List[str]:
    """
    **Arguments**:
        model: str - The model being used to answer

        input_text: str - The text given to the model to generate a response from.
        
        max_tokens: int - The max tokens the answer can be (only used in GPT calls)
        
        temperature: int - The amount of random variation in the answer. Larger number means more variation, 0 will always generate the same response.
        
        n: int - The number of responses to generate.

    **Returns**:
        A list of strings representing the model's responses. 
        This list will be *n* entries long.
    """
    llm_answers: List[str] = [];

    # model = "deepseek-r1:70b";
    # model = "gpt-oss:120b";


    messages = [
      {
        'role': 'user',
        'content': input_text,
      },
    ]

    if local:
        for i in range(n):
            model_response = chat(model, messages=messages, think=False, options={"temperature": temperature, "thinking": "low"}).message.content
            # model_response = chat(model, messages=messages, think=False, options={"temperature": temperature}).message.content
                         
            print(f"Response {i}: {model_response}");

            llm_answers.append(model_response);
    else:
        print("TO-DO: Online LLM calling");


    return llm_answers;

    
