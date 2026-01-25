import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from cok.utils.llm_utils import call_model
from cok.utils.other_prompts import domain_selection_demonstration
# from utils.fetaqa_eval import main as fetaqa_eval    <-- NOT IMPLEMENTED YET

# init gloabl variables
import cok.utils.globalvar
from cok.utils.retrieval.wikidata_embedding import retrieve_wikidata_knowledge
cok.utils.globalvar.init()

import os


def direct_llm_answer(dataset, data_point, model, dataset_name):
    print("****************** Start stage 0: direct LLM response ...")
    question: str = "";
    if dataset_name == "fever":    
        question = f"""
        If the following statement is true, answer with only SUPPORTS.
        If the following statement is false, answer with only REFUTES.
        If the following statement is not verifiable, answer with only NOT ENOUGH INFO.

        {dataset.get_question(data_point)}
        """
        print("****** Question:", question);
    elif dataset_name == "hotpotqa":
        question = f"""
        Answer the following question without explanation. Only write your answer. Write succinctly:

        {dataset.get_question(data_point)}
        """
    else:
        question = f"{dataset.get_question(data_point)}";

    # Retrieve LLM response
    direct_response: str = call_model(model, question, temperature=0)[0]
    print(f"****** Direct response: {direct_response}");
    data_point["direct_response"] = direct_response.strip();

    return data_point;

def kg_rag_llm_answer(dataset, data_point, model, k, width, depth, dataset_name):
    print("****************** Start stage 0: direct kg-rag LLM response ...")

    # Get the knowledge for the question
    knowledge: str = retrieve_wikidata_knowledge(dataset.get_question(data_point), data_point, k, width, depth);
    
    if dataset_name == "fever":
        question: str = f"""
        If the following statement is true, answer with only SUPPORTS.
        If the following statement is false, answer with only REFUTES.
        If the following statement is not verifiable, answer with only NOT ENOUGH INFO.

        Potentially relevant information: {knowledge}

        {dataset.get_question(data_point)}
        """
    elif dataset_name == "hotpotqa":
        question = f"""
        Answer the following question without explanation. Only write your answer. Write succinctly:
        
        Potentially relevant information: {knowledge}

        {dataset.get_question(data_point)}
        """
    else:
        question = f"""
        Potentially relevant information: {knowledge}

        {dataset.get_question(data_point)}

        """

    print("****** Question:", question);

    # Retrieve LLM response
    direct_response: str = call_model(model, question, temperature=0)[0]
    print(f"****** Direct KG-RAG response: {direct_response}");
    data_point["direct_kg_rag_response"] = direct_response.strip();

    return data_point;

def s1_reasoning_preparation(dataset, data_point, model, threshold):
    print("****************** Start stage 1: reasoning preparation ...")
    question = dataset.get_question(data_point)
    print("****** Question:", question)

    ### Domain selection
    domain_selection_prompt = domain_selection_demonstration + "Q: " + question.strip() + "\nRelevant domains: "
    domain_selection_response = call_model(model, domain_selection_prompt, max_tokens=256, temperature=0)
    
    if domain_selection_response is not None:
        domain_selection_text_response = domain_selection_response[0].strip()
        print("****** Relevant domains:", domain_selection_text_response)
        data_point["s1_domains"] = [x.strip() for x in domain_selection_text_response.split(",")]
    
    ### CoT generation
    cot_prompt = dataset.get_s1_prompt(question)
    
    data_point: dict = dataset.get_cot_sc_results(data_point, model, cot_prompt)
    try:
        print("****** CoT answer:", data_point["cot_response"])
        print("****** CoT SC score:", data_point["cot_sc_score"])
        print("****** CoT SC answer:", data_point["cot_sc_response"])
    except Exception as e:
        print("Error during s1_reasoning_preparation()");
        if data_point.get("fail"):
            print("Failed due to failure on this question in processing.");
        else:
            print(f"UNSURE WHY FAILED, THROWING EXCEPTION NOW: {e}")
            raise Exception(e);

    return data_point


def s2_knowledge_adapting(dataset, data_point, model, step, k: int, width: int, depth: int):
    print("****************** Start stage 2: knowledge adapting ...")
    if step:
        print("****** Edit mode: Step by step")
        # Edit the rationales step by step
        data_point = dataset.update_rationales_step_by_step(model, data_point, k, width, depth);

    else:
        # Edit the rationales all at once
        print("****** Edit mode: At once")
        # Edit the rationales step by step
        data_point = dataset.update_rationales_at_once(model, data_point, k, width, depth)

    return data_point

def s3_answer_consolidation(dataset, data_point, model):
    print("****************** Start stage 3: answer consolidation ...")
    data_point = dataset.get_final_answer(model, data_point)
    return data_point





if __name__ == "__main__":

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613", help="Model to use when answering questions.")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    # parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--step", type=bool, default=False, help="Whether to edit the rationales step by step")
    parser.add_argument("--num_train", type=int, default=3, help="How many demonstration samples to use")
    parser.add_argument("--num_test", type=int, default=5, help="How many test samples to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="sc threshold for ka answer")
    parser.add_argument("--one_shot", action="store_true", help="Whether to use 1-shot setting")
    parser.add_argument("--six_shot", action="store_true", help="Whether to use 6-shot setting")
    parser.add_argument("--debug", help="Whether to display debug messages or not.")
    parser.add_argument("--retest", type = bool, default = False, help = "When flagged, will always re-test KG-CoK answers (i.e. all answers with sc < threshold)")
    parser.add_argument("--k", type = int, default = 2, help = "Defines the initial width of retrieved Wikidata nodes during KG-RAG");
    parser.add_argument("--width", type = int, default = 3, help = "Defines the width to search through each node during KG-RAG");
    parser.add_argument("--depth", type = int, default= 2, help = "Defines the maximum node depth from the initial nodes K to explore during KG-RAG");
    parser.add_argument("--base_test", type = bool, default = False, help = "When flagged, will use only a direct query to local LLM.")
    parser.add_argument("--kgrag_test", type = bool, default = False, help = "When flagged, will test directly KG-RAG");

    args = parser.parse_args()

    # Flag a retest variable
    retest: bool = False;
    if args.retest:
        retest = True;
    
    # Flag base-test variable
    base_test: bool = False;
    if args.base_test:
        base_test = True;
    
    # Flag kgrag test
    kgrag_test: bool = False;
    if args.kgrag_test:
        kgrag_test = True;

    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s"
        )

    """
    The below has been altered as this implementation of the framework is only running the factual databases
    """
    # TODO: add other datasets, as well as a parser for each dataset
    if args.dataset == "hotpotqa":
        from cok.utils.hotpotqa_parser import hotpotqa
        dataset = hotpotqa()
    # elif args.dataset == "medmcqa":
        # from utils.medmcqa_parser import medmcqa
        # dataset = medmcqa()
    # elif args.dataset == "mmluphy":
        # from utils.phy_parser import phy
        # dataset = phy()
    # elif args.dataset == "mmlubio":
        # from utils.bio_parser import bio
        # dataset = bio()
    elif args.dataset == "fever":
        from cok.utils.fever_parser import fever
        dataset = fever(six_shot=args.six_shot, one_shot=args.one_shot)
    # elif "feta" in args.dataset:
        # from utils.fetaqa_parser import select_fetaqa_dataset
        # dataset = select_fetaqa_dataset(args.dataset, num_train=args.num_train)
    else:
        raise Exception(f"Invalid dataset name: '{args.dataset}'")


    # Create folder to store test if it doesn't already exist
    output_file_location: str = f"testing_outputs/{args.dataset}";
    if not os.path.exists(output_file_location):
        os.mkdir(output_file_location);


    # Create filename from the inputs of k, width, depth, model, num_test and threshold
    if base_test:
        output_file_name: str = f"{output_file_location}/output_{args.dataset}_{args.num_test}_{args.model.replace(':','').upper()}_direct_test.json"
    elif kgrag_test:
        output_file_name: str = f"{output_file_location}/output_{args.dataset}_{args.num_test}_{args.model.replace(':','').upper()}_kgrag_test.json"
    else:
        output_file_name: str = f"{output_file_location}/output_{args.dataset}_{args.num_test}_{args.model.replace(':','').upper()}_k{args.k}_w{args.width}_d{args.depth}_t0{(int)(args.threshold * 10)}.json";
    print(output_file_name);

    # load data
    Path(output_file_name).parent.mkdir(exist_ok=True, parents=True)
    data = dataset.get_dataset()
    print('original data length:', len(data))
    
    # Check if the output file already exists.
    if os.path.exists(output_file_name) and not base_test:
        print('Found existing outputs, will replace the original data with the existing outputs')
        
        # Read existing outputs
        output_data = json.load(open(output_file_name, "r"))
        print('Found {} existing outputs'.format(len(output_data)))
        
        # Update the results of the loaded dataset with the previously created results. 
        replace_count = 0
        for d in output_data:
            for i in range(len(data)):
                if d['question'] == data[i]['question']:
                    data[i] = d
                    replace_count += 1
                    break
        print('replaced {} existing outputs'.format(replace_count))
        print('Found {} prepared outputs.'.format(len([x["id"] for x in data if 'cot_answer' in x])))
        print('Found {} outputs that need to be edited.'.format(len([x["id"] for x in data if 'cot_answer' in x and 'fail' not in x and x["cot_sc_score"] < args.threshold and 'final_answer' not in x])))
        print('Found {} edited outputs.'.format(len([x["id"] for x in data if 'final_answer' in x])))
    else:
        print(f"No matching test file found for: {output_file_name}. Creating new file...")

    # Loop through all the questions in the dataset.
    count = 0
    failed = 0;
    try:
        if not base_test and not kgrag_test:
            for i in tqdm(range(min(args.num_test, len(data)))): #<-- tdqm is used for the pretty progress bar on command line
                print("####################################", i, "####################################")
                data_point = data[i]
                data_point["id"] = i

                # if args.dataset == "fetaqa" or args.dataset == "fetaqa_query":
                #     question = data_point["question"]
                #     cot_prompt = dataset.get_s1_prompt(question)
                #     data_point = dataset.get_cot_sc_results(data_point, args.model, cot_prompt)
                #     print("****** CoT answer:", data_point["cot_response"])
                #     print("****** CoT SC score:", data_point["cot_sc_score"])
                #     print("****** CoT SC answer:", data_point["cot_sc_response"])
                #     data[i] = data_point
                #     with open(output_file_name, "w") as f:
                #         json.dump(data, f)
                #     continue
                

                # Add filtering to ensure we have not previously produced the results
                if 'cot_sc_score' not in data_point:
                    ##### run stage 1: reasoning preparation
                    data_point = s1_reasoning_preparation(dataset, data_point, args.model, args.threshold)
                    
                    # update the datapoint
                    data[i] = data_point
                    
                    with open(output_file_name, "w") as f:
                        json.dump(data, f)

            
                # Self-consistency threshold
                if not retest:  # If the test is being run for the first time
                    correct_rationals = 'fail' not in data_point and data_point["cot_sc_score"] < args.threshold and 'final_answer' not in data_point and 'fail' not in data_point;
                else:   # The test has been run before, and the rational generation is being re-done
                    correct_rationals = data_point["cot_sc_score"] < args.threshold;
                
                if correct_rationals:   # only run if the final answer not already generated & did not fail previously
                    try:
                        # Implement CoK only when the score is lower than threshold
                        
                        ##### run stage 2: knowledge adapting
                        data_point = s2_knowledge_adapting(dataset, data_point, args.model, args.step, args.k, args.width, args.depth)

                        # update the datapoint
                        data[i] = data_point

                        with open(output_file_name, "w") as f:
                            json.dump(data, f)

                        ##### run stage 3: answer consolidation
                        data_point = s3_answer_consolidation(dataset, data_point, args.model)

                        # update the datapoint
                        data[i] = data_point

                        with open(output_file_name, "w") as f:
                            json.dump(data, f)
                    except MemoryError as me:
                        print(f"MEMORY ERROR: {me}\n\nQuitting program to prevent continuous fails");
                        raise MemoryError(me);
                    except Exception as e:
                        # print("EXCEPTION HANDLING IS TURNED OFF IN CODE")
                        # raise e;
                        # If a major error has occured while trying to update the answer, flag this question as 'failed' and move on.

                        if "CUDA out of memory" in str(e):
                            print("Updating answer failed with CUDA memory issue. Exiting...");
                            raise MemoryError(e);
                        
                        print(f"Updating answer failed: {e}. Moving on without changing original response...");
                        
                        # Increment fail counter
                        failed += 1;

                        # Update datapoint
                        data_point["fail"] = True;
                        data_point["fail_reason"] = str(e);
                        data[i] = data_point;
                        
                        # Dump outputs with flag
                        with open(output_file_name, "w") as f:
                            json.dump(data, f)
                else:
                    if 'fail' in data_point:
                        failed += 1;
                    else:
                        count += 1
                        continue

                # if "feta" in args.dataset:
                    # fetaqa_eval(output_file_name)
        elif kgrag_test:
            # Do some direct kg-rag testing of the model
            for i in tqdm(range(min(args.num_test, len(data)))): #<-- tdqm is used for the pretty progress bar on command line
                print("####################################", i, "####################################")
                data_point = data[i]
                data_point["id"] = i

                if "direct_kg_rag_response" in data_point:
                    print(f" ##{i}## Found previous response. Skipping...");
                    continue;

                try:
                    data_point = kg_rag_llm_answer(dataset, data_point, args.model, args.k, args.width, args.depth, args.dataset);
                except Exception as e:
                    print(f"##{i}## ERROR: {e}");
                    print("Ran into issue while exploring this question. Skipping...");
                    continue;

                data[i] = data_point;

                # Dump outputs with flag
                with open(output_file_name, "w") as f:
                    json.dump(data, f)
        else:
            # Do some direct testing of the model
            for i in tqdm(range(min(args.num_test, len(data)))): #<-- tdqm is used for the pretty progress bar on command line
                print("####################################", i, "####################################")
                data_point = data[i]
                data_point["id"] = i


                data_point = direct_llm_answer(dataset, data_point, args.model, args.dataset);
    
                data[i] = data_point;

                # Dump outputs with flag
                with open(output_file_name, "w") as f:
                    json.dump(data, f)

    except Exception as e:
        import traceback
        print(f"MAJOR ERROR: {e}\nError occured at: {traceback.print_tb(e.__traceback__)}\n-------\nDumping output so far incase error prevented previous dump...")
        with open(output_file_name, "w") as f:
                    json.dump(data, f)


    print("Number of skipped samples (high consistency): ", count)
    print("Number of failed updates (no knowledge adapting performed): ", failed);
    print("ALL DONE!!")

    # log_file.close();

"""
p run.py --dataset fetaqa --output outputs/fetaqa_full.json --num_test 500
p run.py --dataset fetaqa --output outputs/fetaqa.json --num_test 100
p run.py --dataset fetaqa_editing --output outputs/fetaqa_editing.json --num_test 100 --step True --threshold 0.3
"""