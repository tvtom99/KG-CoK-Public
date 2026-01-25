import argparse
import json
from typing import Any, List

if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fever", type=str, help="Dataset name")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--threshold", type=float, help="consistency threshold")

    args = parser.parse_args(); 

    threshold: int = args.threshold
    
    if args.dataset == "fever":
        from cok.utils.fever_parser import fever
        dataset = fever()
    # Re-added support for hotpotqa
    elif args.dataset == "hotpotqa":
        from cok.utils.hotpotqa_parser import hotpotqa;
        dataset = hotpotqa();
    else:
        raise Exception("Invalid dataset name")
    
    print(f"DISPLAYING DATA FOR FILE: {args.output}")

    data: List[dict[str, Any]] = json.load(open(args.output, "r"))
    print(f'Found {len(data)} testing outputs')

    # Assume it is a base test if the direct_response field is found
    base_test: bool = False;
    kg_test: bool = False;
    for d in data:
        if "direct_response" in d:
            base_test = True;
            print("found direct response")
            break;
        elif "direct_kg_rag_response" in d:
            kg_test = True;
            print("found kg_test response");
            break;
    
    edited_count = 0
    domains = {}
    accuracy_records = {'plain': 0, 'cot': 0, 'cot_sc': 0, 'ka': 0, 'ka_standalone': 0, 'direct_response': 0, 'direct_kg_rag_response': 0}
    num_failed: int = 0;
    sc_records = [];    # A record of all self-consistency scores for all questions
    

    if not base_test and not kg_test:
        for i, example in enumerate(data):
            if "cot_answer" in example and "fail" not in example:

                
                cur_q = example["question"]

                # Get the ground-truth answer for the example
                if "label" in example:
                    ground_truth = example["label"];
                else:
                    ground_truth = dataset.get_ans_text(example["answer"]);

                # use get_ans_texamplet to normalize all answers: lower case, etc
                answer: str = dataset.get_ans_text(example['answer'])
                cot_answer: str = dataset.get_ans_text(example['cot_answer'])
                cot_sc_answer: str = dataset.get_ans_text(example.get('cot_sc_answer',' . '))
                ka_answer: str = "";
                
                # Add the self-consistency score of this example to the record of all values
                sc_records.append(example['cot_sc_score']);
                

                # Check if the answer has been generated through KG-CoK
                if example['cot_sc_score'] <= threshold:
                    if 'final_answer' in example and "fail" not in example:   # A final answer was generated! KG-CoK was ran
                        ka_answer: str = example['final_answer']; 
                        edited_count += 1
                        accuracy_records['ka_standalone'] += (ground_truth in ka_answer)
                        for domain in example['s1_domains']:
                            if domain not in domains:
                                domains[domain] = 0
                            domains[domain] += 1
                    elif "fail" in example:
                        # A fail tag has been added, KG-CoK for this section failed... Treat it as if this node is just CoT-SC
                        ka_answer = cot_sc_answer
                    else:
                        print('instance id: ', example['id'])
                        print('cot_sc_score: ', example['cot_sc_score'])
                        raise Exception('Cannot find edited final answer for the above instance.')
                else:
                    ka_answer = cot_sc_answer
                
                # Update the stores of correct answers
                if ground_truth.lower().strip() in answer.lower().strip():
                    accuracy_records['plain'] += 1
                else:
                    print(f"Somehow ground_truth was wrong at {i}: {ground_truth}, answer: {answer}");
                
                if ground_truth.lower().strip() in cot_answer.lower().strip():
                    accuracy_records['cot'] += 1;
                # elif cot_answer in ground_truth.lower().strip():
                #     print(f"QUESTION: {cur_q}");
                #     print(f"cot_answer was a portion of the ground truth. COT: {cot_answer} GT: {ground_truth.lower().strip()}");
                
                if ground_truth.lower().strip() in cot_sc_answer.lower().strip():
                    accuracy_records['cot_sc'] += 1; 
                # elif cot_sc_answer in ground_truth.lower().strip():
                #     print(f"QUESTION: {cur_q}");
                #     print(f"cot_sc_answer was a portion of the ground truth. COT-SC: {cot_sc_answer} GT: {ground_truth.lower().strip()}");
                
                if ground_truth.lower().strip() in ka_answer.strip().lower():
                    accuracy_records['ka'] += 1; 
                # elif ka_answer in ground_truth.lower().strip():
                #     print(f"QUESTION: {cur_q}");
                #     print(f"ka_answer was a portion of the ground truth. KA: {ka_answer} GT: {ground_truth}");
            else:
                answer: str = dataset.get_ans_text(example['answer'])
                
                if "label" in example:
                    ground_truth = example["label"];
                else:
                    ground_truth = dataset.get_ans_text(example["answer"]);

                if ground_truth in answer:
                    accuracy_records['plain'] += 1

    elif kg_test:
        print("Results are based on KG-RAG direct response. ");
        for i, example in enumerate(data):
            if args.dataset == "fever":
                ground_truth: str = example["label"];
            else:
                ground_truth: str = example["answer"];
            try:
                model_answer: str = example["direct_kg_rag_response"];
            except Exception as e:
                print(f"No response on response: {i}")
    
            if ground_truth.lower().strip() in model_answer.lower().strip():
                accuracy_records['direct_kg_rag_response'] += 1;
    else:
        print("Results are based on only a direct LLM response. ");
        for i, example in enumerate(data):
            if args.dataset == "fever":
                ground_truth = example["label"];
            elif args.dataset == "hotpotqa":
                ground_truth = example["answer"];
            else:
                ground_truth = example["label"];
            model_answer = example["direct_response"];
    
            if ground_truth.lower().strip() in model_answer.lower().strip():
                accuracy_records['direct_response'] += 1;

    # Print information
    print('##############################################')
    
    print(f"accuracy_records count full: {accuracy_records['plain']}")
    print(f"accuracy_records count cot: {accuracy_records['cot']}")
    print(f"accuracy_records count cot_sc: {accuracy_records['cot_sc']}")
    print(f"accuracy_records count ka: {accuracy_records['ka']}")
    print(f"accuracy_records count direct: {accuracy_records['direct_response']}")
    print(f"accuracy_records count direct_kg_rag_response: {accuracy_records['direct_kg_rag_response']}")

    print('##############################################')

    print('full acc: ', accuracy_records['plain'] / len(data) * 100, "%");
    print('cot acc: ', accuracy_records['cot'] / len(data) * 100, "%");
    print('cot_sc acc: ', accuracy_records['cot_sc'] / len(data) * 100, "%");
    print(f"ka acc:  {(accuracy_records['ka'] / len(data) * 100):.1f}%");
    try:
        print(f"ka acc when an edit is made: {(accuracy_records['ka_standalone'] / edited_count * 100):.1f}%")
    except Exception as e:
        print("ka acc: untested");
    print(f"direct_response acc:  {(accuracy_records['direct_response'] / len(data) * 100):.1f}%");
    print(f"direct_kg_rag_response acc:  {(accuracy_records['direct_kg_rag_response'] / len(data) * 100):.1f}%");

    print('##############################################')

    print('full acc: ', accuracy_records['plain'] / len(data) * 100, "%");
    print('cot acc: ', accuracy_records['cot'] / len(data) * 100, "%");
    print('cot_sc acc: ', accuracy_records['cot_sc'] / len(data) * 100, "%");
    print(f"ka acc excluding fails:  {(accuracy_records['ka'] / (len(data) - num_failed) * 100):.1f}%");
    try:
        print(f"ka acc when an edit is made: {(accuracy_records['ka_standalone'] / edited_count * 100):.1f}%")
    except Exception as e:
        print("ka acc: untested");
    print(f"direct_response acc:  {(accuracy_records['direct_response'] / len(data) * 100):.1f}%");
    print(f"direct_kg_rag_response acc:  {(accuracy_records['direct_kg_rag_response'] / len(data) * 100):.1f}%");

    print('##############################################')

    print('Edited {} out of {} instances'.format(edited_count, len(data)))
    # print('domains used: ', domains)

    print('##############################################')

    print('sc score between 0 to 0.1: ', len([x for x in sc_records if x < 0.1]))
    print('sc score between 0.1 to 0.2: ', len([x for x in sc_records if 0.1 <= x < 0.2]))
    print('sc score between 0.2 to 0.3: ', len([x for x in sc_records if 0.2 <= x < 0.3]))
    print('sc score between 0.3 to 0.4: ', len([x for x in sc_records if 0.3 <= x < 0.4]))
    print('sc score between 0.4 to 0.5: ', len([x for x in sc_records if 0.4 <= x < 0.5]))
    print('sc score between 0.5 to 0.6: ', len([x for x in sc_records if 0.5 <= x < 0.6]))
    print('sc score between 0.6 to 0.7: ', len([x for x in sc_records if 0.6 <= x < 0.7]))
    print('sc score between 0.7 to 0.8: ', len([x for x in sc_records if 0.7 <= x < 0.8]))
    print('sc score between 0.8 to 0.9: ', len([x for x in sc_records if 0.8 <= x < 0.9]))
    print('sc score between 0.9 to 1.0: ', len([x for x in sc_records if 0.9 <= x < 1.0]))
    print('sc score 1.0: ', len([x for x in sc_records if x == 1.0]))
    
    # json.dump(edited_out_false_ones, open('outputs/fever/edited_out_false_ones.json', 'w'))
    # print('len(edited_out_false_ones): ', len(edited_out_false_ones))