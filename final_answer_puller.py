



import argparse
import json


if __name__ == "__main__":
    # read and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="File to extract final answers from.")
    args = parser.parse_args()

    # Ensure an argument was passed    
    if args.filename is None:
        parser.print_help()
        exit(1)


    # Open the file
    file_name: str = args.filename;
    data: list[dict] = json.load(open(file_name, "r"));
    failed_answer_count: int = 0;
    incorrect_format_final_answer: int = 0;

    for i, answer in enumerate(data):
        print(f"##### {i} #####\n")
        
        found_answer: str = answer.get("final_answer", "")

        if found_answer != "":
            if "The answer is" in found_answer:
                print("FOUND ONE!");
                print(answer["final_answer"]);
                incorrect_format_final_answer += 1;
                answer["backup_final_answer"] = found_answer;
                # answer["final_answer"] = found_answer.split("The answer is")[-1];
                answer["final_answer"] = found_answer.split("The answer is")[-1].strip().split(".")[0];
        # If no found answer was discovered and there should have been one...
        elif found_answer == "" and answer.get("cot_sc_score", 0.0) < 0.8:
            print("No found answer present...");
            failed_answer_count += 1;


    print(f"\n After evaluating file, found {failed_answer_count} entries with no final answer when there should have been one!.")
    print(f"Found {incorrect_format_final_answer} final answers that were not formatted properly.");

    # Dump new file
    with open(file_name, "w") as fw:
        json.dump(data, fw);