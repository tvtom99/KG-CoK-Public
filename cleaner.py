import json
import sys

from cok.utils import fever_parser

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input.json> (OPTIONAL: [output.json])");
    sys.exit(1);

input_file: str = sys.argv[1];

output_file: str = sys.argv[2] if len(sys.argv) > 2 else input_file;

# Keys that will be removed to create clean file
keys_to_remove = {
    "final_answer", # Removed to ensure testing can be done again 
    "fail", # Removed to ensure testing can be done again
    "fail_reason", # Removed to ensure file is clean
    "rationale_1_knowl", # Removed to ensure file is clean
    "edited_rationale_1", # Removed to ensure file is clean
    "new_rationale_2", # Removed to ensure file is clean
    "rationale_2_knowl", # Removed to ensure file is clean
    "edited_rationale_2" # Removed to ensure file is clean
}

# Load JSON as a dictionary
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f);

for obj in data:
    for key in keys_to_remove:
        obj.pop(key, None); # Use none so that no error is thrown if the key doesn't exist.


# Save cleaned JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False);