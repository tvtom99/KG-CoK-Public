# echo "evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_retest.json --threshold 0.7"
# Loop through every file in the 
#!/bin/bash

for file in output_fever_1000_*.json; do
    # Extract the threshold amount
    tval=$(echo "$file" | grep -oP 't0\K\d')
    echo "File: $file -> t = $tval"
    # python evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_retest_1.json --threshold 0.7
# python evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_retest_1.json --threshold 0.7
# python evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_k3_w3_d2_t07.json --threshold 0.7
# nohup bash evaluate.sh > log/fever_evaluate_log_1shot 2>&1 &