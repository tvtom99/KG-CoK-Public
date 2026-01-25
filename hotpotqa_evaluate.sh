#!/bin/bash

if [ $# == 0 ]; then
    # New version - only run on the specified file!
    echo "Please ensure you provide a file to be evaluated!"
    
    # # Loop through every file in the testing_outputs folder and evaluate the results for any files with proper naming structure

    # for file in testing_outputs/output_fever_1000_*.json; do
    #     # Extract the threshold amount
    #     tval=$(echo "$file" | grep -oP 't0\K\d')
    #     echo ""
    #     echo "File: $file -> t = $tval"
    #     echo ""
    #     if [ -z "$tval" ]; then
    #         echo "No threshold specified in filename. Skipping file."
    #     else
    #         # Create logfile
    #         logfile="${file%.json}.log"
            
    #         # Append the date to the log file in chase there is already information there
    #         now=$(date '+%Y-%m-%d $H:%M:%S')

    #         # Create logfile
    #         echo "APPEND DATE: $now | File: $file | tval=$tval" > "$logfile"

    #         # Run program and store output into the logfile
    #         python evaluate_kgcok.py --dataset fever --output "$file" --threshold "0.$tval" >> "$logfile" 2>&1

    #         cat "$logfile"
    #     fi  
    # done
else 
    if [ $# -eq 1 ]; then
        file=$1
        
        # Get the tval
        tval=$(echo "$file" | grep -oP 't0\K\d')
        
        # Create logfile
        
        logfile="${file%.json}.log"
        
        # Append the date to the log file in case there is already information there
        now=$(date '+%Y-%m-%d $H:%M:%S')

        # Create logfile
        echo "APPEND DATE: $now | File: $file | tval=$tval" > "$logfile"

        # Run program and store output into the logfile
        python evaluate_kgcok.py --dataset hotpotqa --output "$file" --threshold "0.$tval" >> "$logfile" 2>&1

        cat "$logfile"
    fi
fi


# OLD TESTS
# python evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_retest_1.json --threshold 0.7
# python evaluate_kgcok.py --dataset fever --output testing_outputs/output_fever_1000_k3_w3_d2_t07.json --threshold 0.7
# nohup bash evaluate.sh > log/fever_evaluate_log_1shot 2>&1 &