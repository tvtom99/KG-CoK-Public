#!/bin/bash

dataset="fever"
model="deepseek-r1:32b"
numtests=1000
threshold=0.7

modelfilename=$(echo "$model" | tr '[:lower:]' '[:upper:]' | tr -d :)

echo "Current values of script..."
echo "DATASET: $dataset"
echo "MODEL: $model"
echo "NUM_TESTS: $numtests"

empty_filename="testing_outputs/output_${dataset}_${numtests}_${modelfilename}_empty.json"
empty_found=0

# Check if there is an empty file already existing!
if [ -e $empty_filename ]; then
    echo ""
    echo "Found empty base file for this test! Setting flag..."
    empty_found=1
else
    echo ""
    echo "Unable to find empty file: $empty_filename".
    echo "Will create empty file after first test!"
fi

for ((k=1; k<=3; k++)); do
    for ((width=1; width<=3; width++)); do
        for ((depth=1; depth<=3; depth++)); do
            # Print the current test
            echo "=================================================================================="
            echo "STARTING TEST - K: $k, width: $width, depth: $depth"

            # Create filename for this test
            this_filename=testing_outputs/output_${dataset}_${numtests}_${modelfilename}_k${k}_w${width}_d${depth}_t${threshold/./}.json

            # Check if the empty file exists (for increased testing speed)
            if [ $empty_found -eq 1 ]; then
                echo "Found empty, checking if output file for this test already exists..."
                if [ -e $this_filename ]; then  # Double check that this test hasn't already been run! We don't want to re-do the results.
                    echo "Output file found for values ... k: $k, width: $width, depth: $depth"
                    echo "Filename: $this_filename"
                    echo "Will continue test from this file."
                else    # The test hasn't been done before. Run the test from the beginning for this value of k, width and depth.
                    echo "File not found. Creating file from empty base."
                    cp $empty_filename $this_filename
                fi
            fi

            # Run the program and track the time it takes to execute!
            SECONDS=0
            python run.py --model $model --dataset $dataset --k $k --width $width --depth $depth --step True --num_test $numtests --threshold $threshold
            echo "Execution took $((SECONDS/60)) minutes and $((SECONDS%60)) seconds for k: $k, width: $width, depth: $depth." >> time.log

            # If empty didn't exist before the first test, create it using the python function.
            if [ $empty_found -eq 0 ]; then
                echo ""
                echo "E M P T Y   N O T   F O U N D"
                echo "  No empty version (unanswered) existed before this test was run." 
                echo "  To increase the speed of future testing, the initial test will be used as a base to conduct further testing via the creation of a cleaned version of the output with no CoK answers."
                echo "E M P T Y   C R E A T E D"
                python cleaner.py $this_filename $empty_filename
                echo ""
                empty_found=1
            fi
        done
    done
done


echo "TESTING COMPLETE!!!"


# export TRANSFORMERS_VERBOSITY=error
# export CUDA_VISIBLE_DEVICES=0
# python run.py \
#     --model gpt-3.5-turbo-0613 \
#     --dataset hotpotqa \
#     --output outputs/hotpotqa/hotpotqa_output.json \
#     --step True \
# python run.py --model gpt-3.5-turbo-0613 --dataset fever --output testing/test_output_fever.json --step True --num_train 1 --num_test 3 --threshold 1 --debug 1
# python run.py --model gpt-3.5-turbo-0613 --dataset fever --output testing_outputs/output_fever_1000_retest_1.json --debug 1 --step True --num_test 1000 --threshold 0.7 --retest 1
# python run.py --model gpt-3.5-turbo-0613 --dataset fever --output testing_outputs/output_fever_1000_GPT-OSS120_k2_w3_d2_t07.json --step True --num_test 1000 --threshold 0.7
# SECONDS=0
# python run.py --model gpt-oss:120b --dataset fever --k 1 --width 3 --depth 2 --step True --num_test 1000 --threshold 0.7 --retest 1
# python run.py --model deepseek-r1:70b --dataset fever --k 2 --width 3 --depth 2 --step True --num_test 1000 --threshold 0.7 --retest 1

# python run.py --model gpt-oss:120b --dataset fever --output testing_outputs/output_fever_1000_GPT-OSS120_k2_w3_d2_t07.json --step True --num_test 1000 --threshold 0.7