@echo off
REM Run Python script run.py with arguments

python run.py ^
    --model gpt-3.5-turbo-0613 ^
    --dataset fever ^
    --output testing_outputs/test_output_fever.json ^
    --step True ^
    --num_train 1 ^
    --num_test 3 ^
    --threshold 1
