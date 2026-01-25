@echo off
REM Run Python script evaluate.py with arguments

python evaluate.py ^
    --dataset fever ^
    --output testing/test_output_fever.json ^
    --threshold 0.7
