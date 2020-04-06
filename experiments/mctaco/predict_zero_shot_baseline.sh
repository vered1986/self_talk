#!/bin/bash

device=$1
set=$2
dataset="mctaco"

declare -a lms=(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased)

echo -e "LM\tAccuracy" > output/${dataset}/baseline_zero_shot_results.tsv;


for lm in "${lms[@]}"
do
    python  -m source.predictors.multiple_choice_baseline_predictor_unsupervised \
            --dataset_file data/${dataset}/${set}.jsonl \
            --out_dir output/${dataset}/ \
            --lm ${lm} \
            --device ${device} > output/${dataset}/temp_baseline_zero_shot_results;

    # Read result and write to a file
    validation_acc=`grep "Accuracy: " output/${dataset}/temp_baseline_zero_shot_results | cut -d ':' -f 2`;
    echo -e "${lm}\t${validation_acc}" >> output/${dataset}/baseline_zero_shot_results.tsv;
done

