#!/bin/bash

device=$1
set=$2
dataset="copa"

declare -a knowledge_sources=(comet conceptnet googlengrams distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased all)
declare -a lms=(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased)

echo -e "LM\tKnowledge Source\tAccuracy" > output/${dataset}/zero_shot_results.tsv;

python -m source.preprocessing.unite_clarifications --dataset_dir data/${dataset};

for knowledge_source in "${knowledge_sources[@]}"
do
    for lm in "${lms[@]}"
    do
        python  -m source.predictors.multiple_choice_asking_questions_predictor_unsupervised \
                --dataset_file data/COPA/${set}_clarified_${knowledge_source}.jsonl \
                --out_dir output/${dataset}/ \
                --lm ${lm} \
                --device ${device} > output/${dataset}/temp_zero_shot_results;

        # Read result and write to a file
        validation_acc=`grep "Accuracy: " output/${dataset}/temp_zero_shot_results | cut -d ':' -f 2`;
        echo -e "${lm}\t${knowledge_source}\t${validation_acc}" >> output/${dataset}/zero_shot_results.tsv;
    done
done
