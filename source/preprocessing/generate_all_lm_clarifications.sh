#!/usr/bin/env bash

dataset=$1
declare -a models=(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased)
declare -a sets=(dev test)
device=8

for model in "${models[@]}"
do
    for set in "${sets[@]}"
    do
        # Find available device
        while [ $device -gt 7 ]
        do
            for ((i=0;i<=7;i++));
            do
                info=`nvidia-smi -i ${i}`
                if [[ $info == *"No running processes found"* ]]; then
                    device=$i
                    echo "Using device ${device}"
                    break
                fi
            done

            if [[ $device -gt 7 ]]; then
                sleep 30s
            fi
        done

        curr_device=${device};
        device=8;
        python -m source.preprocessing.generate_clarifications_from_lm \
                --dataset data/${dataset}/${set}.jsonl \
                --out_file data/${dataset}/${set}_clarified_${model}.jsonl \
                --prefixes_file experiments/${dataset,,}/prefixes.json \
                --max_clarification_question_length 6 \
                --max_answer_length 10 --p_sampling_questions 0.2 \
                --p_sampling_answers 0.5 --question_redundancy 5 \
                --answer_redundancy 10 --device ${curr_device} \
                --lm ${model} --dataset_type ${dataset,,} &
        sleep 60s
    done
done
