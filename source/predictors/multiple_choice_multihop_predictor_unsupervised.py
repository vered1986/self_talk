import os
import json
import tqdm
import math
import torch
import random
import logging
import argparse
import itertools
import numpy as np

from sklearn.metrics import accuracy_score
from source.predictors.multiple_choice_asking_questions_predictor_unsupervised import \
    BATCH_SIZE, INSTANCE_READERS, get_lm_score, init_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(133)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="openai-gpt", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--max_clarifications", default=20, type=int, required=False, help="Max clarifications number")
    parser.add_argument("--n_hops", default=2, type=int, required=False, help="How many clarifications together")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device)

    # Load the dataset
    instance_reader = INSTANCE_READERS[os.path.basename(os.path.dirname(args.dataset_file)).lower()]()
    set_name = os.path.basename(args.dataset_file).replace(".jsonl", "")
    out_file = os.path.join(args.out_dir, f"{args.lm}_{set_name}_predictions.jsonl")
    gold = []
    predictions = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Predict instances
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                clarifications = fields["clarifications"]

                # Choose combination of n_hop clarifications and randomly sample up to max_clarifications
                clarifications = [(" ".join((q1, q2)), " ".join((ans1, ans2)))
                                  for (q1, ans1), (q2, ans2) in itertools.combinations(clarifications, args.n_hops)]

                if len(clarifications) >= args.max_clarifications:
                    clarifications = random.sample(clarifications, args.max_clarifications)

                fields["clarifications"] = clarifications

                context, question, label, choices, clarifications, context_with_choice_and_clarifications = \
                    instance_reader.fields_to_instance(fields)

                gold.append(label)

                # Tokenize and pad
                tokenized = [[tokenizer.encode(text) for text in per_clar]
                             for per_clar in context_with_choice_and_clarifications]
                max_length = [max([len(text) for text in per_clar]) for per_clar in tokenized]
                tokenized = [[text + [pad_token_id] * (max_len - len(text)) for text in per_clar]
                             for per_clar, max_len in zip(tokenized, max_length)]

                # Compute in batches
                num_choices = len(tokenized)
                num_batches = int(math.ceil(len(tokenized[0]) / BATCH_SIZE))
                per_choice_score = [1000] * num_choices

                for batch_index in range(0, num_batches):
                    curr_batch = [tokenized[i][batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE]
                                  for i in range(num_choices)]
                    curr_batch = [torch.tensor(per_clar).long().to(device) for per_clar in curr_batch]
                    curr_scores = [get_lm_score(model, clars_choice) for clars_choice in curr_batch]
                    per_choice_score = [min(per_choice_score[i], curr_scores[i]) for i in range(num_choices)]

                prediction = int(np.argmin(per_choice_score))
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")

        # Don't report accuracy if we don't have the labels
        if None not in gold:
            accuracy = accuracy_score(gold, predictions)
            print(f"Accuracy: {accuracy:.3f}")


if __name__ == '__main__':
    main()
