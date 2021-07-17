import os
import re
import json
import tqdm
import torch
import logging
import argparse
import numpy as np

from overrides import overrides
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass


class CopaInstanceReader(InstanceReader):
    """
    Reads the COPA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['premise']
        if not context.endswith("."):
            context += "."

        question = {"cause": "The cause for it was that", "effect": "As a result,"}[fields['question']]
        label = fields.get('label', None)
        choices = [fields['choice1'], fields['choice2']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{context} {question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


class PiqaInstanceReader(InstanceReader):
    """
    Reads the PIQA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ""
        question = fields["goal"]
        label = fields.get('label', None)
        choices = [fields["sol1"], fields["sol2"]]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, and choices.
    """
    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
              "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
              "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
              "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
              "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
              "What will (.*) do next?": r"[SUBJ] then",
              "How would (.*) feel after?": r"[SUBJ] then",
              "How would you describe (.*)?": r"[SUBJ] is seen as",
              "What kind of person is (.*)?": r"[SUBJ] is seen as",
              "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
              "Why did (.*) do that?": r"Before, [SUBJ] wanted",
              "Why did (.*) do this?": r"Before, [SUBJ] wanted",
              "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
              "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What will happen to (.*)?": r"[SUBJ] then",
              "What will happen to (.*) next?": r"[SUBJ] then"
        }

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        label = ord(label) - 65
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
                break

        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")

        choices = [
            " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
                "?", "").replace("wanted to wanted to", "wanted to").replace(
                "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

        context_with_choices = [f"{context} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices


class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."

        label = fields['answer']
        choices = [fields['option1'], fields['option2']]
        label = int(label) - 1
        question = ''
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [context.replace("_", choice) for choice in choices]
        return context, question, label, choices, context_with_choices


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ''

        question = fields['question']['stem']
        label = ['A','B','C','D','E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['text'] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{context} {question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


class MCTACOInstanceReader(InstanceReader):
    """
    Reads the MCTaco dataset into a unified format with context, question, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        question = fields['question']
        choices = fields["choices"]
        label = fields.get("label", None)
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{context} {question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


INSTANCE_READERS = {"copa": CopaInstanceReader,
                    "socialiqa": SocialIQAInstanceReader,
                    "winogrande": WinograndeInstanceReader,
                    "piqa": PiqaInstanceReader,
                    "commonsenseqa":CommonsenseqaInstanceReader,
                    "mctaco":MCTACOInstanceReader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="openai-gpt", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")

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

    # Predict instances
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                context, question, label, choices, context_with_choices = \
                    instance_reader.fields_to_instance(fields)

                gold.append(label)

                # Tokenize and pad
                tokenized = tokenizer(context_with_choices, return_tensors="pt", padding=True)["input_ids"].to(device)
                prediction = int(np.argmin(get_lm_score(model, tokenized, tokenizer.pad_token_id)))
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")

    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = accuracy_score(gold, predictions)
        print(f"Accuracy: {accuracy:.3f}")


def get_lm_score(model, batch, pad_token_id):
    """
    Get the cross entropy loss of the texts in batch using the langage model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1)
        shift_labels[shift_labels == pad_token_id] = -100
        lm_logits = model(batch).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_choices, -1).mean(1).cpu().numpy()

    return loss


def init_model(model_name: str,
               device: torch.device):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    main()
