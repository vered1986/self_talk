import json
import tqdm
import torch
import string
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from typing import Optional, List, Dict
from transformers import BertTokenizer, BertForMaskedLM
from source.preprocessing.lm_text_generator import LMTextGenerator

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, or copa)")
    parser.add_argument("--prefixes_file", default=None, type=str, required=True,
                        help="Path to a json file with the dictionary of question and answer prefixes.")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--lm", default="gpt2", type=str, required=False, help="Which language model to use")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--max_clarification_question_length", default=8, type=int, required=False,
                        help="max question length in words")
    parser.add_argument("--max_answer_length", default=13, type=int, required=False, help="max answer length in words")
    parser.add_argument("--p_sampling_questions", default=0.0, type=float, required=False,
                        help="p for top_p for questions")
    parser.add_argument("--p_sampling_answers", default=0.0, type=float, required=False, help="p for top_p for answers")
    parser.add_argument("--k_sampling_questions", default=0, type=int, required=False, help="k for top_k for questions")
    parser.add_argument("--k_sampling_answers", default=0, type=int, required=False, help="k for top_k for answers")
    parser.add_argument("--question_redundancy", default=25, type=int, required=False,
                        help="how many questions to generate from each prefix")
    parser.add_argument("--answer_redundancy", default=3, type=int, required=False,
                        help="how many answers to generate from each question")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="BERT model to rank the clarifications")
    parser.add_argument('--max_clarifications', default=3, type=int, help="how many clarifications to keep")

    args = parser.parse_args()
    logger.info(args)

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    generator = LMTextGenerator(args.lm, device=device)

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model).to(device)

    # Prefixes of clarification questions and their corresponding answers.
    prefixes = json.load(open(args.prefixes_file, 'r'))

    num_lines = sum(1 for _ in open(args.dataset))

    with open(args.dataset, "r") as f_in:
        with open(args.out_file, "w") as f_out:
            logger.info(f"Reading instances from lines in file at: {args.dataset}")
            for line in tqdm.tqdm(f_in, total=num_lines):
                fields = json.loads(line.strip())

                if args.dataset_type == 'winogrande':
                    context = fields['sentence']
                    choices = [fields['option1'], fields['option2']]
                elif args.dataset_type == 'commonsenseqa':
                    context = fields['question']['stem']
                    choices = [c['text'] for c in fields['question']['choices']] 
                elif args.dataset_type == 'mctaco':
                    context = fields['context']
                    choices = fields['choices']
                elif args.dataset_type == 'socialiqa':
                    context = fields['context']
                    choices = [fields['answerA'], fields['answerB'], fields['answerC']]
                elif args.dataset_type == 'copa':
                    context = fields["premise"]
                    choices = [fields["choice1"], fields["choice2"]]
                elif args.dataset_type == 'piqa':
                    context = fields["goal"]
                    choices = [fields["sol1"], fields["sol2"]]
                else:
                    assert (False, "Dataset should be one of winogrande,commonsenseqa,copa,piqa,mctaco,socialiqa")

                if "_" in context:
                    substitute = get_best_pronoun(bert_model, bert_tokenizer, device, context)
                    context = context.replace("_", substitute)

                curr_clarifications = generate_clarifications(
                    generator, choices, args.max_clarification_question_length,
                    args.max_answer_length, context, prefixes, args.dataset_type,
                    question=fields["question"] if args.dataset_type in {"socialiqa", "mctaco"} else "",
                    p_sampling_questions=args.p_sampling_questions,
                    k_sampling_questions=args.k_sampling_questions,
                    p_sampling_answers=args.p_sampling_answers,
                    k_sampling_answers=args.k_sampling_answers,
                    question_redundancy=args.question_redundancy,
                    answer_redundancy=args.answer_redundancy)

                fields['clarifications'] = curr_clarifications + [('None', 'None')]

                f_out.write(json.dumps(fields) + '\n')
                f_out.flush()


def generate_clarifications(generator: LMTextGenerator,
                            choices: List[str],
                            max_clarification_question_length: int,
                            max_answer_length: int,
                            orig_question: str,
                            prefixes: Dict[str, str],
                            dataset_type: str,
                            question: str = "",
                            p_sampling_questions: Optional[float] = 0.0,
                            k_sampling_questions: Optional[int] = 0,
                            p_sampling_answers: Optional[float] = 0.0,
                            k_sampling_answers: Optional[int] = 0,
                            question_redundancy: int = 25,
                            answer_redundancy: int = 3):
    """
    Generate multiple clarification questions and answers them.

    instance: str, required
        The original instance.

    orig_question: str, required
        The part of the original instance to ask clarification questions about.

    prefixes: Dict[str, str], required
        Dictionary of question prefixes to answer prefixes.

    label: ``torch.LongTensor``
        The gold label.

    qa_redundancy: int, optional (default = 1)
        How many clarification questions and answers the model generates for each prefix.

    choices: List[Dict[str, str]], optional (default = None)
        The choices for multiple choice question answers.

    Returns:
        A list of (prefix, question, answer)
    """
    question_prefixes = [k for k in prefixes.keys() if not k.endswith("?")]

    # Generate the clarification questions
    clarification_questions = generator.generate([
        ". ".join((orig_question, question_prefix))
        for question_prefix in question_prefixes],
        length=max_clarification_question_length, stop_token='?',
        p=p_sampling_questions, k=k_sampling_questions, num_samples=question_redundancy)

    # Only keep clarification questions that end with a question mark and are longer than one word
    words = lambda s: set(s.translate(str.maketrans('', '', string.punctuation)).split())

    clarification_questions = [(' '.join((question_prefixes[i], clar_q)),
                                get_answer_prefix(prefixes, question_prefixes[i], clar_q.strip()))
                               for i, clar_qs in clarification_questions.items()
                               for clar_q in clar_qs
                               if len(words(clar_q).intersection(words(orig_question))) >= 1]

    clarification_questions = list(set(clarification_questions))
    curr_prefixes = {}

    # Don't ask that about names!
    for ch in choices:
        if (dataset_type == "winogrande" and ch[0] != ch[0].upper()) or dataset_type == "commonsenseqa":
            curr_prefixes[f'What is the definition of "{ch}"?'] = f'"{ch}" is defined as'

        elif dataset_type == "copa":
            curr_prefixes[f'What does it mean that {ch[0].lower()}{ch[1:].replace(".", "")}?'] = \
                f'{ch.replace(".", "")} means that'
            curr_prefixes[f'What happened after {ch[0].lower()}{ch[1:].replace(".", "")}?'] =\
                f'After {ch[0].lower()}{ch[1:].replace(".", "")},'
            curr_prefixes[f'What happened before {ch[0].lower()}{ch[1:].replace(".", "")}?'] = \
                f'Before {ch[0].lower()}{ch[1:].replace(".", "")},'

    if dataset_type == 'mctaco':
        curr_prefixes = {qp: ap for qp, ap in prefixes.items()}
        curr_prefixes[question] = ""
                          
    if dataset_type == "socialiqa":
        names = {w for w in words(orig_question) if w[0] == w[0].upper()}.intersection(
            {w for w in words(question) if w[0] == w[0].upper()})

        if len(names) > 0:
            spec_prefixes = [(qp.replace("[NAME]", name), ap.replace("[NAME]", name)) for qp, ap in prefixes.items()
                             if qp.endswith("?") for name in names]

            if len(spec_prefixes) > 0:
                most_relevant = list(
                    sorted(spec_prefixes,
                           key=lambda p: len(set(p[0].split()).intersection(set(question.split()))), reverse=True))[0]
                curr_prefixes[most_relevant[0]] = most_relevant[1]

    clarification_questions += [(qp, ap) for qp, ap in curr_prefixes.items()]

    # Generate the answers
    clar_questions_and_answers = generate_answers(
        orig_question, clarification_questions, generator, max_answer_length,
        answer_redundancy=answer_redundancy, p_sampling_answers=p_sampling_answers,
        k_sampling_answers=k_sampling_answers)

    return clar_questions_and_answers


def get_best_pronoun(bert_model, bert_tokenizer, device, orig_question):
    """
    Replaces the placeholder with the most likely pronoun
    """
    subs = ["he", "she", "it", "they", "her", "him", "them", "thing", "one", "someone", "ones", "things"]
    subs = sorted(
        zip(subs, get_substitute_probabilities(bert_model, bert_tokenizer, orig_question, subs, device)),
        key=lambda item: item[-1], reverse=True)
    substitute = subs[0][0]
    return substitute


def generate_answers(
        orig_question: str,
        clarification_questions: List,
        generator: LMTextGenerator,
        max_answer_length: int,
        answer_redundancy: int = 3,
        p_sampling_answers: Optional[float] = 0.0,
        k_sampling_answers: Optional[int] = 0):
    """
    Generate answers for the clarification questions.

    prefixes: Dict[str, str], required
        Dictionary of question prefixes to answer prefixes.

    answer_redundancy: int, optional (default = 1)
        How many answers to generate for each prefix.

    Returns:
        A list of (prefix, question, answer)
    """
    words = lambda s: set(s.translate(str.maketrans('', '', string.punctuation)).split())
    generation_prefixes = [" ".join((orig_question, answer_prefix)) for _, answer_prefix in clarification_questions]
    answers = generator.generate(generation_prefixes, length=max_answer_length,
                                 stop_token='.', p=p_sampling_answers, k=k_sampling_answers,
                                 num_samples=max(1, answer_redundancy))

    if len(answers) == 0:
        return []

    _, answers = zip(*sorted(answers.items(), key=lambda x: x[0]))

    # Only keep answers that end with a period
    capitalize = lambda s: s[0].upper() + s[1:]

    clar_questions_and_answers = [(clar_q, ' '.join([s for s in [capitalize(ans_prefix) if len(ans_prefix) > 0 else '',
                                                                 answer.replace("..", ".")] if s is not '']))
                                  for (clar_q, ans_prefix), curr_answers in zip(clarification_questions, answers)
                                  for answer in curr_answers
                                  if len(words(answer.lower()).intersection({"i", "i'm", "you", "your", "me"})) == 0]

    # Save one instance of every answer and add empty answer, i.e. not using any of the generated answers.
    clar_questions_and_answers = {ans: clar_q for clar_q, ans in clar_questions_and_answers}
    clar_questions_and_answers = list(set([(clar_q, ans) for ans, clar_q in clar_questions_and_answers.items()]))
    return clar_questions_and_answers


def get_answer_prefix(prefixes = None,
                      question_prefix: str = None,
                      question: str = None):
    """
    question_prefix, str, required
        The prefix of the clarification question

    question, str, required
        The rest of the clarification question after the prefix

    Returns the answer prefix for each question
    """
    question_prefix = question_prefix.replace("Question:", "").strip()
    answer_prefix = prefixes.get(question_prefix, "")
    answer_prefix = answer_prefix.replace("_", question.replace("?", ""))
    return answer_prefix


def get_substitute_probabilities(bert_model, bert_tokenizer, text, choices, device):
    choices_indices = [bert_tokenizer.convert_tokens_to_ids(
        bert_tokenizer.tokenize(choice))[0] for choice in choices]
    text = " ".join(("[CLS]", text.replace("_", "[MASK]"), "[SEP]"))

    # Tokenize input
    tokenized_text = bert_tokenizer.tokenize(text)
    masked_index = [i for i, token in enumerate(tokenized_text) if token == "[MASK]"][0]

    # Convert token to vocabulary indices
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).long().to(device)
    segments_tensors = torch.tensor([np.ones(len(indexed_tokens))]).long().to(device)

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    predictions = F.softmax(predictions[0, masked_index], dim=-1)

    # Compute the probability of the choices
    probs = [predictions[choice] for choice in choices_indices]

    return probs


if __name__ == '__main__':
    main()
