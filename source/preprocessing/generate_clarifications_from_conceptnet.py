import os
import json
import tqdm
import spacy
import logging
import argparse
import itertools

from source.preprocessing.conceptnet_helper import build_conceptnet, load_conceptnet, shortest_paths, to_natural_language

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, mctaco, or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--answer_redundancy", default=3, type=int, required=False,
                        help="how many answers to generate from each question")
    parser.add_argument('--max_clarifications', default=20, type=int, help="how many clarifications to keep")
    parser.add_argument('--max_length', default=3, type=int, help="maximum path length in edges")
    parser.add_argument("--conceptnet_dir", default="~/resources/conceptnet", type=str, help="ConceptNet directory")

    args = parser.parse_args()
    logger.info(args)

    nlp = spacy.load('en_core_web_sm')

    num_lines = sum(1 for _ in open(args.dataset))

    logger.info("Initializing ConceptNet")
    conceptnet_dir = os.path.expanduser(args.conceptnet_dir)
    if not os.path.exists(os.path.join(conceptnet_dir, 'cooc.npz')):
        logger.info("ConceptNet not found, building it.")
        build_conceptnet(conceptnet_dir)

    conceptnet = load_conceptnet(conceptnet_dir)

    with open(args.dataset, "r") as f_in:
        with open(args.out_file, "w") as f_out:
            logger.info(f"Reading instances from lines in file at: {args.dataset}")
            for line in tqdm.tqdm(f_in, total=num_lines):
                fields = json.loads(line.strip())

                # Get pairs of concepts to query ConceptNet for their relationship
                if args.dataset_type == 'winogrande':
                    context = fields['sentence']
                    choices = [fields['option1'], fields['option2']]

                    # Texts: any pair of content words from the context
                    context_content_words = get_content_words(context, nlp)
                    queries = list(itertools.combinations(context_content_words, 2))

                elif args.dataset_type == 'commonsenseqa':
                    context = fields['question']['stem']
                    choices = [choice['text'] for choice in fields['question']['choices']]

                    # Texts: any pair of content word from the context + choice
                    queries = list(itertools.product(get_content_words(context, nlp), choices))

                elif args.dataset_type == 'socialiqa':
                    context = fields['context']
                    choices = [fields['answerA'], fields['answerB'], fields['answerC']]

                    # Texts: any pair of content word from the context + choice
                    choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
                    queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

                elif args.dataset_type == 'copa':
                    context = fields["premise"]
                    choices = [fields["choice1"], fields["choice2"]]

                    # Texts: any pair of content word from the context + choice
                    choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
                    queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

                elif args.dataset_type == 'mctaco':
                    context = fields["context"] + fields["question"]
                    choices = fields["choices"]

                    # Texts: any pair of content word from the context + choice
                    choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
                    queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

                elif args.dataset_type == 'piqa':
                    context = fields["goal"]
                    choices = [fields["sol1"], fields["sol2"]]

                    # Texts: any pair of content word from the context + choice
                    choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
                    queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

                else:
                    assert(False, "Dataset should be one of winogrande,commonsenseqa,copa,piqa,mctaco,socialiqa")

                curr_clarifications = generate_clarifications(
                    queries, conceptnet, answer_redundancy=args.answer_redundancy, max_length=args.max_length)

                fields['clarifications'] = curr_clarifications + [("None", "None")]

                f_out.write(json.dumps(fields) + '\n')
                f_out.flush()


def get_content_words(text, nlp):
    """
    Return all the adjectives, nouns and verbs in the text.
    """
    doc = nlp(text)
    content_words = [t.text for t in doc if t.pos_ in {"VERB", "NOUN", "ADJ"}]
    return list(set(map(str.lower, content_words)))


def generate_clarifications(word_pairs, conceptnet, answer_redundancy=3, max_length=3):
    """
    Find ConceptNet paths between each word pair.

    word_pairs: list of word pairs.
    conceptnet: an initialized `Resource` object.
    answer_redundancy: how many paths to keep.
    max_length: how many edges in a path.
    """
    results = {f'What is the relationship between "{w1}" and "{w2}"?':
                   shortest_paths(
                       conceptnet, w1, w2, max_length=max_length, exclude_relations={'relatedto', 'relatedto-1'})
               for w1, w2 in word_pairs}

    # Prune
    results = {question: answers for question, answers in results.items() if len(answers) > 0}
    results = [(question, answer, weight)
               for question, answers in results.items()
               for answer, weight in answers
               if len(answer) > 0]

    results = [(question, to_natural_language(answer))
               for question, answer, weight in sorted(results, key=lambda x: x[-1], reverse=True)[:answer_redundancy]]

    return results


if __name__ == '__main__':
    main()
