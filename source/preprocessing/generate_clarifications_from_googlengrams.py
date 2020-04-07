import os
import nltk
import json
import tqdm
import spacy
import logging
import argparse
import itertools
import subprocess

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, mctaco, piqa, or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--answer_redundancy", default=3, type=int, required=False,
                        help="how many answers to generate from each question")
    parser.add_argument('--max_clarifications', default=20, type=int, help="how many clarifications to keep")
    parser.add_argument('--min_freq', default=10, type=int, help="minimum co-occurrence frequency to consider")

    args = parser.parse_args()
    logger.info(args)

    nlp = spacy.load('en_core_web_sm')
    num_lines = sum(1 for _ in open(args.dataset))
    ngrams = {}

    with open(args.dataset, "r") as f_in:
        with open(args.out_file, "w") as f_out:
            for line in tqdm.tqdm(f_in, total=num_lines):
                fields = json.loads(line.strip())
                pairs = get_pairs(fields, args.dataset_type, nlp)
                curr_clarifications = {}

                for w1, w2 in pairs:
                    question = f'What is the relationship between "{w1}" and "{w2}"?'
                    if (w1, w2) in ngrams:
                        curr_clarifications[question] = ngrams[(w1, w2)]
                    else:
                        curr_clarifications[question] = ngrams[(w1, w2)] = get_shared_phrases(w1, w2).items()

                curr_clarifications = [(question, ngram)
                                       for question, answers in curr_clarifications.items()
                                       for ngram, count in list(sorted(answers, key=lambda x: int(x[1]), reverse=True))[
                                                           :args.answer_redundancy]]

                fields['clarifications'] = curr_clarifications + [("None", "None")]
                f_out.write(json.dumps(fields) + '\n')
                f_out.flush()


def get_pairs(fields, dataset_type, nlp):
    """
    Get pairs of words from the context/question and choices.
    :param fields: the dictionary of an instance loaded from a JSON file.
    :param dataset_type: one of winogrande, commonsenseqa, socialiqa, piqa, copa, mctaco.
    :param nlp: Spacy.
    :return: list of word pairs.
    """
    if dataset_type == 'winogrande':
        context = fields['sentence']

        # Texts: any pair of content words from the context
        context_content_words = get_content_words(context, nlp)
        queries = list(itertools.combinations(context_content_words, 2))

    elif dataset_type == 'commonsenseqa':
        context = fields['question']['stem']
        choices = [choice['text'] for choice in fields['question']['choices']]

        # Texts: any pair of content word from the context + choice
        choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
        queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

    elif dataset_type == 'socialiqa':
        context = fields['context']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]

        # Texts: any pair of content word from the context + choice
        choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
        queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

    elif dataset_type == 'copa':
        context = fields["premise"]
        choices = [fields["choice1"], fields["choice2"]]

        # Texts: any pair of content word from the context + choice
        choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
        queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

    elif dataset_type == 'piqa':
        context = fields["goal"]
        choices = [fields["sol1"], fields["sol2"]]

        # Texts: any pair of content word from the context + choice
        choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
        queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

    elif dataset_type == 'mctaco':
        context = fields["context"] + fields["question"]
        choices = fields["choices"]

        # Texts: any pair of content word from the context + choice
        choice_context_words = [w for choice in choices for w in get_content_words(choice, nlp)]
        queries = list(itertools.product(get_content_words(context, nlp), choice_context_words))

    else:
        assert (False, "Dataset should be one of winogrande,commonsenseqa,copa,piqa,mctaco,socialiqa")

    return queries


def get_content_words(text, nlp):
    """
    Return all the adjectives, nouns and verbs in the text.
    """
    doc = nlp(text)
    content_words = [t.text for t in doc if t.pos_ in {"VERB", "NOUN", "ADJ"}]
    content_words = set(map(str.lower, content_words)).difference(STOPWORDS)
    return list(content_words)


def remove_non_ascii(text):
    """
    Remove non ascii characters
    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def get_shared_phrases(term1, term2, google_ngram_dir='~/corpora/google_ngrams'):
    """
    Return phrases starting with term1 and including term2 and vice versa.
    """
    if term1 == term2:
        return {}

    google_ngram_dir = os.path.expanduser(google_ngram_dir)
    term1, term2 = term1.lower(), term2.lower()
    shared_phrases = {}
    gngram_dir = os.path.expanduser(google_ngram_dir)

    for term, other_term in itertools.permutations([term1, term2]):

        # The Google ngrams file is tab separated, containing: ngram and count.
        # Grep the relevant lines first to reduce the search space.
        ngram_files = [os.path.join(gngram_dir, f'googlebooks-eng-all-{n}gram-20120701-{term[:2]}_highfreq')
                       for n in range(2, 5)]

        lines = [item.split("\t")
                 for ngram_file in ngram_files
                 for item in grep_terms(term, other_term, ngram_file)
                 if os.path.exists(ngram_file)]

        shared_phrases = dict([item for item in lines if len(item) == 2])

    return shared_phrases


def grep_terms(term1, term2, filename):
    """
    Get lines from the current file that contain both term1 and term2
    """
    return subprocess.Popen(f'grep -E "{term1} ((.*\s)*){term2}" {filename}',
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True).communicate()[0].decode().split("\n")


if __name__ == '__main__':
    main()
