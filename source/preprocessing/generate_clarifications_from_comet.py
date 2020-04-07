import re
import tqdm
import json
import spacy
import textacy
import logging
import argparse

from comet2.comet_model import PretrainedCometModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


CATEGORY_TO_QUESTION = {"xIntent": "What was the intention of PersonX?",
                        "xNeed": "Before that, what did PersonX need?",
                        "oEffect": "What happens to others as a result?",
                        "oReact": "What do others feel as a result?",
                        "oWant": "What do others want as a result?",
                        "xEffect": "What happens to PersonX as a result?",
                        "xReact": "What does PersonX feel as a result?",
                        "xWant": "What does PersonX want as a result?",
                        "xAttr": "How is PersonX seen?"}

CATEGORY_TO_PREFIX = {"xIntent": "Because PersonX wanted",
                      "xNeed": "Before, PersonX needed",
                      "oEffect": "Others then",
                      "oReact": "As a result, others feel",
                      "oWant": "As a result, others want",
                      "xEffect": "PersonX then",
                      "xReact": "As a result, PersonX feels",
                      "xWant": "As a result, PersonX wants",
                      "xAttr": "PersonX is seen as"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, piqa, or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--device", type=str, required=False, default="cpu", help="cpu or GPU device")
    parser.add_argument("--model_file", type=str, required=False, help="The COMET pre-trained model", default=None)
    args = parser.parse_args()

    logger.info(f"Loading COMET model")

    # Load COMET either from args.model_file or from its default location.
    if args.model_file is not None:
        comet_model = PretrainedCometModel(model_name_or_path=args.model_file, device=args.device)
    else:
        comet_model = PretrainedCometModel(device=args.device)

    nlp = spacy.load('en_core_web_sm')
    get_clarifications = {"copa": get_clarifications_copa,
                          "winogrande": get_clarifications_winogrande,
                          "socialiqa": get_clarifications_socialiqa,
                          "commonsenseqa": get_clarifications_commonsenseqa,
                          "mctaco": get_clarifications_mctaco,
                          "piqa": get_clarifications_piqa}[args.dataset_type]

    with open(args.dataset) as f_in:
        with open(args.out_file, "w") as f_out:
            data_examples = [json.loads(line.strip()) for line in f_in]
            for ex in tqdm.tqdm(data_examples):
                ex["clarifications"] = get_clarifications(ex, nlp, comet_model)
                f_out.write(json.dumps(ex) + "\n")


def get_clarifications_piqa(ex, nlp, comet_model):
    """
    Generate clarifications for the PIQA dataset
    :param ex: a dictionary with the PIQA instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model
    :return: a list of (question, answer) tuples
    """
    # Questions are usually like "how would you do something?"
    personx = "you"

    input_event = ex["goal"].replace("?", "")
    outputs = {category: comet_model.predict(input_event, category, num_beams=5) for category in comet_model.categories}

    # We only care about preconditions and postconditions for X
    relevant_categories = ["xIntent", "xNeed", "xEffect", "xWant"]
    curr_events = []
    for category in relevant_categories:
        prefix = CATEGORY_TO_PREFIX[category]
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)

                question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append((question, out_event))

    return curr_events


def get_clarifications_commonsenseqa(ex, nlp, comet_model):
    """
    Generate clarifications for the CommonSenseQA dataset
    :param ex: a dictionary with the CommonSenseQA instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model
    :return: a list of (question, answer) tuples
    """
    CATEGORY_TO_QUESTION = {"xIntent": "What was their intention?",
                            "xNeed": "Before that, what did they need?",
                            "oEffect": "What happens to others as a result?",
                            "oReact": "What do others feel as a result?",
                            "oWant": "What do others want as a result?",
                            "xEffect": "What happens to them as a result?",
                            "xReact": "What do they feel as a result?",
                            "xWant": "What do they want as a result?",
                            "xAttr": "How are they seen?"}

    CATEGORY_TO_PREFIX = {"xIntent": "Because they wanted",
                          "xNeed": "Before, they needed",
                          "oEffect": "Others then",
                          "oReact": "As a result, others feel",
                          "oWant": "As a result, others want",
                          "xEffect": "They then",
                          "xReact": "As a result, they feel",
                          "xWant": "As a result, they want",
                          "xAttr": "They are seen seen as"}

    context = ex['question']['stem']
    personx,_ = get_personx(nlp, context, use_chunk=False)

    if len(personx) == 0:
        return []

    outputs = {category: comet_model.predict(context, category, num_beams=5) for category in comet_model.categories}

    curr_events = []
    for category, prefix in CATEGORY_TO_PREFIX.items():
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", '', out_event, flags=re.I)
                out_event = re.sub("person x", '', out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)

                question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append((question, out_event))
    return curr_events

                                                                     
def get_clarifications_mctaco(ex, nlp, comet_model):
    """
    Generate clarifications for the MCTACO dataset
    :param ex: a dictionary with the MCTACO instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model
    :return: a list of (question, answer) tuples
    """
    context = ex['context']
    personx, _ = get_personx(nlp, context)
    
    if len(personx) == 0:
        return []
    
    outputs = {category: comet_model.predict(context, category, num_beams=5) for category in comet_model.categories}

    curr_events = []
    for category, prefix in CATEGORY_TO_PREFIX.items():
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)

                question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append((question, out_event))

    return curr_events


def get_clarifications_socialiqa(ex, nlp, comet_model):
    """
    Generate clarifications for the SocialIQA dataset
    :param ex: a dictionary with the SocialIQA instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model objects
    :return: a list of (question, answer) tuples
    """
    context = ex['context']
    question = ex['question']

    question_to_comet_relation = {
          "What will [NAME] want to do next?": "xWant",
          "What will [NAME] want to do after?": "xWant",
          "How would [NAME] feel afterwards?": "xReact",
          "How would [NAME] feel as a result?": "xReact",
          "What will [NAME] do next?": "xReact",
          "How would [NAME] feel after?": "xReact",
          "How would you describe [NAME]?": "xAttr",
          "What kind of person is [NAME]?": "xAttr",
          "How would you describe [NAME] as a person?": "xAttr",
          "Why did [NAME] do that?": "xIntent",
          "Why did [NAME] do this?": "xIntent",
          "Why did [NAME] want to do this?": "xIntent",
          "What does [NAME] need to do beforehand?": "xNeed",
          "What does [NAME] need to do before?": "xNeed",
          "What does [NAME] need to do before this?": "xNeed",
          "What did [NAME] need to do before this?": "xNeed",
          "What will happen to [NAME]?": "xEffect",
          "What will happen to [NAME] next?": "xEffect"
    }

    clarifications = []
    personx, _ = get_personx(nlp, context)
    relation = question_to_comet_relation.get(re.sub(personx, "[NAME]", question, flags=re.I), None)

    if relation is not None:
        outputs = {relation: comet_model.predict(context, relation, num_beams=5)}

        prefix = CATEGORY_TO_PREFIX[relation]
        for out_event in outputs[relation]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)

                clarifications.append((question, out_event))

    return clarifications


def get_clarifications_winogrande(ex, nlp, comet_model):
    """
    Generate clarifications for the Winogrande dataset
    :param ex: a dictionary with the Winogrande instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model objects
    :return: a list of (question, answer) tuples
    """
    personx, persony = ex['option1'], ex['option2']

    # Only extract relations for people
    if personx[0] != personx[0].upper() or persony[0] != persony[0].upper():
        return []

    input_event = ex["sentence"]
    outputs = {category: comet_model.predict(input_event, category, num_beams=5) for category in comet_model.categories}

    curr_events = []
    for category, prefix in CATEGORY_TO_PREFIX.items():
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", persony, out_event, flags=re.I)
                out_event = re.sub("person y", persony, out_event, flags=re.I)

                question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append((question, out_event))

    return curr_events


def get_clarifications_copa(ex, nlp, comet_model):
    """
    Generate clarifications for the COPA dataset
    :param ex: a dictionary with the COPA instance
    :param nlp: Spacy NLP
    :param comet_model: the COMET model objects
    :return: a list of (question, answer) tuples
    """
    category_to_prefix_causes = {"xIntent": CATEGORY_TO_PREFIX["xIntent"],
                                 "xNeed": CATEGORY_TO_PREFIX["xNeed"]}

    category_to_prefix_effects = CATEGORY_TO_PREFIX.copy()
    category_to_prefix_effects.pop("xIntent")
    category_to_prefix_effects.pop("xNeed")
    category_to_prefix_effects.pop("xAttr")

    input_event = ex["premise"]
    personx, is_named_entity = get_personx(nlp, input_event)

    if personx == "":
        return []

    personx = personx if (is_named_entity or personx == "I") else personx.lower()
    outputs = {category: comet_model.predict(input_event, category, num_beams=5) for category in comet_model.categories}

    if ex["question"] == "cause":
        category_to_prefix = category_to_prefix_causes
    else:
        category_to_prefix = category_to_prefix_effects

    curr_events = []
    for category, prefix in category_to_prefix.items():
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = " ".join((prefix, out_event))

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)

                question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append((question, out_event))

    return curr_events


def get_personx(nlp, input_event, use_chunk=True):
    """
    Returns the subject of input_event
    """
    doc = nlp(input_event)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                is_named_entity = noun_chunks[0].root.pos_ == "PROP"
            else:
                logger.warning("Didn't find noun chunks either, skipping this sentence.")
                return "", False
        else:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return "", False
    else:
        subj_head = svos[0][0]
        is_named_entity = subj_head.root.pos_ == "PROP"
        personx = " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])

    return personx, is_named_entity


if __name__ == "__main__":
    main()
