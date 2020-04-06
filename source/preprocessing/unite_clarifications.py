import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None, type=str, required=True, help="dataset directory")
    args = parser.parse_args()

    knowledge_sources = ["comet", "conceptnet", "googlengrams", "distilgpt2", "gpt2", "gpt2-medium", "gpt2-large",
                         "gpt2-xl", "openai-gpt", "xlnet-base-cased", "xlnet-large-cased"]

    for s in ["dev", "train", "test"]:
        f_ks = [[line for line in open(f"{args.dataset_dir}/{s}_clarified_{knowledge_source}.jsonl")] for
                knowledge_source in knowledge_sources
                if os.path.exists(f"{args.dataset_dir}/{s}_clarified_{knowledge_source}.jsonl")]

        with open(f"{args.dataset_dir}/{s}_clarified_all.jsonl", "w") as f_out:
            for lines in zip(*f_ks):
                ex = json.loads(lines[0].strip())
                clarifications = set(
                    [tuple(c) for line in lines[1:] for c in json.loads(line.strip())["clarifications"]]).union(
                    set([tuple(cl) for cl in ex["clarifications"]]))

                ex["clarifications"] = list(clarifications)
                f_out.write(json.dumps(ex) + "\n")


if __name__ == '__main__':
    main()
