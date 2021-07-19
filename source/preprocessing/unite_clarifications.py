import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None, type=str, required=True, help="Dataset directory such as 'data/winogrande'")

    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.replace("copa", "COPA")
    
    for s in ["dev", "test"]:
        all_files = [os.path.join(args.dataset_dir, f)
                     for f in os.listdir(args.dataset_dir) 
                     if f.startswith(f"{s}_clarified") and f != f"{s}_clarified_all.jsonl"]
                     
        all_examples = [[json.loads(line) for line in open(f)] for f in all_files]
        
        with open(f"{args.dataset_dir}/{s}_clarified_all.jsonl", "w") as f_out:
            for ex_idx in range(len(all_examples[0])):
                curr_example = [ex[ex_idx] for ex in all_examples]
                ex = curr_example[0]
                ex["clarifications"] = [clar for i in range(len(curr_example)) 
                                        for clar in curr_example[i]["clarifications"] 
                                        if clar != ['None', 'None']] + [['None', 'None']]

                f_out.write(json.dumps(ex) + "\n")


if __name__ == '__main__':
    main()
