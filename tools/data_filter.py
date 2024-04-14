import argparse
import json
import sys

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from ana_context_len import Analyzer

def main(args):
    try:
        dataset = load_dataset(args.dataset, split="train")
    except:
        dataset = load_dataset('json', data_files=args.dataset, split=f"train")

    dataname = args.dataset.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    output = open(args.output, "w", encoding="utf-8")

    data_len = 0
    min_token_len = sys.maxsize
    max_token_len = 0
    # origin dataset analyzer
    base_ana = Analyzer()
    # new dataset analyzer 
    new_ana = Analyzer()
    for sample in tqdm(dataset):
        input_text = sample[args.feature]
        input_id = tokenizer(input_text).input_ids
        base_ana.visit(len(input_id)//1000)
        if len(input_id) < args.min_token_len:
            continue
        new_ana.visit(len(input_id)//1000)
        print(json.dumps(sample, ensure_ascii=False), file=output)
        min_token_len = min(min_token_len, len(input_id))
        max_token_len = max(max_token_len, len(input_id))
        data_len += 1

    # base_ana.draw(save_png=f"./{dataname}_origin.png")
    # new_ana.draw(save_png=f"./{dataname}_filted.png")

    base_ana.save_to_csv(f"./{dataname}_origin.csv")

    print(f"Dataset size {data_len}")
    print(f"min token len {min_token_len}")
    print(f"max token len {max_token_len}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--min_token_len", type=int, default=128 * 1024)
    args.add_argument("--model_name_or_path", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--feature", type=str, default="text")
    args.add_argument("--dataset", type=str, required=True)
    main(args.parse_args())

