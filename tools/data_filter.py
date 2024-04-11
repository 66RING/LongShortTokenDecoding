import argparse
import json
import sys

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def main(args):
    try:
        dataset = load_dataset(args.dataset, split="train")
    except:
        dataset = load_dataset('json', data_files=args.dataset, split=f"train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    output = open(args.output, "w", encoding="utf-8")

    data_len = 0
    min_token_len = sys.maxsize
    max_token_len = 0
    for sample in tqdm(dataset):
        input_text = sample[args.feature]
        input_id = tokenizer(input_text).input_ids
        if len(input_id) < args.min_token_len:
            continue
        print(json.dumps(sample, ensure_ascii=False), file=output)
        min_token_len = min(min_token_len, len(input_id))
        max_token_len = max(max_token_len, len(input_id))
        data_len += 1

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

