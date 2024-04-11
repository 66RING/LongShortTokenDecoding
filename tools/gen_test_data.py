import argparse
import json

def get_context(file_path):
    file = open(file_path, "r", encoding="utf-8")
    context = ""
    for i in file.readlines():
        context += "".join(i.strip())
    return context

def main(args):
    context = get_context(args.file_path)
    interval = args.max_context_length // args.scope
    context_size = [int(i * args.scaler) for i in range(interval, args.max_context_length + 1, interval)]

    output = open(args.output, "w", encoding="utf-8")

    # generate test data with different context size
    # seperate by line
    for size in context_size:
        output_template = {"context_size": size, "input": context[:size]}
        print(json.dumps(output_template, ensure_ascii=False), file=output)
        output.flush()
    output.close()

    print(f"Dataset size {len(context_size)}")
    print(f"max context len {args.max_context_length}")
    print(f"scope {args.scope}")
    print(f"scaler {args.scaler}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="file path.")
    parser.add_argument("--output", type=str, required=True, help="output file path.")
    parser.add_argument("--max_context_length", type=int, default=128 * 1024, help="max context length.")
    parser.add_argument("--scope", type=int, default=16, help="max context length.")
    parser.add_argument("--scaler", type=float, default=2, help="character number to token number scaler.")
    args = parser.parse_args()
    main(args)
