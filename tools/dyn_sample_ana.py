import json
import argparse

def flatten(_2d_list):
    _1d_list = [item for sublist in _2d_list for item in sublist]
    return _1d_list

def write_csv_line(file, label, data, delim=','):
    print(f"{label}{delim}", file=file, end="")
    for x in data:
        print(f"{x}{delim}", file=file, end="")
    print(f"\n", file=file, end="")

# # demo input
# json_str = '''
# [
#   {
#     "max_sample": [
#       [8, 16],
#       [8, 16],
#       [8, 16]
#     ],
#     "start_token": [
#       [1576, 591],
#       [1559, 540],
#       [278, 29891]
#     ],
#     "start_token_char": [
#       ["The", "we"],
#       ["car", "he"],
#       ["the", "y"]
#     ]
#   },
#   {
#     "max_sample": [
#       [8, 16],
#       [8, 16],
#       [8, 16]
#     ],
#     "start_token": [
#       [1576, 591],
#       [1559, 540],
#       [278, 29891]
#     ],
#     "start_token_char": [
#       ["The", "we"],
#       ["car", "he"],
#       ["the", "y"]
#     ]
#   }
# ]
# '''

def main(args):
    with open(args.input, 'r') as file:
        data = json.load(file)

    delim = "`"
    with open(args.output, "w") as file:
        for item in data:
            # flatten and save
            max_sample_list = flatten(item["max_sample"])
            start_token_list = flatten(item["start_token"])
            start_token_char_list = flatten(item["start_token_char"])
            accuracy_list = flatten(item["accuracy"])
            tp_list = flatten(item["tp"])

            write_csv_line(file, "max_sample", max_sample_list, delim=delim)
            write_csv_line(file, "start_token", start_token_list, delim=delim)
            write_csv_line(file, "start_token_char", start_token_char_list, delim=delim)
            write_csv_line(file, "accuracy", accuracy_list, delim=delim)
            write_csv_line(file, "tp", tp_list, delim=delim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dyn_sample_result.csv")
    parser.add_argument("--input", type=str, default="input.json")
    args = parser.parse_args()

    main(args)

