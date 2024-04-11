import numpy as np
import torch
import argparse
import time
import json
import sys
import gc
from transformers import (
    LlamaTokenizer,
)
from cache_manager import SinkCache
from tqdm import tqdm
from speculative_inference import SPD

from modeling_llama import LlamaForCausalLM
from configuration_llama import LlamaConfig
from viz_utils import draw_line_char, write_csv_line

def test_data_iter(filename, feature):
    with open(filename, 'r', encoding='utf-8') as file:
        # read each line
        for line in file:
            data = json.loads(line)
            context = data.get(feature)
            if context:
                yield context


def main(args):
    model_name_or_path = args.model_name_or_path
    name = model_name_or_path.split("/")[-1]
    print(model_name_or_path)

    # NOTE: not support auto model since model modified
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    # NOTE: add pad_token to use padding
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        # attn_implementation="eager", # use LlamaAttention to test
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    max_gen_len = 128
    max_sample = 8
    name = f"{name}_s{max_sample}"
    kv_cache_manager = SinkCache(
        start_size=args.start_size, recent_size=args.recent_size
    )
    # kv_cache_manager = None
    print(f"max_gen_len: {max_gen_len}")
    print(f"max_sample: {max_sample}")
    print(f"cache_size: {args.start_size + args.recent_size}")
    print(f"kv_cache_manager: {kv_cache_manager}")

    model = SPD(model, cache_manager=kv_cache_manager)

    all_prefill_tps = []
    all_decoding_tps = []
    all_decoding_time = []
    all_acc = []
    all_mem_used = []
    x_data = []
    for input_text in test_data_iter(args.test_data, "input"):
        past_key_values = None
        input_tokens = tokenizer(input_text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=sys.maxsize,
            padding=False)

        input_tokens = input_tokens['input_ids'].to(model.device)
        print("input token size:", input_tokens.shape)
        batch_size, token_len = input_tokens.shape

        if token_len > args.max_token_len:
            break

        total_time = time.time()
        generated_ids, prefill_time, decode_time, accuracy = model.generate(input_tokens, past_key_values, max_gen_len=max_gen_len, max_sample=max_sample)
        torch.cuda.synchronize()
        total_time = time.time() - total_time

        generated_text = (
            tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        # TODO: print
        # print(" ".join(generated_text), flush=True)

        # number of tokens in context / time for processing context * batch size
        prefill_tokens_per_second = token_len / prefill_time * batch_size
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = batch_size * max_gen_len / np.sum(decode_time)

        device = next(model.parameters()).device
        memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        x_data.append(token_len)
        all_mem_used.append(memory_used)
        all_prefill_tps.append(prefill_tokens_per_second)
        all_decoding_tps.append(decode_tokens_per_second)
        all_decoding_time.append(np.sum(decode_time))
        all_acc.append(np.mean(accuracy))
        print(f"token_len: {token_len}, prefill_tps: {prefill_tokens_per_second:.2f}, decode_tps: {decode_tokens_per_second:.2f}, accuracy: {np.mean(accuracy):.2f}")

        generated_ids = None
        input_tokens = None
        gc.collect()
        torch.cuda.empty_cache()

    # draw decoding time graph
    draw_line_char(all_decoding_tps, x_data=x_data,title=f"{name}_tps, total_time={np.sum(all_decoding_time):.2f}", show=False, save_path="./tp_decode_time.png", filter=False)
    # draw accuracy graph
    draw_line_char(all_acc, x_data=x_data, title=f"{name}_acc, mean={np.mean(all_acc):.2f}", show=False, save_path="./tp_accuracy.png", filter=False)
    # draw memory used graph
    draw_line_char(all_mem_used, x_data=x_data, title=f"{name}_mem", show=False, save_path="./tp_mem_use.png", filter=False)

    # save raw data as csv
    with open("tp_data.csv", "w") as file:
        write_csv_line(file, "token_len", x_data)
        write_csv_line(file, "prefill_tps", all_prefill_tps)
        write_csv_line(file, "decode_tps", all_decoding_tps)
        write_csv_line(file, "decode_time", all_decoding_time)
        write_csv_line(file, "accuracy", all_acc)
        write_csv_line(file, "memory_used", all_mem_used)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=1024 * 4)
    parser.add_argument("--max_token_len", type=int, default=64 * 1024)
    args = parser.parse_args()

    main(args)





