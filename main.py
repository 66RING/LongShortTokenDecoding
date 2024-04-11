import numpy as np
import torch
import argparse
import time
from transformers import (
    LlamaTokenizer,
)
from cache_manager import SinkCache
from tqdm import tqdm
from speculative_inference import SPD

from modeling_llama import LlamaForCausalLM
from configuration_llama import LlamaConfig

from viz_utils import draw_line_char

def main(args):
    model_name_or_path = args.model_name_or_path
    name = model_name_or_path.split("/")[-1]

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

    past_key_values = None
    prompt = [
        "One day, Lily met a Shoggoth.",
        # "Once upon a time, there was a dragon.",
        # "Tell a story begin with: One day, Lily met a Shoggoth.",
        # "Once upon a time, there was a dragon.",
        # "What is the capital of China?",
        # "What is the capital of United States?",
        # "Who is Alen Turing?",
        # "Please tell me a joke.",
        # "What is the meaning of life?",
        # "Please recommend me a some movies.",
    ]

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.to(model.device)

    batch_size, seq_len = input_ids.shape
    # max_gen_len = 1024 * 32
    max_gen_len = 1024 * 8
    max_sample = 32

    # TODO: performance cliff at about 1k in this case
    args.recent_size = 1024
    # fileter out out of distribution data
    filter = False
    kv_cache_manager = SinkCache(
        start_size=args.start_size, recent_size=args.recent_size
    )
    # kv_cache_manager = None
    print(f"input shape: {input_ids.shape}")
    print(f"max_gen_len: {max_gen_len}")
    print(f"max_sample: {max_sample}")
    print(f"cache_size: {args.start_size + args.recent_size}")
    print(f"kv_cache_manager: {kv_cache_manager}")

    model = SPD(model, cache_manager=kv_cache_manager)
    total_time = time.time()
    generated_ids, prefill_time, decode_time, accuracy = model.generate(input_ids, past_key_values, max_gen_len=max_gen_len, max_sample=max_sample)
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
    prefill_tokens_per_second = input_ids.shape[1] / prefill_time * batch_size
    # 1 second / median time per token in seconds * batch size
    decode_tokens_per_second = batch_size * max_gen_len / np.sum(decode_time)

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    print(f" ** Speed (Total): {total_time:.2f} second")
    print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
    print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
    print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")

    # draw decoding time graph
    draw_line_char(decode_time, title=f"{name}_tps, total_time={total_time:.2f}", show=False, save_path="./decode_time.png", filter=filter)
    # draw accuracy graph
    draw_line_char(accuracy, title=f"{name}_acc, mean={np.mean(accuracy):.2f}", show=False, save_path="./accuracy.png", filter=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--enable_streaming", action="store_true")
    args = parser.parse_args()

    main(args)




