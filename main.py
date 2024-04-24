from sys import prefix
import numpy as np
import torch
import argparse
import time
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    AutoConfig,
)
from cache_manager import SinkCache, LongShortTokenCache, DynamicCache
from tqdm import tqdm
from speculative_inference import Lstd
from viz_utils import draw_line_char

from configuration_llama import LlamaConfig

def main(args):
    from modeling_llama import LlamaForCausalLM
    model_name_or_path = args.model_name_or_path

    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    # NOTE: add pad_token to use padding
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE: not support auto model since model modified to support yarn
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except:
        config = LlamaConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            device_map="auto",
            # attn_implementation="eager", # use LlamaAttention to test
            attn_implementation="flash_attention_2", # eagle not support flash attention yet
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=True,
        )
        print("try use_safetensors")
    except:
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            device_map="auto",
            # attn_implementation="eager", # use LlamaAttention to test
            attn_implementation="flash_attention_2", # eagle not support flash attention yet
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=False,
        )
        print("not use_safetensors")

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

    tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokenized_prompt.input_ids
    input_ids = input_ids.to(model.device)

    attention_mask = tokenized_prompt.attention_mask
    attention_mask = attention_mask.to(model.device)

    batch_size, seq_len = input_ids.shape
    max_gen_len = 512
    # max_gen_len = 24
    max_sample = 4

    kv_cache_manager = SinkCache(
        start_size=args.start_size, recent_size=args.recent_size
    )
    # kv_cache_manager = DynamicCache(
    #     cache_unit_range=(8, 16),
    #     kick=3,
    #     unit=256,
    #     start_size=4,
    #     slow_up_unum=4,
    #     threshold=0.75,
    # )

    # kv_cache_manager = LongShortTokenCache(
    #     unit_list=[10, 40],
    #     gap=20,
    #     sink=4
    # )
    # kv_cache_manager = None
    print(f"input shape: {input_ids.shape}")
    print(f"max_gen_len: {max_gen_len}")
    print(f"max_sample: {max_sample}")
    print(f"cache_size: {args.start_size + args.recent_size}")
    print(f"kv_cache_manager: {kv_cache_manager}")

    model = Lstd(model, tokenizer=tokenizer, cache_manager=kv_cache_manager)

    total_time = time.time()
    generation_result = model.generate(input_ids, past_key_values, attention_mask=attention_mask, max_gen_len=max_gen_len, max_sample=max_sample, algo=6)
    torch.cuda.synchronize()
    total_time = time.time() - total_time

    past_key_values = generation_result.past_key_values
    generated_ids = generation_result.generated_ids
    decode_time = generation_result.decode_time

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
    print(" ".join(generated_text), flush=True)
    generated_len = generated_ids.shape[1]
    print(generated_len)

    # number of tokens in context / time for processing context * batch size
    # prefill_tokens_per_second = input_ids.shape[1] / prefill_time * batch_size
    # 1 second / median time per token in seconds * batch size
    decode_tokens_per_second = batch_size * generated_len / np.sum(decode_time)

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    print(f" ** Speed (Total): {total_time:.2f} second")
    # print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
    print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
    print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")

    # # draw decoding time graph
    # draw_line_char(decode_time, title=f"{name}_tps, total_time={total_time:.2f}", show=False, save_path="./decode_time.png", filter=filter)
    # # draw accuracy graph
    # draw_line_char(accuracy, title=f"{name}_acc, mean={np.mean(accuracy):.2f}", show=False, save_path="./accuracy.png", filter=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=4096)
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--infer_type", type=str)
    args = parser.parse_args()

    main(args)




