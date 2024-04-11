import numpy as np
import torch
import argparse
import time
from transformers import (
    AutoTokenizer,
)
from cache_manager import SinkCache
from tqdm import tqdm
from modeling_llama import LlamaForCausalLM
from speculative_inference import SPD

from viz_utils import draw_line_char

@torch.no_grad()
def batch_inference(model, tokenizer, input_ids, past_key_values, max_gen_len, kv_cache_manager, **kwargs):
    print("start")
    prefill_time = 0
    decode_time = []

    batch_size = input_ids.size(0)

    # prefill phase
    start = time.time()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    torch.cuda.synchronize()
    end = time.time()
    prefill_time = (end - start)

    past_key_values = outputs.past_key_values
    if kv_cache_manager is not None:
        past_key_values = kv_cache_manager(past_key_values)
    # logits.shape: [bs, seq_len, vocab_size]
    # get the last token and predict the next token idx
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    # init generated_ids
    generated_ids = pred_token_idx

    pbar = tqdm(total=max_gen_len - 1)
    cache_size = 0

    for i in range(max_gen_len - 1):
        start = time.time()
        # decoding phase, generate next token with last token and kv cache
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs,
        )
        past_key_values = outputs.past_key_values
        if kv_cache_manager is not None:
            past_key_values = kv_cache_manager(past_key_values)
        # NOTE: layer 0, key cache
        cache_size = past_key_values[0][0].shape[2]
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = torch.cat([generated_ids, pred_token_idx], dim=1)

        torch.cuda.synchronize()
        end = time.time()
        decode_time.append(end - start)

        pbar.set_postfix(cache_size=cache_size)
        pbar.update(1)



    # convert batch of generated_ids to text 
    generated_texts = []
    for i in range(batch_size):
        generated_text = tokenizer.decode(
            generated_ids[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip().split(" ")
        generated_texts.append(generated_text)

    for ib, text in enumerate(generated_texts):
        print(f"{ib} >")
        print(" ".join(text), flush=True)

    return past_key_values, prefill_time, decode_time

def main(args):
    model_name_or_path = args.model_name_or_path
    name = model_name_or_path.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    # NOTE: add pad_token to use padding
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        # not support flash for now
        # attn_implementation="flash_attention_2",
        attn_implementation="eager", # use LlamaAttention to test
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
    max_sample = 2

    # TODO: performance cliff at about 1k in this case
    args.recent_size = 102400
    filter = args.recent_size > 1024
    kv_cache_manager = SinkCache(
        start_size=args.start_size, recent_size=args.recent_size
    )
    # kv_cache_manager = None
    print("using kv_cache_manager: ", kv_cache_manager)

    model = SPD(model, cache_manager=kv_cache_manager)
    total_time = time.time()
    generated_ids, prefill_time, decode_time = model.generate(input_ids, past_key_values, max_gen_len=max_gen_len, max_sample=max_sample)
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

    print(" ".join(generated_text), flush=True)

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
    draw_line_char(decode_time, title=name, show=False, save_path="./decode_time.png", filter=filter)



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




