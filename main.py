import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from torch.nn import CrossEntropyLoss
import torch
import time
import numpy as np
import sys

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import List, Optional, Set, Tuple, Union
from accelerate.utils.modeling import set_module_tensor_to_device
from accelerate import init_empty_weights

from speculative_inference import speculative_inferece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="nickypro/tinyllama-110M")
    parser.add_argument("--draft_model", type=str, default=None)
    args = parser.parse_args()

    MAX_LENGTH = 128
    prompt = [
        "One day, Lily met a Shoggoth.",
        # "Tell a story begin with: One day, Lily met a Shoggoth.",
        # "Once upon a time, there was a dragon.",
        # "What is the capital of China?",
        # "What is the capital of United States?",
        # "Who is Alen Turing?",
        # "Please tell me a joke.",
        # "What is the meaning of life?",
        # "Please recommend me a some movies.",
    ]
    target_model_path = "/home/ring/Documents/workspace/modules/tinyllama-110M"
    draft_model_path = "/home/ring/Documents/workspace/modules/tinyllama-42m"
    draft_model_path = target_model_path

    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to("cuda")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
    input_ids = input_ids.to(target_model.device)

    print(f"input_ids.shape: {input_ids.shape}")
    past_key_values = None

    generated_ids = speculative_inferece(input_ids, past_key_values, target_model, draft_model, max_len=128)

    '''
    She was so excited to meet him! She asked him, "What's your name?"
    The Shoggoth replied, "My name is Shelly. I'm a very special Shelly."
    Lily asked, "What do you do?"
    Shelly said, "I like to eat yummy food. I eat lots of yummy food."
    Lily said, "That sounds fun! Can I try some?"
    Shelly said, "Sure! I have lots of yummy food. Come with me and I'll show you."
    So Lily followed Shelly to a nearby
    '''

    generated_texts = []
    for i in range(generated_ids.shape[0]):
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




if __name__ == "__main__":
    main()
