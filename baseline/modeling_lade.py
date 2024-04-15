import torch
import time
from tqdm import tqdm
import os

class Lade:
    def __init__(self, model, tokenizer):
        self.model = model
        self.model.tokenizer = tokenizer
        self.device = model.device
        self.tokenizer = tokenizer

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def generate(
            self,
            input_ids,
            past_key_values,
            max_gen_len,
            attention_mask = None,
            **kwargs
        ):
        model = self.model

        torch.cuda.synchronize()
        decode_time = time.time()

        greedy_output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_gen_len, do_sample=False)[0]

        torch.cuda.synchronize()
        decode_time = time.time() - decode_time

        generated_ids = greedy_output[input_ids.numel():].unsqueeze(0)
        acc = [1]
        return generated_ids, decode_time, acc

