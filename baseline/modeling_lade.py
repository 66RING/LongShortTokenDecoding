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
        tokenizer = self.tokenizer

        # prefill
        prefill_generated_ids, past_key_values = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, do_sample=False, use_cache=True)
        prefill_generated_text = tokenizer.decode(prefill_generated_ids[0], skip_special_tokens=True)

        # generate new mask and input
        model_inputs = tokenizer(prefill_generated_text, return_tensors='pt').to(model.device)

        torch.cuda.synchronize()
        decode_time = time.time()

        generated_ids = prefill_generated_ids[:, -1].unsqueeze(1)
        if not generated_ids == tokenizer.eos_token_id:
            # do decoding
            generated_ids, _ = model.generate(input_ids=generated_ids, attention_mask=model_inputs.attention_mask, past_key_values=past_key_values, max_new_tokens=max_gen_len, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        decode_time = time.time() - decode_time
        acc = [1]

        return generated_ids, decode_time, acc

