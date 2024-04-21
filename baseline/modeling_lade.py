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
        if past_key_values is None:
            prefill_generated_ids, past_key_values = model.generate(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, max_new_tokens=1, do_sample=False, use_cache=True)
            prefill_generated_text = tokenizer.decode(prefill_generated_ids[0], skip_special_tokens=True)

            # generate new mask and input
            attention_mask = tokenizer(prefill_generated_text, return_tensors='pt').to(model.device).attention_mask
        else:
            # multi-dim input not work in lade
            for i in range(input_ids.shape[1]):
                next_token = input_ids[:, i].unsqueeze(1)
                prefill_generated_ids, past_key_values = model.generate(input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values, max_new_tokens=1, do_sample=False, use_cache=True)
                prefill_generated_text = tokenizer.decode(prefill_generated_ids[0], skip_special_tokens=True)

                attention_mask = torch.cat([attention_mask, torch.ones(1,1).to(model.device)], dim=1)


        # consider decoding time only since long-context llm may repeat words.
        torch.cuda.synchronize()
        decode_time = time.time()

        generated_ids = prefill_generated_ids[:, -1].unsqueeze(1)
        if not generated_ids == tokenizer.eos_token_id:
            # do decoding
            generated_ids, past_key_values = model.generate(input_ids=generated_ids, attention_mask=attention_mask, past_key_values=past_key_values, max_new_tokens=max_gen_len, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        decode_time = time.time() - decode_time
        acc = [1]

        return past_key_values, generated_ids, decode_time, acc, [], []

