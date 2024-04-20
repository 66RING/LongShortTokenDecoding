import torch
import time
from tqdm import tqdm
from .decoding import self_speculative_sample
import json

with open('skip_layers.json', 'r') as f:
    skip_layers = json.load(f)

class Ssd:
    def __init__(self, model, tokenizer, model_type="ssd-7b"):
        self.model = model
        model.set_skip_layers(skip_layers[model_type]['attention'], skip_layers[model_type]['mlp'])
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

        generated_ids, decode_time, acc, _, _ = self_speculative_sample(model, self.tokenizer, input_ids, max_new_tokens=max_gen_len)

        return generated_ids, decode_time, acc, [], []


