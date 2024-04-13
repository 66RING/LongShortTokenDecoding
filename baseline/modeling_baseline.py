import torch
import time
from tqdm import tqdm

class Base:
    def __init__(self, model):
        self.model = model
        self.device = model.device

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def generate(self, input_ids, past_key_values, max_gen_len, **kwargs):
        model = self.model
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
        # logits.shape: [bs, seq_len, vocab_size]
        # get the last token and predict the next token idx
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        # init generated_ids
        generated_ids = pred_token_idx

        for i in tqdm(range(max_gen_len - 1)):
            start = time.time()
            # decoding phase, generate next token with last token and kv cache
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = torch.cat([generated_ids, pred_token_idx], dim=1)

            torch.cuda.synchronize()
            end = time.time()
            decode_time.append(end - start)

        return generated_ids, prefill_time, decode_time, [1 for _ in decode_time]


