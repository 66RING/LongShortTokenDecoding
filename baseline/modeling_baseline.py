import torch
import time
from tqdm import tqdm

class Base:
    def __init__(self, model, tokenizer):
        self.model = model
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
        batch_size = input_ids.size(0)
        torch.cuda.synchronize()
        decode_time = time.time()

        # prefill phase
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        # logits.shape: [bs, seq_len, vocab_size]
        # get the last token and predict the next token idx
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        # init generated_ids
        generated_ids = pred_token_idx

        for i in tqdm(range(max_gen_len - 1)):
            # decoding phase, generate next token with last token and kv cache
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = torch.cat([generated_ids, pred_token_idx], dim=1)

            if self.tokenizer.eos_token_id in pred_token_idx:
                break

        torch.cuda.synchronize()
        decode_time = time.time() - decode_time
        acc = [1]

        return generated_ids, decode_time, acc


