import torch
import time
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Union
from torch.nn import functional as F
from sampler import TopkToppLogitsSampler, StrictAccepter
from cache_manager import DynamicCache, SinkCache, ShortCache



# TODO: rename
# Long short token decoding
class SPD:
    def __init__(self, model, tokenizer, cache_manager):
        self.draft_model = model
        self.target_model = model
        self.cache_manager = cache_manager
        self.device = model.device
        self.tokenizer = tokenizer

        # TODO: hard code sampler for now
        # NOTE: sampling may cause huge performance downgrade in prefill phase
        # self.logits_sampler = TopkToppLogitsSampler(top_k=20, top_p=0.9)
        self.logits_sampler = TopkToppLogitsSampler(top_k=0, top_p=0.0)
        self.accepter = StrictAccepter()

    def parameters(self):
        return self.target_model.parameters()

    # TODO: batching support
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: torch.Tensor,
        max_gen_len: int,
        max_sample: int = 4,
        attention_mask = None,
        **kwargs,
    # TODO: format return
    ):
        '''
        speculative inference
        genenrate next N token without loss return

        input_ids: (bs, seqlen), input tokens



        1. decode: draft_model gen max_sample token
        2. verify:
            target_model take the output of draft_model as input
            verify each token
        Return:
            generated_ids: (bs, seqlen), generated tokens
            prefill_time: float, prefill time
            decode_time: List[float], decode time for each token
            accuracy: List[float], accept rate for each iteration
        '''
        bsz, input_seqlen = input_ids.shape

        torch.cuda.synchronize()
        decode_time = time.time()
        accuracy = []

        # TODO: wrap as an easy to use kv cache interface

        # prefill phase and kv cache init
        if past_key_values is None:
            outputs = self.target_model(
                input_ids=input_ids,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # NOTE: sampling may cause huge performance downgrade in prefill phase
            # pred_token_idx = self.logits_sampler.sample(outputs.logits[:, -1, :]).argmax(dim=-1).unsqueeze(1)
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = pred_token_idx
            input_ids = pred_token_idx
        else:
            generated_ids = torch.empty((bsz, 0), device=input_ids.device, dtype=input_ids.dtype)

        draft_next_ids = input_ids
        target_next_ids = input_ids

        generated_len = 0
        pbar = tqdm(total=max_gen_len)
        cache_size = 0
        acc = 1.0
        while generated_len < max_gen_len:
            torch.cuda.synchronize()
            stable_len = generated_ids.shape[1]

            if isinstance(self.cache_manager, DynamicCache):
                self.cache_manager.step(acc)

            # create empty draft prob
            draft_generated_prob = torch.empty((bsz, 0, self.draft_model.config.vocab_size), device=input_ids.device, dtype=input_ids.dtype)
            if self.cache_manager is not None:
                draft_past_key_values = self.cache_manager(past_key_values)
            else:
                draft_past_key_values = past_key_values

            # NOTE: draft gen max_sample next tokens
            for i in range(max_sample):
                draft_outputs = self.draft_model(
                    input_ids=draft_next_ids,
                    past_key_values=draft_past_key_values,
                    use_cache=True,
                )

                draft_past_key_values = draft_outputs.past_key_values
                if self.cache_manager is not None and not isinstance(self.cache_manager, ShortCache):
                    draft_past_key_values = self.cache_manager(draft_past_key_values)
                else:
                    draft_past_key_values = draft_past_key_values
                cache_size = draft_past_key_values[0][0].shape[2]

                draft_token_prob = draft_outputs.logits[:, -1, :].unsqueeze(1)
                draft_generated_prob = torch.cat([draft_generated_prob, draft_token_prob], dim=1)
                draft_next_ids = self.logits_sampler.sample(draft_generated_prob[:, -1, :]).argmax(dim=-1).unsqueeze(1)
                # draft_next_ids = draft_generated_prob[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = torch.cat([generated_ids, draft_next_ids], dim=1)

            # NOTE: target model parallel genenrate
            target_next_ids = torch.cat([target_next_ids, generated_ids[:, -max_sample:]], dim=1)

            target_outputs = self.target_model(
                input_ids=target_next_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = target_outputs.past_key_values
            # target_pred_token_prob: (bs, seqlen, vocab)
            # NOTE: target model generate max_sample+1 tokens
            target_generated_prob = target_outputs.logits[:, -(max_sample+1):, :]
            # target_generated_ids: (bs, seqlen)
            target_generated_ids = target_generated_prob.argmax(dim=-1)

            # TODO: multi batch verify
            # NOTE: verify each token

            assert target_generated_prob.shape[1] == draft_generated_prob.shape[1] + 1
            accept_len = 0
            for i in range(max_sample):
                # TODO: should have chance to accept if target model do not trust draft model
                if self.accepter.match(target_generated_prob[:, i, :], draft_generated_prob[:, i, :]):
                    accept_len += 1
                else:
                    # reject
                    # TODO: impl as paper say. keep it simple for now
                    break

            generated_ids = generated_ids[:, :stable_len + accept_len]

            # take target model last valid token
            target_last_accept = target_generated_ids[:, accept_len].unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, target_last_accept], dim=1)

            #
            # update state for next iter
            #

            # rollback kvcache
            # past_key_values = [layer_num,..][k, v](batch, head, seq, hidden_dim)
            past_key_values_trimmed = []
            assert past_key_values
            # NOTE: -1 to drop prob token, since append target_last_accept to generated_ids
            end_pos = input_seqlen + generated_ids.shape[1] - 1
            # TODO: support **LLAMA** kvcache truncte only for now
            # k, v (batch, head, seq, hidden_dim)
            for kv in past_key_values:
                k, v = kv
                # NOTE: the indexing is specific for bloom. This won't work for other models
                # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)

                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)

            target_next_ids = generated_ids[:, -1].unsqueeze(1)
            draft_next_ids = generated_ids[:, -1].unsqueeze(1)
            past_key_values = past_key_values_trimmed
            generated_len = generated_ids.shape[1]

            acc = accept_len/max_sample if max_sample != 0 else 1
            accuracy.append(acc)
            pbar.set_postfix({"cache_size": cache_size, "acc": f"{acc:.2f}"})
            pbar.update(accept_len+1)

            if self.tokenizer.eos_token_id in generated_ids[-(max_sample+1):]:
                break

        torch.cuda.synchronize()
        decode_time = time.time() - decode_time

        return generated_ids, decode_time, accuracy


