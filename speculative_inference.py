import torch
from tqdm import tqdm
import torch
from typing import List, Optional, Set, Tuple, Union
from torch.nn import functional as F

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)


# TODO: API refector

@torch.no_grad()
def speculative_inferece(
    input_ids: torch.Tensor,
    past_key_values: torch.Tensor,
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    max_len: int,
    max_sample: int = 4,
# TODO: format return
# ) -> Union[Tuple, CausalLMOutputWithPast]:
) -> torch.Tensor:
    '''
    speculative inference
    genenrate next N token without loss return

    input_ids: (bs, seqlen), input tokens



    1. decode: draft_model gen max_sample token
    2. verify:
        target_model take the output of draft_model as input
        verify each token
    '''

    bsz, input_seqlen = input_ids.shape

    # TODO: wrap as an easy to use kv cache interface

    draft_next_ids = input_ids
    target_next_ids = input_ids
    generated_ids = torch.empty((bsz, 0), device=input_ids.device, dtype=input_ids.dtype)

    while generated_ids.shape[1] < max_len:
        stable_len = generated_ids.shape[1]

        # create empty draft prob
        draft_generated_prob = torch.empty((bsz, 0, draft_model.config.vocab_size), device=input_ids.device, dtype=input_ids.dtype)
        draft_past_key_values = past_key_values

        # NOTE: draft gen max_sample next tokens
        for i in range(max_sample):
            draft_outputs = draft_model(
                input_ids=draft_next_ids,
                past_key_values=draft_past_key_values,
                use_cache=True,
            )

            draft_past_key_values = draft_outputs.past_key_values
            draft_token_prob = draft_outputs.logits[:, -1, :].unsqueeze(1)
            draft_generated_prob = torch.cat([draft_generated_prob, draft_token_prob], dim=1)
            draft_next_ids = draft_generated_prob[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = torch.cat([generated_ids, draft_next_ids], dim=1)

        # NOTE: target model parallel genenrate
        target_next_ids = torch.cat([target_next_ids, generated_ids[:, -max_sample:]], dim=1)

        target_outputs = target_model(
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
            # have chance to accept if target model do not trust draft model
            r = torch.rand(1, device = target_generated_ids.device)

            # TODO: multi batch support
            token_id = generated_ids[:, stable_len + i]

            # NOTE:
            # if target_pred_token_prob(x) > draft_pred_token_prob(x), have confidence to accept token x
            # if target_pred_token_prob(x) < draft_pred_token_prob(x), have change to accept token x

            # if r < torch.min(torch.tensor([1], device=r.device), target_generated_prob[:, i, token_id] / draft_generated_prob[:, i, token_id]):
            # TODO: strict limit some time reject event target and draft is the same model
            # if target_generated_prob[:, i, token_id] == draft_generated_prob[:, i, token_id]:
            # TODO: accept in more data science way
            if target_generated_prob[:, i, :].argmax(-1) == draft_generated_prob[:, i, :].argmax(-1):
            # if F.softmax(target_generated_prob, dim=-1)[:, i, token_id] == F.softmax(draft_generated_prob, dim=-1)[:, i, token_id]:
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
        past_key_values_trimmed = []
        assert past_key_values
        end_pos = input_seqlen + generated_ids.shape[1]
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

    return generated_ids


