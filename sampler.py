import torch
from typing import List, Optional, Set, Tuple, Union

class CacheSampler:
    """
    Base, abstract class for all cache samplers. The actual data structure is specific to each subclass.
    """

    def sample(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Return:
            A tuple containing the sampled key and value states.
        '''
        raise NotImplementedError("Make sure to implement `sample` in a subclass.")

class LogitsSampler:
    """
    Base, abstract class for all logits samplers. The actual data structure is specific to each subclass.
    """

    def sample(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Parameters:
            logits
                assert logits.shape = (bs, vocab)
                latest token logits

        Return:
            Sampled logits.
        '''
        raise NotImplementedError("Make sure to implement `sample` in a subclass.")

class Accepter:
    """
    Base, abstract class for all prob accepter. The actual data structure is specific to each subclass.
    """

    def match(
        self,
        # TODO: strict shape to (bs, vocab) for now
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> bool:
        '''
        Parameters:
            target_probs (`torch.Tensor`):
                last target token model probs.
            draft_probs (`torch.Tensor`):
                laste draft token model probs.

        Return:
            True if accept, False if reject.
        '''
        raise NotImplementedError("Make sure to implement `match` in a subclass.")

# TODO: topk match one impl

# TODO: is norm logits needed?


class TopkToppLogitsSampler(LogitsSampler):
    def __init__(self, top_k: int = 0, top_p: float = 0.0):
        self.top_k = top_k
        self.top_p = top_p

    def sample(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.top_k_top_p_filter(logits, self.top_k, self.top_p)
        return logits

    # copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
    def top_k_top_p_filter(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
        """

        Args:
            logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
            top_k (int, optional): top_k. Defaults to 0.
            top_p (float, optional): top_p. Defaults to 0.0.

        Returns:
            torch.Tensor: a renormalized logits
        """
        if top_k > 0:
            filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
            logits[logits < filter[:, [-1]]] = float('-inf')
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            filter = cumulative_probs > top_p
            filter[..., 1:] = filter[..., :-1].clone()
            filter[..., 0] = 0
            indices_to_remove = filter.scatter(1, sorted_indices, filter)
            logits[indices_to_remove] = float('-inf')
        return logits

class TopkAccepter(Accepter):
    def __init__(self, top_k: int = 4):
        self.top_k = top_k

    def match(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        # TODO: return optional[id]
    ) -> bool:
        target_ids = torch.multinomial(target_probs, num_samples=self.top_k)
        draft_ids = torch.multinomial(draft_probs, num_samples=self.top_k)
        # if topk match any, pick the match one
        matches = torch.eq(target_ids.unsqueeze(1), draft_ids)
        matched_indices = torch.any(matches, dim=1)
        index = torch.nonzero(matched_indices)
        if index.numel() == 0:
            return False
        else:
            return True

class StrictAccepter(Accepter):
    def __init__(self):
        pass

    def match(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> bool:
        return torch.argmax(target_probs) == torch.argmax(draft_probs)




