import torch
import numpy as np

def slice1d(x, start, end):
    return x[:, start:end, ...]

def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class SinkCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        '''
        past_key_values = [layer_num,..][k, v](batch, head, seq, hidden_dim)
        '''

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

class LongShortTokenCache:
    # TODO: gap list
    def __init__(self, unit_list: List, gap: int = 4, sink: int = 4, k_seq_dim= 2, v_seq_dim=2):
        '''
        Parameters:
            unit_list: List
                List of the number of tokens in each unit.
            gap: int
                The gap between the long-term and short-term cache.
            sink: int
                The number of tokens to sink.
        '''
        self.cache_size = np.sum(unit_list) + sink
        self.unit_list = unit_list
        self.gap = gap
        self.sink = sink

        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values, gap: Optional[int] = None, sink: Optional[int] = None):
        '''
        Parameters:
            past_key_values: List
                List of the past key and value.
            gap: int
                The gap between the long-term and short-term cache.
                Dynamically change the gap.
            sink: int
                The number of tokens to sink.
                Dynamically change the sink.
        '''

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        new_past_key_values = []

        if gap is None:
            gap = self.gap
        if sink is None:
            sink = self.sink

        # iter over each layer kvcache
        for k, v in past_key_values:
            cache_remain = self.cache_size
            seq_end = seq_len

            bs, head, _, headdim = k.shape
            layer_k = torch.empty((bs, head, 0, headdim), device=k.device, dtype=k.dtype)
            layer_v = torch.empty((bs, head, 0, headdim), device=v.device, dtype=v.dtype)

            # save space for attention sink
            cache_remain -= sink

            # init long-short token cache
            for unit_size in reversed(self.unit_list):
                if cache_remain > seq_end - sink:
                    layer_k = torch.cat(
                        [
                            self.k_slice(k, sink, seq_end),
                            layer_k,
                        ],
                        dim=self.k_seq_dim,
                    )
                    layer_v = torch.cat(
                        [
                            self.v_slice(v, sink, seq_end),
                            layer_v,
                        ],
                        dim=self.v_seq_dim,
                    )
                    break
                else:
                    layer_k = torch.cat(
                        [
                            self.k_slice(k, seq_end - unit_size, seq_end),
                            layer_k,
                        ],
                        dim=self.k_seq_dim,
                    )
                    layer_v = torch.cat(
                        [
                            self.v_slice(v, seq_end - unit_size, seq_end),
                            layer_v,
                        ],
                        dim=self.v_seq_dim,
                    )

                    seq_end = seq_end - unit_size - gap
                    cache_remain -= unit_size

            # init sink cache
            layer_k = torch.cat(
                [
                    self.v_slice(k, 0, sink),
                    layer_k,
                ],
                dim=self.k_seq_dim,
            )
            layer_v = torch.cat(
                [
                    self.v_slice(v, 0, sink),
                    layer_v,
                ],
                dim=self.v_seq_dim,
            )

            new_past_key_values.append([layer_k, layer_v])
        return new_past_key_values





