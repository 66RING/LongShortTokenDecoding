from dataclasses import dataclass, field

@dataclass
class GenerationResult:
    '''
    past_key_values: kv cache
    generated_ids: generated token id
    decode_time: decoding only time
    accuracy: speculative accuracy
    max_sample_list: dynamic sample metrix
    tp_list: throughput metrix
    start_token_list: input token of each speculative decoding
    '''
    past_key_values: list = field(default_factory=list)
    generated_ids: list = field(default_factory=list)
    decode_time: float = 0
    accuracy: float = 1
    max_sample_list: list = field(default_factory=list)
    tp_list: list = field(default_factory=list)
    start_token_list: list = field(default_factory=list)

