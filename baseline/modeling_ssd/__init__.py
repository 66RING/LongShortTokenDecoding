# transformers==4.33.1 https://github.com/dilab-zju/self-speculative-decoding/issues/14

from .decoding import exact_self_speculative_generate, self_speculative_sample
from .searching import LayerSkippingSearching
from .modeling_llama import LlamaForCausalLM as SsdLlamaForCausalLM
from .modeling_ssd import Ssd

