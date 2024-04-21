try:
    from .modeling_cohere import CohereForCausalLM
except:
    CohereForCausalLM = None
    print("CohereForCausalLM only support transformers>=4.39.1")

# from .modeling_eagle import EAGLE
try:
    from .modeling_ssd import Ssd, SsdLlamaForCausalLM
except:
    SsdLlamaForCausalLM = None
    Ssd = None
    print("transformers version of CohereForCausalLM conflcit with Ssd")

from .modeling_lade import Lade
from .modeling_baseline import Base
