# LSTD

## Usage

```python
from speculative_inference import Lstd

kv_cache_manager = SinkCache(start_size=args.start_size, recent_size=args.recent_size)
model = Lstd(model, tokenizer=tokenizer, cache_manager=kv_cache_manager)
model.generate(input_ids=input_ids)
```


## NOTE

A concurrent work of [TriForce](https://github.com/Infini-AI-Lab/TriForce). TriForce is a great job with much much more experiment and much much more robust than us.

peace.

