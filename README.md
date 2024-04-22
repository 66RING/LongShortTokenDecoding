# LSTD

## Usage

```python
from speculative_inference import SPD

kv_cache_manager = SinkCache(start_size=args.start_size, recent_size=args.recent_size)
model = SPD(model, tokenizer=tokenizer, cache_manager=kv_cache_manager)
model.generate(input_ids=input_ids)
```


## NOTE

和TriForce撞车了, 很难受。他们真的做得又快又好，下次争取提高手速。

