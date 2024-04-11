# Speculative decoding

> 投机推理不一定要小模型, 小输入也能推得快

## Roadmap

- [x] streaming llm
- [x] speculative + streaming
- [ ] key cache position hacking
    - maybe arrange is not the best
- [ ] batching support
    - [ ] naive batching
    - [ ] effective batching
- [x] naive sampler
    - [x] cache sampler
    - [x] logits sampler
    - [x] accepter
- [ ] advance sampler
    - [ ] cache sampler
    - [ ] logits sampler
    - [ ] accepter
- [ ] flash attn support
- [ ] cache manager design
- [ ] tree attention
    - [ ] tree flash attention
- [ ] target model cache manager?
- [ ] ppl test
    1. reproduce streaming llm
    2. test model performence

## bug killing

- [x] memory larger then no speculative
    - infer mode: `with torch.no_grad()` or `@torch.no_grad()`
- [x] benchmark bug
    - [x] decoding tps should lower
        - only happend in LLM, small model may not
    - [x] speculative total time should not be median

## tips

- how dose medusa do batching?
- how long the seqlen cause decoding performance down
    * note, it deppends

## NOTE

- Sink may cause repeat
- cliff point is not the balance point

