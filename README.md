# Speculative decoding

> 投机推理不应定要小模型, 小输入也能推得快

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

## bug killing

- [x] memory larger then no speculative
    - infer mode: `with torch.no_grad()` or `@torch.no_grad()`

## tips

- how dose medusa do batching?

## NOTE

- Sink may cause repeat
