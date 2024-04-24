# Speculative decoding

> 投机推理不一定要小模型, 小输入也能推得快

A concurrent work of [TriForce](https://github.com/Infini-AI-Lab/TriForce). TriForce is a great job with much much more experiment and much much more robust than us.

As for my first job in LLM serving I was in awe by the speed of deveploment and feeling depressed. So I tidy up my code and open source for more blade leanrer like me.

May chaos guide thee. May locality take the world.


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
- [x] flash attn support
- [x] yarn support
- [x] test data generation
- [ ] advance sampler
    - [ ] cache sampler
        - [x] Long short token cache sampler
    - [ ] logits sampler
    - [ ] accepter
- [ ] cache manager design
- [ ] tree attention
    - [ ] tree flash attention
- [ ] target model cache manager?

## tree attention

- tree forward
- top k generation
- batching

## bug killing

- [x] memory larger then no speculative
    - infer mode: `with torch.no_grad()` or `@torch.no_grad()`
- [x] benchmark bug
    - [x] decoding tps should lower
        - only happend in LLM, small model may not
    - [x] speculative total time should not be median
- [x] some cache and full cache generation not match
    1. wrong `end_pos`, since `generated_ids` have include the prob one. so need to -1
    2. `[:outrange]` may not crash in python
- [ ] draft and base not match in some time
    - since fp16/bf16 have some precision loss
- [ ] transformer version

## tips

- how dose medusa do batching?
- how long the seqlen cause decoding performance down
    * note, it deppends

## NOTE

- Sink may cause repeat
- cliff point is not the balance point
- pytorch memory leak like this: `output1 = model()`, `output2 = model()`

