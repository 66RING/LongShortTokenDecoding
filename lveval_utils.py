LVEVAL_DATASET_NAMES = [
    "factrecall_en_128k",
    "hotpotwikiqa_mixup_128k",
    "loogle_CR_mixup_128k",
    "loogle_MIR_mixup_128k",
    "loogle_SD_mixup_128k",
    "multifieldqa_en_mixup_128k",
    "factrecall_en_256k",
    "hotpotwikiqa_mixup_256k",
    "loogle_CR_mixup_256k",
    "loogle_MIR_mixup_256k",
    "loogle_SD_mixup_256k",
    "multifieldqa_en_mixup_256k",
    "loogle_MIR_mixup",
    "hotpotwikiqa_mixup",
    "loogle_CR_mixup",
    "multifieldqa_en_mixup",
    "loogle_SD_mixup",
    "factrecall_en",
]

LVEVAL_DATASET_PROMPT = {
    "hotpotwikiqa_mixup": "Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",    
    "loogle_SD_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_CR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_MIR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh_mixup": "请阅读以下文章并用中文回答问题，问题和答案只与其中一篇文章有关。只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n文章：{context}\n\n请基于上面的文章回答下面的问题，问题和答案只与其中一篇文章有关。只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n问题：{input}\n回答：",
    "factrecall_en": "Please answer the following questions based on the given article.\n\nArticle: {context}\n\nPlease answer the following questions based on the above article.\n\nQuestion: {input}\nAnswer:",
    "factrecall_zh": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "cmrc_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
    "lic_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n请现在基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
    "dureader_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
}

def truncate_prompt(tokenizer, prompt, max_length):
    # following LongBench, we truncate middle content to fit max_length
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return prompt



