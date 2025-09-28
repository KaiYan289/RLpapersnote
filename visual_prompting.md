# Visual Prompting for VLM

https://arxiv.org/pdf/2409.15310 survey on visual prompting

## Bounding box

https://arxiv.org/pdf/2403.20271 curate dataset for markers on the image

https://arxiv.org/pdf/2306.15195 Shikra: let LLM output coordinates of objects; includes training

https://arxiv.org/pdf/2404.04514 VTprompt: add bounding box to ground the object and then answer; use a step-by-step prompt engineering method

https://arxiv.org/pdf/2406.07549 A3VLM: curation of dataset and training - first ground with coordinate, then answer question (uses 3D bounding boxes to locate actionable parts of an image)

https://arxiv.org/pdf/2405.03194 CityLLaVa: similarly, but scale up in city

https://arxiv.org/pdf/2404.09797 early bbox+rescaling work

https://arxiv.org/abs/2403.02325 CRG: mask the object, use probability difference before and after masking to select the right answer

https://arxiv.org/pdf/2404.13013 groma： user-defined bbox QA; use localized visual tokenization, feed some "box" into VLM. With region proposer & region encoder. Also training-based. ("enhancing the localization ability of MLLMs by directly integrating them into user instructions.")

https://arxiv.org/pdf/2310.05136 instructdet: encode user-specified regions (i.e., bounding boxes) into visual tokens, enhancing the localization ability of MLLMs by directly integrating them into user instructions.

https://arxiv.org/pdf/2407.04681 further enhances the localization capabilities of MLLMs by integrating contextual embeddings from external knowledge within bounding boxes, serving as visual prompts to boost the finegrained cognitive abilities of various MLLMs.
