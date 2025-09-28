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

## Markers

https://arxiv.org/pdf/2304.06712 some earliest work of adding circle to VLM for visual prompting

https://arxiv.org/pdf/2407.15850 marker of person of interest on video

https://arxiv.org/pdf/2310.11441 Simply overlaying ids on image regions unleashes visual grounding
and corrects answers

https://arxiv.org/pdf/2312.00784 arbitrary visual prompts (we annotate each region with visual prompts sampled from the following 8 possibilities: rectangle, ellipse, point, triangle, mask, mask contour, arrow, and scribble created using B´ ezier curves) but has training

https://arxiv.org/abs/2404.16375 List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs

we propose a new learning paradigm: “list items one by one,” which asks the model to enumerate and describe all visual tags placed on the image following the alphanumeric order of tags.

evaluate our finetuned SoM models on seven MLLM benchmarks. We find that this new dataset, even in a relatively small size (10k-30k images with tags), significantly enhances visual reasoning capabilities and reduces hallucinations for MLLMs

https://arxiv.org/pdf/2404.06510 Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves? self-correction with feedback

## Pixel-level

There are also coordinate prompt methods, such as SCAFFOLD (Lei et al., 2024a) and AO-Planner (Chen et al., 2024a), which convert input images into coordinates using metrics, enhancing spatial understanding and reasoning abilities in MLLMs.

## Soft visual prompt


