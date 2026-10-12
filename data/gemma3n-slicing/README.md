---
tags:
- matformer
---


This repository contains configurations to slice Gemma 3n E4B, which is enabled thanks to it being a MatFormer. 
The E4B model can be sliced into small models, trading off quality and latency/compute requirements.
We recommend exploring the [MatFormer Lab](TODO: add link) to getting started with slicing Gemma 3n E4B yourself.

For each configuration, we calculate the MMLU accuracy.
Although these are not the only configurations possible, they are optimal configurations
identified by calculating the accuracy of the pre-trained model 

To learn more about MatFormers, please review the  and generate your own submodels
with the [MatFormer Lab](TODO: add link).

![alt text](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/Artboard_1.original.png  "This chart show’s MMLU performance vs model size of Gemma 3n Mix-n-Match (pretrained) capability.")
This chart show’s MMLU performance vs model size of Gemma 3n Mix-n-Match (pretrained) capability.

Some additional resources:

* [Gemma 3n launch blog](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide)
* [MatFormer paper](https://huggingface.co/papers/2310.07707)
