---
name: project-cale-business-ideas
description: "Cale money ideas (2026-10-16: LoRA retraining subscription, device-testing service) and the cloud-GPU-later idea (2026-10-12)"
metadata:
  type: project
---

## Monetization notes (Shonu, 2026-10-16, verbatim in spirit)
- **Capability: auto fine-tuning service.** Developer writes a recipe for their LoRA. The base model is already on every consumer's computer; when a consumer opens the developer's app or website, Cale puts the LoRA on the base. When the base updates, every LoRA must be retrained; Cale sells a subscription that retrains them automatically on each update.
- **Trust: managed device-testing service.** Test how an app feels across many real consumer laptops through one managed service. Can sit inside a developer's (or a coding model's) harness. Needs cloud resources.
- **Philosophy:** everything falls out of "how do we make laptop-scale AI reliable enough to build real applications on". More ideas will come.
- Why: research serves the product; nesting must leave every slice something for a LoRA to build on.
- How to apply: design nested models so one LoRA works at every size (elastic LoRA training; size-aware adapter and developer LoRA share one slot). Measure how badly LoRAs break after base updates, as pricing data. Per-app RCF folds are a feature here ([[project-research-history]] section 7).

## Cloud GPU (later; Shonu, 2026-10-12)
- For now test GPU work only on the HP (MX150 2 GB, Intel UHD 620). He wants to try renting a cloud NVIDIA GPU one day. He believes the HP's GPUs work but hit bugs (Vulkan: MX150 out of device memory, UHD 620 access violation on load, build b11558, no driver versions recorded, no CUDA build tried, partial offload not tried).
- How to apply: do not plan around a cloud GPU yet. Record exact driver versions and error text when re-testing. Mark strong-GPU results "not tested". Long research runs (whole-model, fixed references) may move to a rented GPU.

See [[project-cale-vision]].
