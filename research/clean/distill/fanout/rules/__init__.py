"""Calculation rules. Each module defines vit(T, f) -> vitgen dict and r18(A, f) -> ResNet-18 student module.
T = teacher dict (vitgen.from_open_clip), A = torchvision ResNet-18 (eval, CPU).
Rules may use weights and fanout.generic inputs only. No photo loader may be imported here."""
