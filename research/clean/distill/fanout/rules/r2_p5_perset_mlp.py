"""P5 r2: CLIP MLP-only cut by the per-set greedy order (only MLP masked while measured). See r2_p5_perset."""
from rules import r2_p5_perset as Q

vit, r18 = Q.make("mlp")
