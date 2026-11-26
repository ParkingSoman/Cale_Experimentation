"""P5 r2: CLIP heads-only cut by the per-set greedy order (only heads masked while measured). See r2_p5_perset."""
from rules import r2_p5_perset as Q

vit, r18 = Q.make("heads")
