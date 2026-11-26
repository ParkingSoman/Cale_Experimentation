"""P5 one-shot row (p5rank1): keep the top f of each set by the step-1 p5rank score (no re-measurement). Free from the p5rank cache."""
from rules import p5rank as P

vit, r18 = P.make(oneshot=True)
