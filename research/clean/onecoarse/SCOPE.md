# Scope of the one-coarse system (Shonu sets this file)

Shonu, 2026-11-23: "whenever you increase the scale of an experiment, I need you to have some sort of guard so that
you ask me first and you make it very explicit."

The guard `code/scale_gate.py` reads the block below. A script in `code/` runs only if its `# scale:` line names
items from this block alone. Only Shonu widens this file.

| Axis | Allowed now |
|---|---|
| Neurons | parent s3:69, coarse B128aug_s1 n101, the 11 R1 children of s3:69 in A. A's block-1 read weights and A's head, as weights only (no block-1 neuron read as a neuron) |
| Networks | A, B128aug_s1 |
| Photos | the saved summaries of the 19,000 held + test photos (no new forward pass) |
| Where | Mac, numpy (no torch, no cluster) |

Anything else needs this block, posted in chat, and Shonu's word:

```
SCALE-UP REQUEST
From: <current scope>
To:   <new scope: neurons / photos / network / Mac or cluster>
Why:  <the question it answers, one line>
Cost: <time, cluster or Mac>
```

Random-neuron controls and all-512 sweeps are scale-ups.

<!-- scale-allow
neurons: s3:69, B:n101, A:children, A:block1-weights, A:head-weights, s3:C1-groups
net: A, B128aug_s1
photos: saved
where: mac
-->
