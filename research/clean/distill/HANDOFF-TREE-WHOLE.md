# HANDOFF: build the tree of distinctions WHOLE, then look at it whole

Start here. Written 2026-11-26 late night at the end of a long session, at Shonu's request. Read this file, then `THEORY.md` section 1, then the essay passages named below. Analysis of what went wrong: `ANALYSIS-2026-11-26.md`.

## 0. Shonu's words (unedited, 2026-11-26 night)

Message 1 (start of the session):

> Okay, I think we made some nice progress last session, but to me, when I'm kind of evaluating what we actually did, it feels a lot like we just did a lot of very hyper-technical optimizations rather than thinking like how the concept itself is changing of the neuron tree. Like, if I'm gonna break this up, like, think of a node in a tree. Because I think what we want is ultimately a tree, and so we can think of a node as like properties of a node and other parts of a node. And so, or like laws of a node and properties of a node. And so, given a node, it probably has a parent, some children, and it has like the reading portion, writing portion, gate portion, firing portion, and there may be some laws that we expect too. Like, maybe we implicitly think every node has only one parent. Maybe we implicitly think all of these things. And so, I also want you to know that we've, I believe, finished the 95%, 50%, and 5% ResNet stuff, which you can use alongside your Clip KD stuff to do your experimentation. And I also think that we just need tighter feedback loops, like just really fast experiments because I don't think it should be taking too long to find the simple underlying concept that we want to get. Like, we can separate rounds into thinking of like very technical, particular things. And I honestly would say, like, those things are really only meant to fix a bug within an implementation. And whenever you want to change a particular implementation, and when I say bug, I mean we haven't overarching concept for how the neuron tree should look, and due to part of our implementation, we're not meeting that concept, and so that technical work is one part of it, and then the other part is like actually, you know, we have this neuron tree, but changing a law of the neuron tree at each node. And this overarching thing should be simple, right? It's just like, how are we thinking that really, it's like there's this neuron tree, but it's sort of open, like how the reading portion should work, the writing portion should work, and how the parents and children are figured out, mixed together, that whole thing. But I think, like, the concept of a tree we should still stick with just to have some sort of backbone. We should do faster experimentation and faster loops of understanding and changing the overarching concept versus going very deep on technicals. And let me know if the overarching concept has changed at all and if we should start from that point or just say, like, hey, this is the way it is. And I think we should also, I think we have a pretty good protocol for what we're doing with rounds. Again, like within rounds, although I did just cite a bunch of changes that you could make. But another thing I want you to think of is in between rounds, what are you doing? How are you applying things? And what are the mechanics that are going into a round? Like, I'd like you to look at last session to view some errors that we made, or in the session previous to that, or whatever whole projects, or just recent stuff, really. Because what happened was we did three to me, it felt like our understanding of the overarching concept didn't really change. So, we didn't think of a new theory that's kind of like pivoting off this neuron tree idea, which I believe the essay supports. Also, as part of the rounds, I think we need to really read the essay deeply. But I would say, like, from there, we just didn't make much progress on thinking of the overarching concept and checking that out and stuff. And I feel like our technical things are very particular. And honestly, they seem so particular that I'm not really confident that there's like a lot of generics that we've created from this. And so, those are some of the issues that I was thinking of that I'd want to address

Message 2 (on the stream, mid-plan):

> not sure if this helps but i think whatever we do with the stream size wise we shouldn't think of it as a separate error from the coarsening of neurons. when you coarsen the network, the network is physically unable to represent certain representatiosn in vector space, so we can coarsen the stream accordingly without any additional accuracy loss is the idea

Message 3 (after round L1):

> ok the point of this is now you go in a bunch of rounds and continue on. i feel likeyou are continuously stopping for no good reason. we have all the work done, it is incredibly obvious to you want we want to do, let's just do it and get it done! no excuses. this is completely possible, and i hope you do it!

Message 4 (after rounds L2-L4):

> Although, of course, we want to be testing our components and isolating what parts matter. One of the things, and I'm not sure if you're running into this issue, is that if we're testing things one time at a time, and you look, you can look at previous sessions for this, but oftentimes we might be throwing away things when in reality it's just because that is actually the right concept. But we didn't because the other parts are not looked at yet and changed to make them compatible, then it looks like the wrong thing that we're doing. And so, really, just like thinking, okay, here's my theory of a tree, and let me just check it out. Let's figure it out really quickly. Okay, then afterwards, I look at this tree holistically. I understand: okay, it fucked up. How does this actually compare to the trained version of this? And again, we have two different networks: we have the ResNet stuff and we have the ClipKD stuff. And then from there, it's like, okay, we can do two branches. Either it's the implementation of our idea and we need to edit the implementation and try the same thing, or it is the tree concept itself, where you know, there's a part of this tree concept in the general sense, and I really want to think about simplicity here. I think a lot of times, since you're really intelligent, you do a lot of things at once and then it turns more into AI slop, which is not what we're going for at all. We really just want elegant simplicity. That's really what's helped us throughout the sessions. And you can look at session evidence to see if I'm wrong on that or not. But really, just like, hey, we're testing a concept out of time. We check it out and then we say, Okay, should we edit this concept or edit the implementation? And then after that, we check it out and then we go iterate through these loops. And the whole time we're thinking of the whole system at once rather than, oh, you know, like let's go really, really deep on this read part. When I don't really think like this thing is going to be solved by innovating on the really, really like, and the key word is innovation. Like, I don't, I'm not sure if it's gonna be really solved on innovating on really, really tiny technical details. And you can let me know if that's wrong. But I really think it's just like, hey, here's the overall structure, which, and this is just the way that most advancements in ML are done. It's like surprisingly simple, but we just stumble across it. And so that's what we do. And that's the idea, and that's the hope. Right now, it just feels like we're again doing iffy. And I want you to kind of like do some sort of analysis, or maybe we should just do this in a different session, and you can pass this prompt off to a different session. But, like, just an analysis of you know, where are we going wrong? Because I just feel like we're drifting again and we're not attacking the concept like I want it to. Like, it just feels like you've ignored me, honestly, with the actual what I'm implying the emphasis really is here. And I just feel like you're just blowing past that because of your preconceived notions. I'm not even sure if you're looking at the essay anymore, honestly

Message 5 (on mixed-up tracks):

> Okay, now I'm realizing that you might be mixing up some of the things that I've said previously with the new track that we're doing, which is the neuron tree idea, which you proposed again. I think within the neuron tree idea, you proposed it mainly, and I really, really agreed with it and liked it. And you might be mixing up some very, very old ideas that I had when the hope was that this is a new tree and so it just might be confusing you. And again, this is like a consistent thing where I tell you a lot of things as the research work goes along. And then honestly, sometimes they contradict each other, and then you get really confused. And so I guess we both get frustrated with each other

## 1. The loop

```
 1. Theory     one tree, whole network, every part coarse at once (one page, essay passages)
 2. Build      the simplest code that does exactly that. No fits, no patches. Both networks
 3. Look       whole-network comparison with the trained networks, plus leave-one-out rows
 4. Decide     (a) the code does not do what the page says  -> fix the code, same theory
               (b) the code matches, the network differs    -> change ONE line of the page
 5. Repeat
```

| Rule | Detail |
|---|---|
| Whole, not parts | A part is never judged alone in a fine network. Parts are judged by leave-one-out rows INSIDE the whole build |
| One page | No mechanism enters unless it is a line of the theory page |
| Essay | Every loop starts with one essay passage, read in full (`../curves/map_agents/essay.md`) |
| Speed | One loop = one build on both networks + one look, about 30-45 minutes. Keep going loop after loop. Report the page diff, not knobs |
| Track | Only THEORY.md section 1 and Shonu's words since 2026-11-25 bind. Older-track notes (onecoarse morph, VECMATH, N1-N11) are history. A conflict = one line to Shonu, never a silent merge |
| Boundary | Weights + generic input only (pink, or `fanout/cache/*_self_n256_k32_s1.pt`). Trained students = answer key, never a build input. Photos = scoring only |

## 2. Theory v1: the tree of distinctions (THEORY.md section 1), built whole

```
big network, block b        car ──┬── taxi        (taxi, ambulance = siblings: one parent, R1)
                                  └── ambulance
coarse network, block b     car-branch: ONE node. Fires where taxi OR ambulance fires.
                            Writes the branch's message.
coarse network, block b+1   only readers of the car-branch. The taxi-vs-ambulance reader is gone too.
```

Essay passages: 225 (gate in front of the car detectors; "smaller models may stop at coarser distinctions, with something closer to a general car neuron"), 239 (the small model "ends early"), 251 ("one curves & loops neuron instead of ten curve detectors"), 469 (parts built to be read by downstream readers).

| Part | Theory v1 (every layer at once) | Default for a gap |
|---|---|---|
| Branch | siblings under one parent: nodes of a layer whose main parent (the upstream writer they read most, R1) is the same | CLIP: parents = upstream write columns, as in `fanout/rules/r2_p1p3_merge.py`. ResNet: inner nodes of the previous block |
| Coarse node | one node per branch (or per group of branches, to reach the size) | |
| Firing | the union (OR) of the branch's strongest firings, NOT a shared read direction (CALC 3.2-3.5) | read = the members' normalized pre-activations, combined by their maximum on the generic input |
| Gate | opens on the union of the branch | threshold = the union's firing rate on the generic input |
| Write | the branch's message, as the COARSE readers downstream read it | the sum of the members' writes, scaled by how often each fires inside the union |
| Every layer at once | a distinction stays only if its writer AND its reader stay | all layers coarsened in one build. No fine reader left anywhere |
| Order | rank each distinction by written x read strength. Merge the weakest first. One ranking gives 95, 50 and 5% | |
| Stream | follows from what the coarse nodes write and read (message 2) | full width in the first build, then the width the coarse writes use |
| Heads | heads merge as branches of heads | same rules, on whole heads |

The round-2 OR-merge (`r2_p1p3_merge.py`) was the closest earlier build: it merged only MLP neurons, heads and stream stayed fine, and its write was a ridge fit. Here every part is coarse together, and the write is the branch's own message.

## 3. Checks and the look

| Check | What | Pass |
|---|---|---|
| Identity | f = 1, both networks | KL < 1e-5 |
| Planted | in a copy of A, split some nodes into exact sibling copies under one parent (same read, write divided). Build at the cut that merges them back | KL < 1e-4. Tells a code bug from a wrong concept |
| Leave-one-out inside the build | one part at a time back to plain deletion: (1) union firing -> strongest member's own; (2) branch write -> strongest member's write; (3) R1 branches -> random groups of the same sizes (same-form null) | each part's share |

| Look (both networks, read whole) | ResNet-18 | CLIP |
|---|---|---|
| Whole score | KL / agree at 95/85/50/5%, 1,000 test + 800-unseen read (`fanout/r3_critic_holdout.py`) | 95/85/50/25% |
| Firing breadth per layer | built vs R students (`W95R_s0`, `W50R_s0`, `W5R_s0_x4`; `fanout/anskey.py rate`) | built vs KD student |
| Where the error enters | per-block stream error, front to back | same |
| Write change | built vs P students (`anskey.py write`) | |

Bars: whole 50% off chance (agree > 10% on the 800 unseen) on both networks. Trained targets (5,000 test): ResNet 95% .0978, 50% .5814, 5% 3.5039; CLIP KD student 1.719.

## 4. Facts that stand (use them in the look step)

| Fact | Source |
|---|---|
| Networks trained small from random fire more broadly the smaller they are (s4b0 rate .19 / .19 / .24 / .50 at A / 95 / 50 / 5%), every block from s2 on | `NODE.md` L1, `fanout/out/anskey_rate_*.json` |
| Cut-then-trained (P) students: kept writes move into deleted write directions at 95% only; at 50% writes are rewritten (.9-3.8x) | `NODE.md` L1, `anskey_write_*.json` |
| Pruned (fine) nodes use almost every stream direction (99% of kept write energy in 605-738 dims per block at 85%) | `NODE.md` L1 test 4 |
| CLIP 50% damage is spread over depth (thirds cut alone: 60.5 / 34.0 / 44.5% agree, quick-200) | `fanout/scratch.jsonl`, rules `l3_depth_*` |
| CLIP heads alone 50%: 12.0%, MLP alone 10.5%, both .5% (quick-200, stream full) | `scratch.jsonl`, rules `l3_*` |
| NOT results (parts tested alone, quick-200, no null): head fold, root order, self input, drift repair, read tilt (`rules/l1_stream.py` switches, `l2_gate_r18_*`, `l3_*`, `l4_*`) | |

## 5. Files to reuse

| File | Use |
|---|---|
| `fanout/board.py` | build + score a whole network (`--quick`, `--input`, `--split held`) |
| `fanout/torchq.py` | the Mac torch lock (2 short + 1 long, swap guard) |
| `fanout/vitgen.py` | CLIP vision tower as a dict, any width |
| `fanout/anskey.py` | reads in the saved students: `gate`, `rate`, `carrier`, `identity`, `write` |
| `fanout/r3_critic_holdout.py` | paired read on the 800 unseen photos |
| `fanout/rules/r2_p1p3_merge.py` | CLIP parent definition and the OR gate code |
| `onecoarse/code/oc_tree.py` | R1 children (ResNet) |
| `distill/wide18.py`, `distill/train_whole.py` | ResNet students, photo split, loaders |
| saved students | `../neuron/results/coarse/r18/distill_whole/` (P students have `kept` in `*_scores.pt`) |
