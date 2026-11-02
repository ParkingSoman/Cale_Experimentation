# Brief for a map agent, round 4 screen S4

Your job: read a trained ResNet the essay's way, and write down, as a tree on its 1000 classes, what its output table is built from. Code scores it. You are not building an algorithm. Do not simply cluster and hand in the result. One Ward clustering of the head weights is already the baseline, and we want what judgment can add. Clustering as a tool to look at things is fine.

Shonu, who leads this project: "you need to have subagents figure that stuff out rather than trying to develop a one size fits all sort of algorithm".

## The object

Class-row table: sbar_c = the mean of softmax(y(x)) over the photos x of class c. 1000 rows, 1000 classes. A tree weight claim says, off the diagonal,

   log sbar_c[i] = alpha_c + beta_i + f(lca(c, i))

with a row scale alpha_c, a column scale beta_i, and one number f for each internal node of the tree. Weights in the fit and in the score are sbar_c[i], so mass-bearing entries count, and tiny ones do not. Definition DQ19 in `COARSE-THEORY.md`.

## Files (under `research/clean/curves/`)

| Item | Where |
|---|---|
| The essay | `map_agents/essay.md`. Read lines 33-35, 69-105, 145-161, 225-251, 399-403, 465-482 first |
| Channel-reading notes from an earlier pilot, with tools | `map_agents/BRIEF.md`, `map_agents/NOTES-resnet18.md`, `map_agents/NOTES-resnet50.md`, `map_agents/evidence.py`, `map_agents/cards/` |
| Outputs y, pre-activations z, values h | `cache/map/resnet18_y.npy`, `_z.npy`, `_h.npy`, same for `resnet50`. 20000 validation crops, 20 per class |
| Labels, photos, head weights W, bias b | `cache/map/resnet18_meta.npz`, keys `labels`, `photos`, `W`, `b` |
| Class names | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]`, in `.venv` if torchvision is there |
| Scoring tool | `.venv/bin/python pc_r4_s4_eval.py --net resnet18 --tree yourtree.json --split A` |

Tree format: nested JSON lists, leaves are class indices 0 to 999, each exactly once, any fan-out.

## Rules

1. Score on `--split A` only. Never run `--split B`. The main session runs B after you finish. Breaking this voids your result. Do not read or recompute the row split in order to fit to B's rows, and do not use the rows of B in choosing your tree: the row split is in the tool, with a fixed seed. If your reading uses the whole table, that is fine only for reading N's weights, channels and photos. Do not tune against table rows outside A.
2. Run Python with `.venv/bin/python`. Any script needs a first line `# serves: DQ19`, or a guard will block it. Inline Python may not import torch. Put your scripts in your own folder.
3. Do not edit `COARSE-THEORY.md`, `DECISIONS.md`, `LADDER.md`, `ROADMAP.md`, or any existing script. Write only in your own folder.
4. A good explanation is specific enough to lose. Write a prediction before you score a tree: the number on A, and why.
5. Stop at about 300000 tokens, or sooner when you stop improving. A tree with a clear reason beats a slightly lower number with none.

## Hand in

In your folder: `tree_resnet18.json`, `tree_resnet50.json` if you reach it, and `REPORT.md`: in short plain sentences, what structure you read, how it became a tree, your predictions and their scores, what the tree could not express, and whether a different claim form would fit what you saw. If a different form seems right, state it: what numbers it names, what its error is, that the error is 0 exactly when it holds, and that it names no build, size or zoom. Do not claim more than you measured. Return your REPORT.md text as your reply.

For reference, on split A of ResNet-18: Ward on W 1.357, shuffled tree 1.730. A rank-14 fit is about 1.33, on all rows. The line is 1.0.
