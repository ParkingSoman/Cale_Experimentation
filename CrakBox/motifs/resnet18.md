# ResNet-18 motifs

Seeded from `research/clean/curves/map_agents/resnet18/NOTES2.md`, the agents' shared notes of round 2. All entries are "seen", since no blind test backs them yet. Stage 4 only so far. Nothing here is known below stage 3.

## Patterns

| Motif | What it looks like | Seen at | How to test |
|---|---|---|---|
| Radial on the negative pole | Spokes, wheels, domes, clock faces on the negative side of many inner channels | `s4b1i:64, 198, 47, 460, 58, 393, 189, 452, 239` | Look at the negative pole for a round or radial thing before naming a group of nouns |
| Wedge in side view | A forward-pointing wedge, from muzzle and beak up to a hull's bow | `s3b1:141/214 -> s4b0i:261 -> s4b0:418 -> s4b1i:317 -> s4b1:32` | Rotate by 90 degrees: firing should fall to about .2 to .3. Gray keeps about .9 |
| Thin line, then loop, then handle | Thin curves build chains and loops, then a handle on a body | `s4b0i:496 -> s4b0:327 -> s4b1i:110 -> s4b1:175` | Draw a loop against spokes. Draw a mug with and without a handle |
| Repeated thin parallels | Keys, folds, ribs and bristles | `s4b0:467`, `s4b1i:265`, `s4b1:467` | Drawn parallel lines at several spacings |
| Long thin strands | Silky hair and clock hands, by separate routes | `s4b1:242`, `s4b1i:29` | Drawn strands do not fire it. Use real crops |
| Spots on a ground | Small regular marks on a plain ground, against a solid coat | `s4b0i:326 -> s4b0:35 -> s4b1i:189 -> s4b1:35` | Drawn dark spots on a pale blob |
| Soft white billow | Fluff and spray | `s4b0i:146 -> s4b0:168 -> s4b1i:349 -> s4b1:168` | Gray and blur, on real crops |
| Big smooth mass | One big sloping mass in an open scene | `s3b1:154 -> s4b0i:344 -> s4b0:226 -> s4b1i:23 -> s4b1:226` | Zoom in and out |
| Mirror pairs | Each one's negative pole is the other's positive | `s4b1:156` and `s4b1:9`, `s4b0:153` and `s4b0:345`, `s4b1:255` and `s4b1:163` | Compare both poles of the pair |

## Reading the tools on this network

| Lesson | Detail |
|---|---|
| Drawn shapes | They work for line structure. They mostly fail for round bodies and late-stage texture: under .25 of the peak |
| Cheap checks | `edit` with rot90 and gray. Many stage 4 nodes fall to .2 to .4 on rot90 |
| Skip at block 1 | Stages 1 to 3: the skip is about .7 of z at a peak. Stage 4: about .07, so block 1 builds the job there |
| Same number, same job | At stage 4 a block 0 stream channel often feeds the inner channel that its own number feeds. It is a lead, and not a rule |
