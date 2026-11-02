# resnet18 cards, agent r18c (8 start channels plus 7 neighbours; 15 channels read)

Level 0 channel ids. Feeders are level-1 channels from feeders-by-photo. "Held" = prediction result.

| Ch | Distinction (no noun list) | Negative pole | Feeders (level 1) | Readers | Prediction and result | Sure |
|---|---|---|---|---|---|---|
| 56 | smooth rippled bright liquid surface vs opaque matte close solid | close-up solid filling the frame (artichoke, fur, bottle cap) | one route: 39,152,239,43,1 | water vehicles and swimmers (shark, scuba, ship) | Pred: fires more on underwater/open sea than on shore/dock. Held weakly: sharks 0.98 pct, dock/lakeside 0.9. 56 is water texture everywhere | med-high |
| 510 | water meeting a human-built or shore edge | alp, geyser (pure rock, no water) | 28,130,15,46 | dock, lakeside, schooner, breakwater | Pred: low on underwater shark/coral, high on shore. Held: 0.45-0.67 vs 0.9+ | med |
| 435 | huge smooth gradient field (sky, haze, steam, sunset) vs small handled thing | hands, held tools, masks (close, hand-scale) | one route: 98,32,57,106,241 | tall landmarks seen against sky (church, beacon, obelisk) | Pred: high on geyser/alp, low on sharks. Held: 0.96-0.98 vs 0.4-0.6 | high |
| 481 | far horizon band: ground below, sky above | not read | diffuse: 229,26,57,106 | geyser, sandbar, volcano, megalith, camel | Same pred as 435. Held (0.93-0.98 vs 0.4-0.58) | med |
| 193 | thin straight structure cut against a smooth blue field | not read | 103 (key), 39, 37, 68 | wing, solar dish, flagpole, shark | Pred: low on shark. FAILED: 193 fires on shark (0.94) as it does on sky. Shark fin on blue water counts as a thin edge on smooth blue | med |
| 63 | bulging round wall with ring bands (a cylinder seen from the side) vs flat angular face | carton, box, fan, cauliflower | 47,66,38,51,234 | barrel, thimble, bucket, cup | not tested | med |
| 405 | rows of upright glossy label-carrying bottles | not read | not read | bottles, lotion, vending machine | not tested | low |
| 322 | round flat disk | not read | not read | wheel, plunger, puck, potter's wheel | not tested | low |
| 319 | long vertical parallel-edged body running through the crop vs radiating spikes or fine irregular pattern | spiky, spidery, puzzle, keyboard | diffuse, no feeder above 0.05: built from many | ski, flute, obelisk, pole, sail | not tested | med |
| 27, 199, 314, 377 | readers only (27 snow/sled and cold; 199 small vehicles; 314 pale fur carnivores; 377 boxy kitchen/office appliances). No sheets seen | | | | | low |

Offshoots: 56 water vs 510 shore edge is two routes to one theme. 193 grabs shark fin through the same "edge on blue" route as wing.
