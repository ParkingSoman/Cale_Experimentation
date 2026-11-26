# Pattern reader for the one-coarse project

Shonu's standing grant (2026-11-22) lets general-purpose agents act as this project's blind readers. You describe what one unit of an image network responds to, from pictures alone.

For each job in your list you get two images:

- a crop sheet: 20 photo crops (2 rows of 10), each centred on a place where the unit responds most strongly in one direction;
- a mean image: on the left, the average of 60 aligned windows (colour, then edge map); then up to three sub-groups, each as a colour average and an edge map, with its size. Bright lines in an edge map are edges shared by many windows.

Write words that let a stranger pick out NEW crops of this kind from a mixed set. Rules:

1. Describe the shared PATTERN at the centre: shape, outline, layout, texture, contrast, colour, scale (how much of the crop it fills), and what surrounds it.
2. Use NO names of objects, animals, people or body parts (no "dog", "face", "eye", "nose", "wheel", "person", "flower" and so on). If you catch yourself naming a thing, say instead what shape or layout makes it look that way.
3. One or two sentences, at most 50 words. If there are two sub-patterns, name the larger first.
4. Read only the images in your list. Read nothing else in this project. Do not guess what the network is.

Write one JSON file per job to the path your task gives, as {"<key>": "<your words>"}. Your final message says only "done".
