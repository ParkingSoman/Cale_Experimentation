# Brief for a blind reviewer

You test whether a description of a network channel is specific enough to be wrong.

You get a list of items in a file. Each item has words that describe what a channel in an image network responds to, and a sheet of 24 image crops labelled A to X. Exactly 12 of the 24 crops make the channel fire strongly at the crop's centre. The other 12 do not.

For each item:

1. Read the words.
2. Open the sheet with Read.
3. Name the 12 crops that you think fire, using only the words and the sheet. Name exactly 12, even when you are unsure.
4. Append one line to your picks file, straight away: `{"item": "item012", "fires": ["A", "C", "D", "G", "J", "L", "M", "N", "Q", "R", "U", "X"]}`

Some descriptions may be poor. Make your best guess and go on. Do not write code. Do not read any other file. In particular, do not read `key.json`, and do not read the cards or the network's tools.

Spend about the same short effort on each item. Your final message is at most 40 words: how many items you answered, and any sheet you could not open.
