# Pilot reader brief (N10)

You read ONE neuron of ConvNeXt-Base (an ImageNet classifier) and write a card that a blind judge can test.

## Your inputs (in your packet folder)

| File | What it holds |
|---|---|
| `evidence.txt` | Numbers and class lists: the neuron in Base, in B (Base shrunk to half its neurons, no training) and in T (B after training) |
| `sheet_base.png` | Photo crops centred where the neuron is strongest. Rows: top 0.1%, p99-99.9, p90-99, p50-90, then the negative pole (most negative 0.1%, p0.1-1) |
| `sheet_B.png`, `sheet_T.png` | Same crops for B and T (top rows and most negative row) |

Values are pre-activations, before the GELU. The positive pole is the max over positions in a photo. The negative pole is the min.
For a DELETED neuron, B and T values are virtual: Base's read row applied to that network's block input. They say whether the information this neuron read still reaches its block.

## How to read (the essay's rules)

1. Name the job as a VERB: what the neuron lets through and what it keeps out. "Separates small perched birds from water birds" is a job. "Birds, dogs, boats" is a noun list, and a noun list is not a reading.
2. Read BOTH poles. The negative pole is the distinction that matters at this point. It is not "the opposite".
3. Read by percentile band. Say whether the middle bands (p90-99, p50-90) follow the top band (a dial) or not (a detector).
4. Use the feeders and readers to say where the job comes from and what uses it.
5. Say how Base, B and T differ. For a survivor: did T widen, narrow or shift the job? For a deleted neuron: does the information it read still reach the block in B and T?
6. If the evidence is messy, say what tool or view is missing. Do not invent a story to cover it. Blame the microscope first.

## Your output

Write `card.md` in your packet folder, with these headings:

- **Job (verb)**: one or two sentences.
- **Positive pole**: what the top bands show. Say the shared pattern, not only object names.
- **Negative pole**: what the most negative bands show, and the distinction this makes.
- **Dial or detector**: one line, with the evidence.
- **Feeds from / read by**: one line each.
- **Base vs B vs T**: two or three lines.
- **Confidence**: high / medium / low, with the reason.
- **Missing tools**: a short list of views or measurements that would have helped.

Keep the card under 300 words. A blind judge will use only the card to guess which classes make the neuron fire most and least, so write the shared pattern clearly.
