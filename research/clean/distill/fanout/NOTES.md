# FANOUT NOTES: one line per experiment, newest at the bottom

Read the last 40 lines before every experiment (`tail -40`). Append one line after every experiment:

```
HH:MM [WHO] tried: <one change> | result: net f KL top-1 agree (quick or full) | vs: <the row it beats or loses to> | next: <what you try next> | @OTHER: <message, optional>
```

Address a teammate with @P0, @P1-P3, @P4, @P5, @P6, @JOINT. Read every line addressed to you. Answer in your next line.
Append with `echo "..." >> distill/fanout/NOTES.md` (one short line, no line breaks inside).

Reference rows (quick = 200 photos, every 5th class; full = 1,000 photos):
- floor (noisecut, shared pink input): CLIP 95% 1.139 full; ResNet 95/50/5% 1.665 / 9.386 / 14.949 full
- flat level: CLIP 6.281, ResNet 5.737. Trained: CLIP ViT-T 1.801, ResNet .097 / .856 / 3.564

---
