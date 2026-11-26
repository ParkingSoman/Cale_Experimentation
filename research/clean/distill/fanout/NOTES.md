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
14:51 [P0] tried: CLIP self generator (make_input --net clip --kind self, K32, smoke 1.27 s/step) | result: running, holds the torch lock about 5 min; null generator another 5 min later | predict (written before): CLIP self score rank .571 -> about .70, MLP rate ratio .353 -> about .6, floor 95% agree up 2+ points over pink, null stays at pink | next: floor quick .95,.85 on self vs pink | @ALL: P0 holds the lock ~5 min twice (CLIP generators)
14:58 [P6] tried: stream-only, keep a PCA SUBSPACE of the centred stream on X (one basis, pooled raw 2nd moment, 25 sites), rms divisor 768*rho_l | result: clip quick f1 9e-7/100%; .95 .251/81.5%; .85 1.336/54.5%; .5 9.473/0.0% (kept energy mean .907, embed site .158) | vs: r1 coord rho .245/82.2% (.95 full), 9.718/.2% (.5 full) = tie, 50% still at chance | next: reader-weighted basis + per-site-normalized PCA + random-Q null | @all: 91% of stream energy kept and still chance at 50%, energy is not content
14:59 [P1-P3] tried: row (a) = floor MLP-only + P4 fold via my own MLP loop (r2_p1p3_a) | result: clip quick .95 .330/79.0%, .85 1.038/64.0%, .5 5.968/3.0% | vs: fold-off (p4write_off_mlp) .493/75.5, 2.589/41.0, 9.140/.5 | next: rows d (R1 OR-merge) and f=1 check | @P4: your MLP fold reproduces in my loop; .85 gain 2.589->1.038 is large
15:00 [JOINT] tried: r2_joint_core (ResNet masked full-width; switches order floor/p5/p5null, mean off/mean/perm, fold off/on/null) all-floor row | result: r18 quick .95 1.937/35.5%, .85 7.890/3.5% (85% already near chance on ResNet) | vs: noisecut quick (check queued in my sweep) | PREDICT before sweep: at .95 P5 (+3.7 agree) and meanfold (+4.6) add, P4 inner fold overlaps meanfold (r1 null repeated 91% of P4 gain), joint about +7 agree over floor; .85 all near chance | next: one-job sweep 2x2x2 + 3 nulls at .95,.85 | @P5 @P6 @P4: I use your round-1 code read-only (p5rank cache, _p6diag_r18, p4write.fold_map)
15:01 [P5] tried: CLIP MLP-only cut by weight^(1-a) x noise-activity^a, a=0 (= weight ranking, same code) | result: clip quick .95 .202/83.5%, .85 1.271/63.0% | vs: round-1 weight ranking .205/83.7% full (reproduces) | next: one batched process (queue waits 6 min): a=.25/.5/.75/1, per-set greedy order (MLP only, heads only), heads mix | @ALL: queue wait is ~6 min per call, batch your rows in one process
