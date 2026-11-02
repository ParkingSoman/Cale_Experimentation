"""Anchor merge: apply the worktree's changes (base -> wt) onto main by placing each hunk next to unchanged
neighbour lines found in main. Refuses a hunk whose anchors are not unique in main. Usage: base wt main [--apply]."""
import difflib, sys
base, wt, main = (open(p).read().split('\n') for p in sys.argv[1:4])
apply = '--apply' in sys.argv
ops = [o for o in difflib.SequenceMatcher(None, base, wt, autojunk=False).get_opcodes() if o[0] != 'equal']
out = list(main); shift_ok = True
# work from the end so earlier indices stay valid
for tag, i1, i2, j1, j2 in reversed(ops):
    K = 3
    before = base[max(0, i1 - K):i1]; old = base[i1:i2]; after = base[i2:i2 + K]
    pat = before + old + after
    hits = [k for k in range(len(out) - len(pat) + 1) if out[k:k + len(pat)] == pat]
    if len(hits) != 1:
        print(f"REFUSE hunk base {i1}-{i2}: {len(hits)} anchor hits"); shift_ok = False; continue
    k = hits[0] + len(before)
    out[k:k + len(old)] = wt[j1:j2]
print(f"{len(ops)} hunks, all placed: {shift_ok}")
if apply and shift_ok:
    open(sys.argv[3], 'w').write('\n'.join(out)); print("written", sys.argv[3])
