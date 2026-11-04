# Motif library

A motif is a pattern that a reader should expect, so that the next node is cheaper to read. Essay line 43: "you start to know what to expect from the next one". A motif is a lead, and it never replaces a test.

| Status | Meaning |
|---|---|
| seen | An agent wrote it down. No blind test backs it yet |
| tested | Cards that use it passed the judge. The file says which |

A reader adds a motif with `add`-style edits to the network's file: one short entry, what it looks like, where it was seen, and how to test it. A scientist agent promotes it to "tested" with evidence from the state store.

Link grammar, for any network. Source: `research/clean/MAP-STATE.md`.

| Link | What it looks like |
|---|---|
| Refinement | A finer version of the parent's pattern |
| Near-twin relay | Nearly the same job, passed along |
| Pooling | Several variants treated as one |
| Sibling competitor | The strongest negative input does a related job |
| Part inside a whole | The part feeds the whole, and may also be held back |
| Dial to thing | A graded signal feeds things of that kind |
| Child on the parent's off side | A gate's negative side leads to its own specialists |

Files: one per network, such as `resnet18.md`.
