# The mapper's constitution

These rules are the fixed part of the mapper. They hold for every network. A rule that names one network does not belong here, and `tests/test_quotes.py` scans for this. What changes per network lives in `networks/<name>/`.

Why each rule exists: `PRINCIPLES.md`.

## 1. What clean means

Shonu, 2026-11-03: "a clean map is simple. for the scope which we care about within a network, we understand 100% of that scope. if we don't understand any part of what is in our scope, it isn't a clean map. it should be measured along an axis of progress since we are already assuming the network is clean."

| Rule | Meaning |
|---|---|
| Clean | Every node in the scope is understood |
| The number | The bar: understood nodes over nodes in scope |
| A gap in the bar | Work left. It names a suspect in the map, the tools or the test. It is never a finding against the network |
| Forbidden words in any result | "the network is not clean", "messy", "polysemantic", "no structure", as a finding |
| Forcing | Nobody forces an explanation to move the bar. An open node stays open |

## 2. A scope

A scope is written down before any reading starts.

| A scope names | Rule |
|---|---|
| Nodes | A connected slice from the logits down to the first layer. Code chooses it from the goal, with a threshold fixed in advance. An agent checks the result. The count is reported before reading. A scope can also be a whole network, and then no threshold applies. The firing limit below still holds |
| Goal | A developer's goal: weights on examples. It comes with a random control goal |
| Firings | Strong firings and both poles. Also any weaker range that code shows the goal depends on: remove that range, measure KL on the goal, and keep it in scope if KL moves |

Shonu confirmed this limit on 2026-11-03: "let's do the standing limit the essay talks about". He also tied the final build to the scope for now.

## 3. Roles

| Role | Does | May not |
|---|---|---|
| Code | Proposes links, builds sheets, computes every error, builds and scores every test | Group units by one rule to make the map |
| Reader | Follows a workflow and writes a card with judgment | Judge. See sealed examples. Pass a statement off as observed when it is inferred |
| Critic | Tries to break one card with a test specific to it. Code runs the test | See sealed examples. Rewrite the card |
| Judge | Answers from words and sealed test items only | See evidence, node name, layer, network position or author |
| Scientist | Reads stuck records and drafts a workflow | Add it to the library without the test in section 9 |
| Shonu | Spot-checks a random sample, and decides what the product promises | |

## 4. The loop

```
queued -> routed to a workflow -> read -> claimed -> critic -> judged -> UNDERSTOOD
                 ^                                               |
                 |                                               v
                 +---- next workflow, or a new one <---------  OPEN (names its suspect)
```

Only a judged pass moves a node to understood. Every move goes in the node's history.

## 5. What understood means

A node is understood when it passes all of these. A workflow may add tests for its situation, and it may not remove these.

| Part | Test | Who scores |
|---|---|---|
| The center of its job | From words alone, a judge ranks the 12 items of a sealed pool most likely to fire. Center: the firing reached by the first 6, as a share of the largest | Code |
| Its edges | From the same answer. Edge: fire items in the 12, minus near items. Near items look alike and stay quiet. Where a network takes written examples, the judge writes them from the words alone: examples that should fire it, and near misses that should not. Each node then gets a win, a tie or a loss. The exact win rule and its false-pass rate come from calibration on wrong cards, fixed before any node is judged | Code |
| Nothing unrelated in it | From the same answer. Far: fire items in the 12, minus random items. Fire items are a sample of all strong firings, so a narrow card also loses here | Code |
| Its shape | A dial gets a predicted ordering of fixed drawn items in place of the sheet above | Code |
| The negative pole | The same three scores. Where this test has failed its control on a network, rule 8 of section 6 applies there: it is reported on a sample, and it is not a pass until it is fixed. A card still states the negative side, and a critic may attack it. The bar says what it excludes | Code |
| Offshoots | The same ranking per route, with the other routes filled. Every route with a real effect must pass | Code |
| What builds it | A true parent is picked from words alone, with words from an independent reader | Key held apart |
| What reads it | A prediction of which readers it moves | Code |
| It survived attack | The critic's specific test did not break it, or a revised card passed again. The test must be able to fail: a test only on images under .5 of the node's largest firing is refused, and it counts for nothing either way | Code |

An understood node also writes its claims: the parent claims with errors computed by code. Words pass the judge. Claims feed the build.

The crop test stays as a reported number, so that rounds can be compared. It is not a gate.

## 6. The judge rules

1. An author never judges its own card.
2. A judge's packet holds only words and sealed test items. A test scans packets for leaks.
3. Every batch mixes in wrong cards, and the key stays apart from the judge.
4. Pass marks come from calibration: fixed so that wrong cards pass rarely, with the false-pass rate stated.
5. Two judges per item. A split goes to a third. Judge agreement is reported.
6. Where possible, a claim becomes a prediction that code scores with no model in the loop. Judges share a model and may share a bias.
7. Shonu spot-checks a random sample of understood nodes.
8. A test that wrong cards pass is not used until it is fixed.

## 7. No guessing, no confounding

1. Each card marks each statement as observed or inferred. An inference counts for nothing until tested.
2. Predictions are written before their results exist. Agents' self-grades do not count.
3. A comparison is fixed in advance, changes one thing, and uses matched nodes and a control arm.
4. Known confounds to check each time: node strength, reading order, author, fill, and improvement on a second read.
5. A claim about a whole layer or network needs a random sample of it.
6. Sealed examples score work. Nobody who writes a claim sees them. A guard enforces this.
7. Raw numbers are reported, including the ones that go against the work.

## 8. Workflows

A workflow is a short file, with these fields.

| Field | Holds |
|---|---|
| Trigger | What code can see that selects it |
| Steps | Ordered evidence calls, each with the question it answers |
| Step limit | The most calls for one node |
| Exits | Claim ready, switch to a named workflow, or open with a suspect |
| Required tests | What this situation adds to section 5 |
| Version and record | Which nodes used it, and how they scored |

A reader may leave the steps when the evidence calls for it, and must write why. The step limit and the exits still hold.

## 9. Adding a workflow

1. A node leaves its workflows and stays open, with a stuck record: what was tried, what the suspect is, and what a new tool would show.
2. A scientist reads the stuck records and the relevant part of the essay, and drafts one workflow.
3. The draft runs on the stuck nodes, and on a sample of understood nodes as a regression check.
4. The judge scores both. The workflow enters the library only if stuck nodes move, understood nodes hold, and wrong cards still fail.
5. It gets a version number and a log line. Any new motif goes in the library.

A tool gets built only when a named stuck node needs it.

## 10. Suspects for an open node

Each open node names one, and the next step tests it.

| Suspect | Test |
|---|---|
| A tool shows the node wrongly | A better view either resolves it or does not |
| The explanation is wrong or too broad | The critic, or a fresh reader |
| The data is thin or wrong | A broader set of examples |
| A motif that we do not have yet | Read the neighbours, then read it again |
| An offshoot from growth | Split by routes. Watch it grow if checkpoints exist |
| A regime of the architecture | Split by the sign or state of its parts |
| How the network learned | Only a retrain test can show this. Never write it without one |

The essay's sixth cause, a change of learning principle, stays last. It is never written as a result.

## 11. Limits

| Limit | Value | When it is hit |
|---|---|---|
| Steps per node | The workflow's step limit | The reader takes an exit |
| Workflows tried per node | 3 | The node stays open, with its stuck record |
| Reads after a tool fix | 1 per node. It spends no workflow slot, since the workflow is not what failed. Added by delegation on 2026-11-04, and Shonu can remove it | The next read spends a workflow slot |
| New-workflow drafts per stuck group | 2 | The group stays open, and Shonu sees what was tried |
| Tokens in one step of a plan | 2 times that step's written prediction | Stop and show Shonu |
| Frozen tests of the library | 2 | Stop and show Shonu the result, whatever it is |
| A large batch | Runs only after a smoke batch passes | |

When a limit is hit, the map stays "not done". Nobody lowers a test to finish.

Shonu, 2026-11-03: "we shouldn't think of this as a token rampage ... this should be thought of almost as its own tool which is engineered properly".

## 12. Knowing that we can rinse and repeat

Shonu, 2026-11-03: "the least amount of the network we need to feel highly confident that we could just rinse and repeat for the rest of the network".

1. Bring a scope to 100%.
2. Freeze the library.
3. Run it on an unseen slice with a different goal. Where the scope was a whole network, "unseen" means the next network.
4. Pass: the slice or network reaches understood with no new workflow, at the same cost per node.

So the size of the scope comes from this test.

## 13. Records

| Record | Rule |
|---|---|
| Every result | Raw numbers, its prediction, and tokens per node |
| Each log entry | The two rows "Cale line" and "Whole-network build", the second saying yes or no |
| Bright line | Fixed process, record, judge rules and workflow format: yes. One fixed rule that groups or reads units: no |
