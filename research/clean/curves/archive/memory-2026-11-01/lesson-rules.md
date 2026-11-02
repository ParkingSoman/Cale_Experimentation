---
name: Lesson Capture Rules
description: Durable corrections captured across sessions — follow these when applicable
type: feedback
---

RULES — follow these when applicable.

[L143|machine] Before declaring a proof or result complete, check that the proved case covers real instances; a proof on a degenerate subcase (e.g. an empty or zero set) is not completion.
  Why: The user rejected the claim that the session was done because only the trivial zero case was handled, and the assistant then changed plan to prove the graded case that real networks actually exercise.
[L142|machine] For theory-heavy requests, start from the mathematical objective and argue it paper-style, with statements and proofs; engineering-framed reasoning gives weaker results for this user.
  Why: The user said engineering-style framing performs worse for them, and the assistant then rewrote the plan around the math objective and drafted the section as theorems and proofs.
[L141|machine] When a reimplemented op matches the framework's outputs but its gradients are off, compare the selected indices (e.g. max-pool argmax) against the framework's own index output; tie-breaking differences cause the gradient gap.
  Why: The reimplementation matched logits exactly but gradients differed by 6%; comparing max-pool window selections to PyTorch's indices located the mismatch and brought the check to 2e-7.
[L140|machine] When an experiment fails, derive the cause from the math before changing settings; stacking optimizer tweaks is guessing, and each blind run costs time without isolating the failure.
  Why: After the user said four optimizer tweaks on G2 in a row were engineering, not math, the assistant saved that rule, wrote the mathematical explanation of the failure before any new run, and stopped for the user's decision.
[L139|machine] Before iterating on an experiment, shrink it (fewer layers, samples, or fit steps) until one run takes seconds to minutes; multi-hour runs cut the number of attempts you can make.
  Why: The user said the 2-hour runs were too slow to iterate and asked for a smaller scale, and the assistant then cut the layers, parents and fit time to get sub-minute loss checks.
[L138|machine] When a design needs limits, thresholds, or security rules, look up the established practice in primary sources before choosing values; invented limits drift from what browsers and tools actually do.
  Why: The user said not to invent a solution when a good practice exists, and the assistant then replaced the sub-agent's invented limits with browser-standard rules from primary sources.
[L136|machine] Report a cross-machine dry run as passed only if every step ran end-to-end on that target machine with the final artifacts in a fresh folder; a partial run elsewhere does not count.
  Why: The user asked whether the dry run had been done on the HP, the assistant admitted it was only partial, and after the user said yes it reran the full install on the HP with final files and reported that.
[L137|machine] Before asking the user a question, search past session logs for an existing answer; the user has often already tested it, and re-asking wastes their time and signals the prior work was ignored.
  Why: The user asked twice to mine previous sessions because many questions were already tested, and the assistant then searched the logs and dropped the Milestone 0 questions that were already answered.
[L135|machine] Once the user picks an option, carry the whole chosen plan through without asking permission at each step; pause only at real decision points, since repeated check-ins make the user repeat their instruction.
  Why: The user said "continue rather than asking for permission continuously. go with A," and the assistant then ran the remaining build steps without re-asking.
[L134|machine] To close an unexpected channel a sandboxed process still uses, deny all system call-outs, read the denial log to name the serving service, then block that service's socket file directly.
  Why: DNS kept working after network was denied, so the assistant denied every call-out, read the log, found the mDNSResponder socket, blocked it, and confirmed DNS failed.
[L133|machine] When GitHub's PR list returns mergeable state UNKNOWN, it has not computed the answer yet; query each PR individually before reporting it clean, or conflicts will be missed.
  Why: The report script initially called conflicting PRs 'fine' because the list view returned UNKNOWN, and after switching to per-PR queries the report matched the manual check.
[L132|machine] When review rounds keep finding one more unlisted case, replace the deny-list with an allow-list; a deny-list is only as complete as the cases someone already thought of.
  Why: After four review rounds each adding one more forgotten protected file, the assistant stopped patching the list and rewrote the shipper to accept only listed files, which was pushed and tested.
[L131|machine] Omit AI-tool attribution footers and session links from commit messages and PR bodies, and strip them from existing PRs too; the user doesn't want them in review.
  Why: After the user said not to put Claude session links or 'generated by Claude Code' in the PR, the assistant left them out of commit and PR text and removed the footer from three existing PRs.
[L130|machine] After a squash merge, a branch still holding the PR's original commits conflicts with the squashed copy, because git can't match them; rebase onto the new base so only your own commit remains.
  Why: PR 75 was squash-merged, its old commits clashed with the squashed copy on testing, and the assistant rebased to keep only its own commit, which applied cleanly and passed 218 of 218 tests.
[L129|machine] Keep long text out of GitHub Actions ${{ }} expressions: past about 21000 chars GitHub marks the whole workflow invalid, so read long prompts from a repo file at run time.
  Why: GitHub rejected pipeline-agent.yml with 'Exceeded max expression length 21000' after the inlined prompt grew, and the assistant then measured the prompt, trimmed it to 20,045 chars, and added a length guard test.
[L128|machine] On Windows OpenSSH, a job started detached inside the SSH session dies when the session closes; launch long jobs via WMI Win32_Process Create and confirm a log appears before polling.
  Why: The assistant saw a 'started' launch produce no log, found WMI start ran to completion after SSH closed, and switched scripts/hp to that launch method.
[L127|machine] When a remote machine needs a large file that exists online, have the remote download it directly; relaying it through your own machine over SSH is capped by that link and is much slower.
  Why: The user said a 3 GB download on the HP took about a minute while the assistant's SSH relay ran at about 10 MB/s, and the assistant stopped the copy and switched to having the HP pull from Hugging Face.
[L125|machine] Don't trust a green CI test step until its duration and uploaded output confirm it ran; a near-instant pass with no artifact usually means the command silently did nothing.
  Why: The Windows screen-reader job passed in about one second with no transcript, and the assistant caught the empty pass from the step timing and log, then edited the workflow and re-ran CI.
[L124|machine] Rules meant to constrain an unattended or cloud agent must be enforced in the repo's code or config, not in personal memory notes, because those runs never read the memory.
  Why: After the user said a memory note was not enough, the assistant noted the cloud agent never reads memory, then moved the design-approval lock into the repo's resume step on a new branch and opened a PR.
[L123|machine] When delegated reviewers complete assignments suspiciously fast without engaging core review steps, add explicit process requirements to the next batch; unchecked speed indicates insufficient diligence.
  Why: The assistant observed the first two review agents finish in ~15 seconds without reading the voice guide and report 'no changes', then explicitly added guidance to subsequent agents—'reminder to read the voice guide and check each card'—resulting in more deliberate reviews from the rest.
[L125|session 2026-10-29] When merging untracked paper files from a worktree back into a main tree that another session also edited, do not trust `diff3 -m`: on macOS it silently dropped 35 lines (Definitions D1 to D4) with exit 0. Back up main first, apply my edits as context-anchored patches (difflib base -> mine, each anchored uniquely in main), and verify: no main line lost, every new line of mine present, line count = main + mine - base. Appends at the end of a file go by hand (anchors put them before the other session's appended section).
  Why: the first merge passed lint on its own items but lost unrelated text; a set-difference check caught it and main was restored from the backup.
[L126|session 2026-10-29] Before writing a report file, check that the name is free (REFEREE-BV3-2026-10-29.md already existed for an older round).
  Why: I overwrote the worktree copy of an older referee report; the main tree's copy let me restore it.
