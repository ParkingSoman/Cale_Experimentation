---
name: feedback-disk-hygiene
description: Keep disk free of useless files; the disk is shared with another session and is near full
metadata:
  type: feedback
---
Shonu asked (2026-10-01) that I actively keep the disk free of useless files. He told the other session (BlinkFix) the same.

**Why:** the 926 GB disk sits at ~97% full; two sessions write large model files. Running out breaks both projects and the system.

**How to apply:**
- Keep `Cale_Experimentation/DISK.md` current: every large file, why it exists, when to delete it.
- Build -> score -> delete. At most one 8-bit and one 4-bit nest build at a time. No duplicate weight copies.
- Delete dataset download caches once samples are saved to data/.
- Check `df -h ~` before writing >5 GB. Stop if free space would drop below 15 GB.
- Never delete outside the project except HF cache entries this project created. Never touch BlinkFix-models.
- GitHub backup exists ([[reference-github-vercel]]). History: [[project-research-history]] section 1.
