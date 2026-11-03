#!/usr/bin/env bash
# Self-test for the math-mode guards. Each case states the expected verdict. Prints PASS/FAIL per case.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
CURVES="$(dirname "$T")"
TMP="$(mktemp -d)"
fails=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1 (expected $2, got $3)"; fails=$((fails+1)); fi
}

# ---------- paper linter ----------
good="$TMP/good.md"
cat > "$good" <<'EOF'
# Part A: starting points
<!-- literature-table -->
| Source | Verdict |
|---|---|
| Sparse autoencoders | incompatible as a framing |
<!-- /literature-table -->
**Hypothesis H-clean.** The family is an exact orbit.
# Part B: theorems
**Theorem 1 (under: none).** Group averaging cannot raise a convex invariant objective.
Proof. Jensen, then invariance. QED.
Lean: reynolds_le
**Theorem 2 (under: H-clean; CONJECTURE).** The coarse unit is the orbit average.
Route: Fourier on the cyclic group.
Lean: none, continuous group
# Part C: distance from the hypotheses
H-clean: eps = .22 measured on 3b, dev split.
EOF
python3 "$T/paper_lint.py" "$good" >/dev/null; check "lint: good paper is clean" 0 $?

bad1="$TMP/bad1.md"; sed 's/^Proof. Jensen.*/CHECKED on the dev images./' "$good" > "$bad1"
python3 "$T/paper_lint.py" "$bad1" >/dev/null; check "lint: CHECKED + dev in Part B fails" 1 $?
bad2="$TMP/bad2.md"; sed 's/(under: none)//' "$good" > "$bad2"
python3 "$T/paper_lint.py" "$bad2" >/dev/null; check "lint: missing hypothesis tag fails" 1 $?
bad3="$TMP/bad3.md"; sed 's/^Lean: reynolds_le/Lean: no_such_theorem/' "$good" > "$bad3"
python3 "$T/paper_lint.py" "$bad3" >/dev/null; check "lint: unknown Lean name fails" 1 $?
bad4="$TMP/bad4.md"; sed 's/^Route: Fourier.*/We expect it from the dictionary of features./' "$good" > "$bad4"
python3 "$T/paper_lint.py" "$bad4" >/dev/null; check "lint: conjecture w/o route + dictionary framing fails" 1 $?
bad5="$TMP/bad5.md"; sed 's/^H-clean: eps.*/nothing here/' "$good" > "$bad5"
python3 "$T/paper_lint.py" "$bad5" >/dev/null; check "lint: hypothesis without Part C eps fails" 1 $?
bad6="$TMP/bad6.md"; sed 's/^Proof. Jensen.*/The cos is .94 so it works./' "$good" > "$bad6"
python3 "$T/paper_lint.py" "$bad6" >/dev/null; check "lint: experiment number in Part B fails" 1 $?
bad7="$TMP/bad7.md"; sed 's/^Proof. Jensen.*/The fit is 0.43 here./' "$good" > "$bad7"
python3 "$T/paper_lint.py" "$bad7" >/dev/null; check "lint: leading-zero decimal in Part B fails" 1 $?

# ---------- decision ledger ----------
python3 "$T/test_ledger.py"; check "ledger: self-test" 0 $?
echo "{\"tool_name\":\"Edit\",\"tool_input\":{\"file_path\":\"$CURVES/DECISIONS.md\"}}" | python3 "$T/paper_lint.py" --hook
check "lint hook: edit of DECISIONS.md is linted and clean" 0 $?

# hook mode ignores other files
echo '{"tool_name":"Edit","tool_input":{"file_path":"/tmp/other.md"}}' | python3 "$T/paper_lint.py" --hook
check "lint hook: other file ignored" 0 $?

# ---------- run gate (needs the marker; create it only for the test if absent) ----------
made_marker=0
[ -f "$CURVES/.math-mode" ] || { touch "$CURVES/.math-mode"; made_marker=1; }
had_paper=0; [ -f "$CURVES/COARSE-THEORY.md" ] && had_paper=1
[ $had_paper = 1 ] || printf '# Part B\n**Theorem T1 (under: none).** x\nProof. x\nLean: none, test\n# Part C\nH-sym: test\n' > "$CURVES/COARSE-THEORY.md"
mkdir -p "$CURVES/model"
s_none="$CURVES/model/_gate_test_none.py"; printf 'print(1)\n' > "$s_none"
s_ok="$CURVES/model/_gate_test_ok.py"; printf '# serves: T1\nprint(1)\n' > "$s_ok"
s_inc="$CURVES/model/_gate_test_inc.py"; printf '# serves: T1\nimport lucent\n' > "$s_inc"
s_inc2="$CURVES/model/_gate_test_inc2.py"; printf '# serves: H-sym\nimport lucent\n' > "$s_inc2"
gate() { jq -n --arg c "$1" --arg d "$CURVES" '{tool_name:"Bash",tool_input:{command:$c},cwd:$d}' | python3 "$T/run_gate.py" 2>/dev/null; echo $?; }
check "gate: ls allowed" 0 "$(gate 'ls')"
check "gate: script without serves blocked" 2 "$(gate "python model/_gate_test_none.py")"
check "gate: script serving T1 allowed" 0 "$(gate ".venv/bin/python model/_gate_test_ok.py")"
check "gate: InceptionV1 script serving a theorem blocked" 2 "$(gate "python model/_gate_test_inc.py")"
check "gate: InceptionV1 script serving H-sym allowed" 0 "$(gate "python model/_gate_test_inc2.py")"
check "gate: inline torch blocked" 2 "$(gate "python -c 'import torch; print(1)'")"
check "gate: tools/ exempt" 0 "$(gate "python3 tools/paper_lint.py")"
check "gate: cd then relative tools path allowed" 0 "$(jq -n --arg c "cd $CURVES && python3 tools/paper_lint.py" '{tool_name:"Bash",tool_input:{command:$c},cwd:"/tmp"}' | python3 "$T/run_gate.py" 2>/dev/null; echo $?)"
check "gate: cd then unanchored script blocked" 2 "$(printf 'x\n' > "$CURVES/model/_gate_cd.py"; jq -n --arg c "cd $CURVES/model && python3 _gate_cd.py" '{tool_name:"Bash",tool_input:{command:$c},cwd:"/tmp"}' | python3 "$T/run_gate.py" 2>/dev/null; echo $?; rm -f "$CURVES/model/_gate_cd.py")"
check "gate: .py name inside inline code is not a run" 0 "$(gate "python3 - <<'EOF'
p='model/t3_pairing.py'
EOF")"
check "gate: interpreter flags before an unanchored script still blocked" 2 "$(printf 'x\n' > "$CURVES/model/_gate_flag.py"; gate "python3 -I -u model/_gate_flag.py"; rm -f "$CURVES/model/_gate_flag.py")"
check "gate: script created in the same command blocked" 2 "$(gate "printf x > model/_gate_new.py && python3 model/_gate_new.py")"
rm -f "$s_none" "$s_ok" "$s_inc" "$s_inc2"
[ $had_paper = 1 ] || rm -f "$CURVES/COARSE-THEORY.md"

# ---------- ladder and card lint ----------
lg="$TMP/ladder_good.md"; lc="$TMP/cards_good.md"
cat > "$lg" <<'EOF'
## 44. Old section, not checked
plain words only
## 45. New section
```
photo --> channels --> units --> logits
```
Why it matters: the build is a real network.
| Trunk check | used "new units that read only that group's units" |
EOF
cat > "$lc" <<'EOF'
### 3-33. Old card, not checked
### 3-34. A new card
Directional because: it changes which zooms the product promises.
### 3-35. A technical one (ANSWERED 2026-11-01 by delegation)
EOF
python3 "$T/ladder_lint.py" "$lg" "$lc" >/dev/null; check "ladder lint: good section and cards are clean" 0 $?
b="$TMP/l1.md"; grep -v '^```\|^photo' "$lg" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: section without a diagram fails" 1 $?
b="$TMP/l2.md"; grep -v 'Why it matters' "$lg" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: section without why-it-matters fails" 1 $?
b="$TMP/l3.md"; grep -v 'Trunk check' "$lg" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: section without trunk check fails" 1 $?
b="$TMP/c1.md"; grep -v 'Directional because' "$lc" > "$b"
python3 "$T/ladder_lint.py" "$lg" "$b" >/dev/null; check "ladder lint: card without directional line fails" 1 $?
l55="$TMP/l55.md"
cat > "$l55" <<'EOF'
## 55. A section after the meta-analysis
```
photo --> channels --> logits
```
Why it matters: the build is a real network.
| Trunk check | used "a map" |
| Cale line | nothing yet |
| Whole-network build | no |
EOF
python3 "$T/ladder_lint.py" "$l55" "$lc" >/dev/null; check "ladder lint: section 55 with both rows is clean" 0 $?
b="$TMP/l55a.md"; grep -v 'Cale line' "$l55" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: section 55 without a Cale line fails" 1 $?
b="$TMP/l55b.md"; grep -v 'Whole-network build' "$l55" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: section 55 without the build row fails" 1 $?
b="$TMP/l55c.md"; sed 's/| Whole-network build | no |/| Whole-network build | later |/' "$l55" > "$b"
python3 "$T/ladder_lint.py" "$b" "$lc" >/dev/null; check "ladder lint: build row must say yes or no" 1 $?
python3 "$T/ladder_lint.py" >/dev/null; check "ladder lint: real files are clean" 0 $?

# ---------- stop gate and reminder ----------
out="$(echo '{"stop_hook_active":false}' | bash "$T/stop_gate.sh")"; check "stop gate: runs" 0 $?
r="$(bash "$T/turn_reminder.sh" | head -1)"; check "reminder: prints in math mode" "[math mode: research/clean/curves/CLAUDE.md is the rule set]" "$r"
[ $made_marker = 1 ] && rm -f "$CURVES/.math-mode"
r="$(bash "$T/turn_reminder.sh")"
if [ $made_marker = 1 ]; then check "reminder: silent without marker" "" "$r"; fi

rm -rf "$TMP"
echo "failures: $fails"
exit $fails
