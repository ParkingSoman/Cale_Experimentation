"""CALC 3.6 Step 3 scores (plan id-like-you-to-fizzy-cherny). Hits = picked tiles that are member m2/m3 top SCORE
crops, out of 24 (chance 12). Per group and condition: mean over the two sheets (A, B), each judged by a different judge.
J1: student words (S) above the shuffled control (X) for at least 7 of 10 groups.
J2: in the groups where J1 passes, the single teacher neuron's words (T) are NOT as good as S (T < S).
Output: judge/SCORES.json
"""
import json
import os

JD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge")


def main():
    ans = json.load(open(os.path.join(JD, "ANSWER.json")))
    A = json.load(open(os.path.join(JD, "ASSIGN.json")))
    S = {}
    for j, items in A.items():
        picks = json.load(open(os.path.join(JD, f"{j}_answers.json")))
        for n, it in enumerate(items):
            p = picks[str(n + 1)]
            assert len(set(p)) == 24, (j, n, len(set(p)))
            key = it["sheet"].replace("sheet_", "").replace(".png", "")
            S.setdefault(it["group"], {}).setdefault(it["cond"], []).append(len(set(p) & set(ans[key])))
    rows, j1, j2 = [], 0, 0
    for g in sorted(S):
        m = {c: sum(v) / len(v) for c, v in S[g].items()}
        p1 = m["S"] > m["X"]
        p2 = p1 and m["T"] < m["S"]
        j1 += p1; j2 += p2
        rows.append(dict(group=g, **{c: S[g][c] for c in "STX"}, J1=p1, J2=p2))
        print(g, {c: S[g][c] for c in "STX"}, "J1" if p1 else "", "J2" if p2 else "")
    tot = {c: sum(sum(r[c]) for r in rows) / (24 * 2 * len(rows)) for c in "STX"}
    print(f"J1 groups S > X: {j1} of 10 (prediction >= 7). J2 groups S > X and T < S: {j2}. Hit rate", tot)
    json.dump(dict(rows=rows, hit_rate=tot, J1=j1, J2=j2), open(os.path.join(JD, "SCORES.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
