"""S4b: checks on the S4 emulation (plan's S4 fail protocol).

  python3 research/clean/liquid/s4b_check.py

S4's whole_slice left the final norm and the tied head at full width. The final norm's output gamma * x_hat leaves
span(P), and the full table E reads that part, so the scored model really had a second, untied output table.
Here the final norm's output is projected again (norm(x) @ P P^T). That is the tied design in the S4 docstring:
logits = (E P)(P^T diag(gamma_f) P)(P^T x_hat), one extra d' x d' matrix.
Rows: identity at d = 576 (old and new head), then pca_global at d 518 and 432 with the tied head.
"""
from lib import *  # noqa: F403
from s3_stream import collect, bases  # noqa: E402
from s4_stream_whole import whole_slice  # noqa: E402


class PostNorm(nn.Module):
    def __init__(self, norm, P):
        super().__init__(); self.norm = norm; self.PP = mx.array((P @ P.T).astype(np.float32))
    def __call__(self, x):
        return self.norm(x) @ self.PP


def main():
    S = Setup(); t = Timer(); acc = collect(S); B, sal = bases(S, acc, 16); print(f"bases {t()}s", flush=True); rows = []
    def rec(name, dp, edits):
        res = S.score(edits)
        row = {"step": "S4b", "arm": name, "d": dp, "bytes_frac_layers": round(1 - dp / D_MODEL, 4), **{k: boot(v) for k, v in res.items()}}
        rows.append(row); print(f"  {name:24s} d{dp}  prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}  ({t()}s)", flush=True)
    P = B["pca_global"]
    rec("identity_untied", 576, whole_slice(S, P))
    rec("identity_tied", 576, whole_slice(S, P) + [(S.model.model, "norm", PostNorm(S.model.model.norm, P))])
    for dp in (518, 432):
        Pd = P[:, :dp]
        rec("pca_global_tied", dp, whole_slice(S, Pd) + [(S.model.model, "norm", PostNorm(S.model.model.norm, Pd))])
    print("saved", save("s4b-check", rows))


if __name__ == "__main__":
    main()
