import sys; sys.argv = ["d4_depth.py", "16", "3072", "energy"]
src = open("d4_depth.py").read(); exec(src.split("out = []")[0])
for nm, I_ in (("CAL", CAL[:100]), ("EV", EV[:100])):
    x = to_block(mx.array(norm_img(np.asarray(X[I_]))), B); Oi = np.array(b1_(b0(x)) - x).reshape(-1, 768); Om = np.array(merged(x) - x).reshape(-1, 768)
    print(nm, "output kept share %.3f" % (1 - ((Om - Oi) ** 2).sum() / ((Oi - Oi.mean(0)) ** 2).sum()), "| |O| %.1f  |merged| %.1f" % (np.abs(Oi).mean(), np.abs(Om).mean()))
    hn = np.array(gelu(new1(b1_.pre(x)))).reshape(-1, len(nb1)); print("   new act mean %.3f vs CAL readings mean %.3f; max %.1f vs %.1f" % (hn.mean(), Hn.mean(), hn.max(), Hn.max()))
