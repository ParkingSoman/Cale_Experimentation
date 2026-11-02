import numpy as np, torch
src = open("c23_netmade.py").read(); exec(src.split("Xn = np.load")[0])
refs = REFS(); refs["netmade256"] = np.load("cache/c23_netmade256.npy")
K = K_from(np.load("cache/e1_self_32.npy"), rep=80); r = factor_build(K, "c29_build_netmade32"); paired("netmade32", r, refs); np.save("cache/c29_netmade32.npy", r)
