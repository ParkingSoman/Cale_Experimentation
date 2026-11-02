# serves: T1
"""Symbolic checks for Theorem T1 parts 3-4 (COARSE-THEORY.md). Prints each residual; all must be exactly 0.

1. Circulant matrices on Z_N have the Fourier modes as eigenvectors (N = 10, symbolic symmetric first row).
2. The three bumps 1 + cos(t - phi_j), phi_j = 2 pi j / 3, span {1, cos t, sin t} (explicit identities).
3. The 2d + 1 shifted Fejer kernels of degree d span the trigonometric polynomials of degree <= d (d = 1, 2):
   their coefficient matrix in the basis {1, cos nt, sin nt} has full rank.
"""
import sympy as sp

t = sp.symbols("t", real=True)
out = []

# 1. circulant eigenvectors, N = 10, symmetric first row c0..c5
N = 10
c = sp.symbols("c0:6", real=True)
row = [c[min(j, N - j)] for j in range(N)]
C = sp.Matrix(N, N, lambda i, j: row[(j - i) % N])
for n in range(N // 2 + 1):
    for f in (sp.cos, sp.sin):
        v = sp.Matrix([f(2 * sp.pi * n * i / N) for i in range(N)])
        if all(sp.simplify(x) == 0 for x in v):
            continue
        lam = sum(row[j] * sp.cos(2 * sp.pi * n * j / N) for j in range(N))
        res = C * v - lam * v
        # exact: each entry is linear in c0..c5; every coefficient must simplify to 0
        coeffs = [sp.radsimp(sp.simplify(sp.expand(x).coeff(ci))) for x in res for ci in c]
        exact = sum(abs(sp.N(q, 50)) for q in coeffs)
        out.append((f"circulant mode n={n} {f.__name__} (sum |coeff|, 50 digits)", sp.Integer(0) if exact < 1e-40 else exact))

# 2. bumps
phis = [2 * sp.pi * j / 3 for j in range(3)]
bumps = [1 + sp.cos(t - p) for p in phis]
out.append(("sum of bumps - 3", sp.simplify(sum(bumps) - 3)))
out.append(("sum cos(phi_j) bump_j - 3/2 cos t", sp.simplify(sum(sp.cos(p) * b for p, b in zip(phis, bumps)) - sp.Rational(3, 2) * sp.cos(t))))
out.append(("sum sin(phi_j) bump_j - 3/2 sin t", sp.simplify(sum(sp.sin(p) * b for p, b in zip(phis, bumps)) - sp.Rational(3, 2) * sp.sin(t))))

# 3. Fejer kernels: F_d(t) = sum_{|n|<=d} (1 - |n|/(d+1)) e^{int} = 1 + 2 sum_{n=1}^d (1 - n/(d+1)) cos nt
for d in (1, 2):
    def fejer(x):
        return 1 + 2 * sum((1 - sp.Rational(n, d + 1)) * sp.cos(n * x) for n in range(1, d + 1))
    shifts = [2 * sp.pi * j / (2 * d + 1) for j in range(2 * d + 1)]
    basis = [sp.Integer(1)] + [sp.cos(n * t) for n in range(1, d + 1)] + [sp.sin(n * t) for n in range(1, d + 1)]
    rows = []
    for s in shifts:
        e = sp.expand(sp.expand_trig(fejer(t - s)))
        # coefficients by projection onto the orthogonal basis over one period
        coeffs = [sp.simplify(sp.integrate(e * b, (t, 0, 2 * sp.pi)) / sp.integrate(b * b, (t, 0, 2 * sp.pi))) for b in basis]
        rows.append(coeffs)
    rank = sp.Matrix(rows).rank(simplify=True)
    out.append((f"Fejer d={d}: 2d+1 - rank of shifted kernels", (2 * d + 1) - rank))
    # non-negativity: F_d(t) = (1/(d+1)) (sin((d+1)t/2)/sin(t/2))^2
    closed = sp.sin((d + 1) * t / 2) ** 2 / ((d + 1) * sp.sin(t / 2) ** 2)
    out.append((f"Fejer d={d}: kernel - non-negative closed form", sp.simplify(sp.expand_trig(fejer(t) - closed).rewrite(sp.cos))))

for name, r in out:
    print(f"{name:55s} residual = {r}")
bad = [n for n, r in out if sp.simplify(r) != 0]
print("ALL ZERO" if not bad else f"NONZERO: {bad}")
