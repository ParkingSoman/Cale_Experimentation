# serves: T3
"""Symbolic checks for Lemma T3.0 and Theorem T3 (COARSE-THEORY.md). Prints each residual; all must be exactly 0.

1. Lemma T3.0: on a circle traversed counter-clockwise, the tangent angles at two points satisfy
   phi_1 + phi_2 = 2 alpha (alpha = chord angle).
2. Theorem T3: Wiener-integral variances and covariances of (phi(s), y(s)) for the three linearised laws, and the
   pairing slope beta(s) = s Cov(phi, y) / Var(y).
"""
import sympy as sp

a, b, R, s, w, tau, sig = sp.symbols("a b R s w tau sigma", positive=True)
out = []

# 1. circle of radius R centred at the origin, points at angles a and b; tangent angle = point angle + pi/2
p1 = sp.Matrix([R * sp.cos(a), R * sp.sin(a)])
p2 = sp.Matrix([R * sp.cos(b), R * sp.sin(b)])
phi1, phi2 = a + sp.pi / 2, b + sp.pi / 2
chord = p2 - p1
# chord direction angle alpha satisfies tan(alpha) = dy/dx; check via the unit vector (cos, sin) of (a+b)/2 + pi/2
alpha = (a + b) / 2 + sp.pi / 2
cross = sp.simplify(chord[0] * sp.sin(alpha) - chord[1] * sp.cos(alpha))  # 0 iff chord is parallel to alpha
out.append(("chord parallel to alpha", cross))
out.append(("phi1 + phi2 - 2 alpha (mod 2 pi)", sp.simplify(sp.Mod(phi1 + phi2 - 2 * alpha, 2 * sp.pi))))

# 2. Wiener integrals: kernels in w on [0, s]
def var(k):
    return sp.integrate(k ** 2, (w, 0, s))

def cov(k1, k2):
    return sp.integrate(k1 * k2, (w, 0, s))

# curvature process: phi = kappa0 s + sigma int (s - w) dB ; y = kappa0 s^2/2 + sigma int (s - w)^2/2 dB
kphi, ky = sig * (s - w), sig * (s - w) ** 2 / 2
Vy = tau ** 2 * s ** 4 / 4 + var(ky)
Cpy = tau ** 2 * s ** 3 / 2 + cov(kphi, ky)
out.append(("curvature: Var y - (tau^2 s^4/4 + sigma^2 s^5/20)", sp.simplify(Vy - (tau ** 2 * s ** 4 / 4 + sig ** 2 * s ** 5 / 20))))
out.append(("curvature: Cov - (tau^2 s^3/2 + sigma^2 s^4/8)", sp.simplify(Cpy - (tau ** 2 * s ** 3 / 2 + sig ** 2 * s ** 4 / 8))))
rho = sig ** 2 / tau ** 2
beta_c = sp.simplify(s * Cpy / Vy)
out.append(("curvature: beta - 2(1+rho s/4)/(1+rho s/5)", sp.simplify(beta_c - 2 * (1 + rho * s / 4) / (1 + rho * s / 5))))
out.append(("curvature: beta(s->0) - 2", sp.limit(beta_c, s, 0) - 2))
out.append(("curvature: beta(s->oo) - 5/2", sp.limit(beta_c, s, sp.oo) - sp.Rational(5, 2)))
out.append(("circle (sigma=0): beta - 2", sp.simplify(beta_c.subs(sig, 0) - 2)))
# direction process: phi = sigma B(s) = sigma int 1 dB ; y = sigma int (s - w) dB
Vy_d, Cpy_d = var(sig * (s - w)), cov(sig, sig * (s - w))
out.append(("direction: beta - 3/2", sp.simplify(s * Cpy_d / Vy_d - sp.Rational(3, 2))))

# 3. sharpness, any curvature law: phi - 2 y / s has no kappa0 term; its Brownian kernel is (s-w) - (s-w)^2/s
eps_kernel = sp.simplify(kphi - 2 * ky / s)
out.append(("curvature: kernel of phi - 2y/s - sigma (s-w) w / s", sp.simplify(eps_kernel - sig * (s - w) * w / s)))
out.append(("curvature: Var(phi - 2y/s) - sigma^2 s^3/30", sp.simplify(var(eps_kernel) - sig ** 2 * s ** 3 / 30)))
# direction process: regression residual variance Var phi - Cov^2 / Var y
res_d = sp.simplify(var(sig + 0 * w) - cov(sig, sig * (s - w)) ** 2 / Vy_d)
out.append(("direction: residual variance - sigma^2 s / 4", sp.simplify(res_d - sig ** 2 * s / 4)))

# 4. Theorem T3 part 1 ratio and part 2 conditional variance, part 3 extra statistic
rho_ = sp.symbols("rho", positive=True)
Vphi = tau ** 2 * s ** 2 + var(kphi)
out.append(("T3.1 ratio Var(eps)/Var(phi) - rho s/(10(rho s+3))", sp.simplify((var(eps_kernel) / Vphi).subs(sig, sp.sqrt(rho_) * tau) - rho_ * s / (10 * (rho_ * s + 3)))))
condvar = sp.simplify(Vphi - Cpy ** 2 / Vy)
out.append(("T3.2 Var(phi|y) - sigma^2 s^3 (rho s+8)/(48(rho s+5))", sp.simplify((condvar - sig ** 2 * s ** 3 * (rho_ * s + 8) / (48 * (rho_ * s + 5))).subs(sig, sp.sqrt(rho_) * tau))))
out.append(("Cov(phi - 2y/s, y) - sigma^2 s^4/40", sp.simplify(cov(eps_kernel, ky) - sig ** 2 * s ** 4 / 40)))
out.append(("direction: Var(phi - 2y/s) - sigma^2 s/3", sp.simplify(var(sig - 2 * sig * (s - w) / s) - sig ** 2 * s / 3)))

# 5. Corollary T3.2 with four explicit independent errors: statistic phi' - 2 y'/s,
#    y' = y + d1 - d0 - s e0, phi' = phi + e1 - e0. The statistic is linear in (curve part, d0, d1, e0, e1),
#    so its variance is the curve part plus the sum of squared coefficients times the error variances.
wy, wp = sp.symbols("w_y w_phi", positive=True)
d0, d1, e0, e1 = sp.symbols("d0 d1 e0 e1")
err = (e1 - e0) - 2 * (d1 - d0 - s * e0) / s
coef = [sp.expand(err).coeff(v) for v in (d0, d1, e0, e1)]
width_var = sum(c ** 2 * sd ** 2 for c, sd in zip(coef, (wy, wy, wp, wp)))
out.append(("T3.2 width terms - (2 w_phi^2 + 8 w_y^2/s^2)", sp.simplify(width_var - (2 * wp ** 2 + 8 * wy ** 2 / s ** 2))))
out.append(("T3.2 curvature D(s)", sp.simplify(var(eps_kernel) + width_var - (sig ** 2 * s ** 3 / 30 + 2 * wp ** 2 + 8 * wy ** 2 / s ** 2))))
out.append(("T3.2 direction D(s)", sp.simplify(var(sig - 2 * sig * (s - w) / s) + width_var - (sig ** 2 * s / 3 + 2 * wp ** 2 + 8 * wy ** 2 / s ** 2))))

# 6. Theorem T3 part 4: curvature fixed at both ends (condition on B(s)); B(s) = int 1 dB
cov_eB = cov(eps_kernel, sig + 0 * w) / sig          # Cov(eps, B(s)) with B(s) kernel 1
out.append(("T3 part 4: Cov(eps, B(s)) - sigma s^2/6", sp.simplify(cov_eB - sig * s ** 2 / 6)))
out.append(("T3 part 4: conditional variance - sigma^2 s^3/180", sp.simplify(var(eps_kernel) - cov_eB ** 2 / s - sig ** 2 * s ** 3 / 180)))

for name, r in out:
    print(f"{name:55s} residual = {r}")
bad = [n for n, r in out if sp.simplify(r) != 0]
print("ALL ZERO" if not bad else f"NONZERO: {bad}")
