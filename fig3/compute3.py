"""Fig. 3 data: observed entropy production at vacuum versus the detector parameters.

At T_b = 0 the aligned reverse word is pure dark counts, q_Y = Bern(r)^{(x)M}, so
Sigma_Y(T,0) = D_KL(p_Y || Bern(r)^{(x)M}).  p_Y is enumerated exactly over all 2^M words
with the Fock-space maps of model.py.  The single-bin flag probability p* is checked
against an independent Gaussian (Riccati) evaluation of the generating function G_delta.
"""
import numpy as np, scipy.linalg as sl, json
from scipy.integrate import solve_ivp
from model import build, steady, Delta, g, gam, delta

M = 8
N = 22
L, Je, Ja, _ = build(0.0, N)
trv = np.eye(N).reshape(-1, order='F')
rho = steady(L, N)
P = sl.expm(L*delta)
_cache = {}
def veto(eta):                      # e^{delta (L - eta J_e)}
    k = round(eta, 12)
    if k not in _cache: _cache[k] = sl.expm((L - eta*Je)*delta)
    return _cache[k]

def word_probs(E0, E1):
    vs = [rho]
    for _ in range(M):
        vs = [m@v for v in vs for m in (E0, E1)]       # y_1 most significant
    return np.array([np.real(trv@v) for v in vs])

ones = np.array([bin(i).count('1') for i in range(2**M)])
def sigma_Y(eta, th, r):
    """per-bin observed entropy production at vacuum, and p*"""
    F0 = (1-th)*P + th*veto(eta)
    E0 = (1-r)*F0; E1 = P - E0
    p = word_probs(E0, E1)
    p = np.clip(p, 1e-300, None)
    logq = ones*np.log(r) + (M-ones)*np.log1p(-r)
    S = float(np.sum(p*(np.log(p) - logq)))/M
    pstar = float(np.real(trv@E1@rho))
    return S, pstar

def dB(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

# ---------- independent check: Gaussian generating function G_delta(1-eta) ----------
fa = gam**2 + 4*Delta**2; fd = fa - 4*g**2
Xss = (1/(2*fd))*np.array([[fa+2*gam*g, -4*Delta*g], [-4*Delta*g, fa-2*gam*g]])
A = np.array([[g-gam/2, Delta], [-Delta, -g-gam/2]]); D = (gam/2)*np.eye(2)
def G_riccati(eta):
    """G = tr[e^{delta(L - eta J_e)} rho_ss] from the Riccati flow of the normalised covariance:
       dX/dt = AX + XA^T + D - eta*gam*(X - I/2)^2,  d ln Z/dt = -eta*gam*(tr X - 1)/2."""
    def f(t, y):
        X = y[:4].reshape(2, 2); Y = X - 0.5*np.eye(2)
        dX = A@X + X@A.T + D - eta*gam*Y@Y
        return np.concatenate([dX.ravel(), [-eta*gam*(np.trace(X)-1)/2]])
    sol = solve_ivp(f, (0, delta), np.concatenate([Xss.ravel(), [0.0]]), rtol=1e-12, atol=1e-14)
    return float(np.exp(sol.y[4, -1]))

if __name__ == '__main__':
    print('check p* (Fock) against Gaussian G_delta:')
    for eta in (0.25, 0.5, 1.0):
        for th in (0.8, 1.0):
            r = 0.05
            _, ps = sigma_Y(eta, th, r)
            ps_g = r + (1-r)*th*(1 - G_riccati(eta))
            print('  eta=%.2f th=%.1f  p*_Fock=%.10f  p*_Gauss=%.10f  diff=%.1e' % (eta, th, ps, ps_g, abs(ps-ps_g)))
    S, ps = sigma_Y(1.0, 0.8, 0.05)
    print('base detector r=0.05: Sigma_Y/M=%.6f (Fig. 2 plateau 0.083071)' % S)

    out = {}
    # (a) base detector eta=1, theta=0.8, versus ln(1/r)
    xs = np.concatenate([np.linspace(0.02, 1, 12), np.linspace(1.25, 20, 76)])
    rowsA = []
    for x in xs:
        r = np.exp(-x); S, ps = sigma_Y(1.0, 0.8, r)
        rowsA.append(dict(x=float(x), r=float(r), SY=S, lower=float(dB(ps, r)), upper=float(dB(r+(1-r)*0.8, r)), pstar=ps))
    out['a'] = rowsA
    p0 = 0.8*(1 - G_riccati(1.0))
    out['pstar0_base'] = p0
    # (b) photon loss: eta in [0,1], theta = 1 ; (c) gating: theta in [0,1], eta = 1
    rs = [1e-2, 1e-4, 1e-6, 1e-8]
    grid = np.linspace(0, 1, 41)
    out['rs'] = rs; out['grid'] = grid.tolist()
    out['b'] = {str(r): [sigma_Y(e, 1.0, r)[0] for e in grid] for r in rs}
    out['c'] = {str(r): [sigma_Y(1.0, t, r)[0] for t in grid] for r in rs}
    json.dump(out, open('fig3_data.json', 'w'), indent=1)
    # asymptotic slope check for the base detector
    a = rowsA; x1, x2 = a[-11]['x'], a[-1]['x']
    slope = (a[-1]['SY'] - a[-11]['SY'])/(x2 - x1)
    print('large-ln(1/r) slope of Sigma_Y/M = %.5f ; p*_0 = theta(1-G) = %.5f' % (slope, p0))
    print('intercept c in Sigma_Y/M ~ p*_0 ln(1/r) - c :', p0*x2 - a[-1]['SY'])
    for r in rs: print('r=%g  b(eta=1)=%.5f  c(theta=1)=%.5f  b(eta=0)=%.2e c(th=0)=%.2e'
                       % (r, out['b'][str(r)][-1], out['c'][str(r)][-1], out['b'][str(r)][0], out['c'][str(r)][0]))
