"""Fig. 2 data: resolved, observed and hidden entropy production per bin
versus hbar*omega_c/(k_B T_b), at fixed window M bins.

Model (rotating frame, units gamma = 1):
  H = Delta a^dag a + (i g/2)(a^dag^2 - a^2)
  L_e = sqrt(gamma (nbar+1)) a,  L_a = sqrt(gamma nbar) a^dag
Detector (emission-only, Sec. IV.A):
  E_0 = (1-r)[(1-th) e^{dL} + th e^{d(L - eta J_e)}],  E_1 = e^{dL} - E_0
Aligned reverse law (Eq. 44): q_Y(y) = p_a(y^rev), absorption counter on the
forward experiment.  Cross-checked against the explicit reverse experiment
(g -> -g, reverse stationary state).
"""
import numpy as np, scipy.linalg as sl, itertools, sys, json

Delta, g, gam, delta = 1.0, 0.6, 1.0, 1.0
eta, th, r, M = 1.0, 0.8, 0.05, 8

def build(eps, N, reverse=False):
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex); ad = a.conj().T
    gg = -g if reverse else g
    H = Delta*ad@a + 0.5j*gg*(ad@ad - a@a)
    nb = eps/(1-eps)
    Le = np.sqrt(gam*(nb+1))*a; La = np.sqrt(gam*nb)*ad
    I = np.eye(N)
    spre = lambda A: np.kron(I, A); spost = lambda B: np.kron(B.T, I)
    jump = lambda L: np.kron(L.conj(), L)
    Je, Ja = jump(Le), jump(La)
    K = Le.conj().T@Le + La.conj().T@La
    L = -1j*(spre(H)-spost(H)) + Je + Ja - 0.5*(spre(K)+spost(K))
    return L, Je, Ja, N

def steady(L, N):
    # solve L v = 0 with trace normalisation
    A = L.copy(); b = np.zeros(N*N, complex)
    tr = np.eye(N).reshape(-1, order='F')
    A[0, :] = tr; b[0] = 1.0
    v = np.linalg.solve(A, b)
    return v

def tree(maps, v0, trv):
    """probabilities of all words y (y_1 applied first), returned in lexicographic order."""
    vs = [v0]
    for _ in range(M):
        vs = [m@v for v in vs for m in maps]   # word index: y_1 most significant
    return np.array([np.real(trv@v) for v in vs])

def observed(eps, N, check_reverse=False):
    L, Je, Ja, N = build(eps, N)
    trv = np.eye(N).reshape(-1, order='F')
    rho = steady(L, N)
    P = sl.expm(L*delta)
    E0 = (1-r)*((1-th)*P + th*sl.expm((L-eta*Je)*delta)); E1 = P-E0
    A0 = (1-r)*((1-th)*P + th*sl.expm((L-eta*Ja)*delta)); A1 = P-A0
    p = tree([E0, E1], rho, trv)
    pa = tree([A0, A1], rho, trv)
    # q(y) = p_a(y^rev)
    words = list(itertools.product([0, 1], repeat=M))
    idx = {w: i for i, w in enumerate(words)}
    q = np.array([pa[idx[w[::-1]]] for w in words])
    out = dict(SY=float(np.sum(p*np.log(p/q))), pstar=float(np.real(trv@E1@rho)),
               J=float(np.real(trv@(Je-Ja)@rho)), norm=float(p.sum()),
               nss=float(np.real(np.trace(rho.reshape(N, N, order='F')@np.diag(np.arange(N))))))
    if check_reverse:
        Lr, Jer, Jar, _ = build(eps, N, reverse=True)
        rhor = steady(Lr, N); Pr = sl.expm(Lr*delta)
        B0 = (1-r)*((1-th)*Pr + th*sl.expm((Lr-eta*Jar)*delta)); B1 = Pr-B0
        pt = tree([B0, B1], rhor, trv)            # reverse-experiment absorption words
        qx = np.array([pt[idx[w[::-1]]] for w in words])
        out['rev_check'] = float(np.max(np.abs(qx-q)))
    return out

# analytic resolved quantities
fa = gam**2 + 4*Delta**2; fd = gam**2 + 4*Delta**2 - 4*g**2
J0 = 2*gam*g**2/fd
def Sfull_per_bin(x):           # x = hbar w_c / k_B T_b
    eps = np.exp(-x); nb = eps/(1-eps)
    nu = (nb+0.5)*np.sqrt(fa/fd); zeta = (nu-0.5)/(nu+0.5)
    B = 8*gam**2*g**2/(fa*fd)*nu*np.log(1/zeta)
    return J0/np.tanh(x/2)*x*delta + B/M, B
def nss_exact(x):
    eps = np.exp(-x); nb = eps/(1-eps); return nb + (2*nb+1)*2*g**2/fd

if __name__ == '__main__':
    # truncation check at the hottest point
    for N in (30, 40):
        o = observed(np.exp(-1.0), N)
        print('x=1 N=%d  SY=%.8f  nss=%.6f (exact %.6f)  J=%.6f (exact %.6f) norm=%.12f'
              % (N, o['SY'], o['nss'], nss_exact(1.0), o['J'], J0/np.tanh(0.5), o['norm']))
    o = observed(np.exp(-3.0), 30, check_reverse=True)
    print('x=3 reverse-experiment check max|q - q_explicit| =', o['rev_check'])
    o = observed(0.0, 30)
    db = lambda p, q: p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    print('vacuum: SY/M=%.6f  p*=%.6f  lower=%.6f upper=%.6f uniform=%.6f'
          % (o['SY']/M, o['pstar'], db(o['pstar'], r), db(r+(1-r)*th, r), np.log(1/min(r, (1-r)*(1-th)))))
    print('B0 =', Sfull_per_bin(np.inf)[1] if False else Sfull_per_bin(60.0)[1])
