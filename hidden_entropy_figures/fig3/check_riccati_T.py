import numpy as np, scipy.linalg as sl
from scipy.integrate import solve_ivp
from model import build, steady, Delta, g, gam, delta
fa=gam**2+4*Delta**2; fd=fa-4*g**2
for x in (1.0, 2.5):
    eps=np.exp(-x); nb=eps/(1-eps); c=2*nb+1
    N=34; L,Je,Ja,_=build(eps,N); rho=steady(L,N); trv=np.eye(N).reshape(-1,order='F')
    Xss=(c/(2*fd))*np.array([[fa+2*gam*g,-4*Delta*g],[-4*Delta*g,fa-2*gam*g]])
    A=np.array([[g-gam/2,Delta],[-Delta,-g-gam/2]]); D=gam*(nb+0.5)*np.eye(2)
    for eta in (0.5,1.0):
        for kind in ('e','a'):
            J = Je if kind=='e' else Ja
            fock=np.real(trv@sl.expm((L-eta*J)*delta)@rho)
            kap = eta*gam*(nb+1) if kind=='e' else eta*gam*nb
            sgn = -0.5 if kind=='e' else 0.5
            def f(t,y):
                X=y[:4].reshape(2,2); Y=X+sgn*np.eye(2)
                dX=A@X+X@A.T+D-kap*Y@Y
                return np.concatenate([dX.ravel(),[-kap*(np.trace(X)+2*sgn)/2]])
            sol=solve_ivp(f,(0,delta),np.concatenate([Xss.ravel(),[0.]]),rtol=1e-12,atol=1e-14)
            print('x=%.1f eta=%.1f %s  Fock=%.10f  Riccati=%.10f  diff=%.1e'%(x,eta,kind,fock,np.exp(sol.y[4,-1]),abs(fock-np.exp(sol.y[4,-1]))))
