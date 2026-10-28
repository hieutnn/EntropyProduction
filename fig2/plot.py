import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compute import Sfull_per_bin, J0, M, delta

plt.rcParams.update({
    'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'font.family': 'serif', 'font.serif': ['cmr10'],
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
})
C_FULL, C_HID, C_OBS = '#009E73', '#0072B2', '#D55E00'
BAND = '#E9E9E9'; INK2 = '#555555'

D = json.load(open('fig2_data.json')); rows = D['rows']; vac = D['vac']
x = np.array([q['x'] for q in rows]); SY = np.array([q['SY'] for q in rows])
SH = np.array([q['Shid'] for q in rows])
xf = np.linspace(0.6, 37, 400); SF = np.array([Sfull_per_bin(t)[0] for t in xf])
r, th = D['params']['r'], D['params']['theta']
db = lambda p, q: p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
lo, hi, uni = db(vac['pstar'], r), db(r+(1-r)*th, r), np.log(1/min(r, (1-r)*(1-th)))
B0M = vac['B0']/M; lim_hid = B0M - vac['SY']

fig, axs = plt.subplots(1, 3, figsize=(7.05, 2.45), gridspec_kw=dict(wspace=0.36))
ax, bx, cx = axs
def tag(a, s): a.text(-0.2, 1.02, s, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom')
def bands(a):
    a.axvspan(3, 14, color=BAND, lw=0, zorder=0)
    a.axvline(31, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=0)

# ---------------- (a) all three, log scale ----------------
ax.plot(xf, SF, color=C_FULL, lw=2.4, solid_capstyle='round', label=r'$\Sigma_{\rm full}$ (exact)', zorder=3)
ax.plot(x, SH, color=C_HID, lw=0.9, marker='s', ms=2.8, mew=0, label=r'$\Sigma_{\rm hid}$', zorder=4)
ax.plot(x, SY, color=C_OBS, lw=1.1, marker='o', ms=3.0, mew=0, label=r'$\Sigma_Y$', zorder=4)
ax.set_yscale('log'); ax.set_xlim(0, 37); ax.set_ylim(1e-3, 40)
ax.set_xlabel(r'$\hbar\omega_c/k_BT_b$'); ax.set_ylabel(r'entropy production per bin')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.04), frameon=True, framealpha=1, edgecolor='none',
          handlelength=2.0, borderpad=0.3, labelspacing=0.25)
tag(ax, '(a)')

# ---------------- (b) remainder after the Clausius term ----------------
bx.plot(xf, SF - J0*delta*xf, color=C_FULL, lw=2.4, label=r'$\Sigma_{\rm full}$', zorder=3)
bx.plot(x, SH - J0*delta*x, color=C_HID, lw=0.9, marker='s', ms=2.8, mew=0, label=r'$\Sigma_{\rm hid}$', zorder=4)
bx.axhline(B0M, color=C_FULL, lw=0.7, ls=(0, (4, 2)), zorder=2)
bx.axhline(lim_hid, color=C_HID, lw=0.7, ls=(0, (4, 2)), zorder=2)
xm = 22; i = np.argmin(abs(x-xm))
bx.annotate('', xy=(xm, SH[i]-J0*delta*x[i]), xytext=(xm, Sfull_per_bin(xm)[0]-J0*delta*xm),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color=C_OBS, shrinkA=0.5, shrinkB=0.5))
bx.text(xm+0.7, 0.5*(B0M+lim_hid), r'$\Sigma_Y/M$', color=INK2, fontsize=6.8, va='center')
bx.set_xlim(0, 37); bx.set_ylim(-0.12, 0.36)
bx.set_xlabel(r'$\hbar\omega_c/k_BT_b$')
bx.set_ylabel(r'$(\Sigma - \mathcal{T}\dot Q_0/k_BT_b)/M$')
bx.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='none', handlelength=2.0,
          borderpad=0.3, labelspacing=0.25)
tag(bx, '(b)')

# ---------------- (c) observed fraction ----------------
T = 1/x; frac = SY/np.array([Sfull_per_bin(t)[0] for t in x])
Tg = np.concatenate([[0], T]); fg = np.concatenate([[0], frac]); o = np.argsort(Tg)
Ts = np.linspace(0, 0.5, 50)
cx.plot(Ts, vac['SY']*Ts/(J0*delta), color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=2,
        label=r'$\Sigma_Y(\mathcal{T},0)\,k_BT_b/\mathcal{T}\dot Q_0$')
cx.plot(Tg[o], fg[o], color=C_OBS, lw=1.1, marker='o', ms=3.0, mew=0, zorder=3,
        label=r'$\Sigma_Y/\Sigma_{\rm full}$')
cx.set_xlim(0, 0.5); cx.set_ylim(0, 0.22)
cx.set_xlabel(r'$k_BT_b/\hbar\omega_c$'); cx.set_ylabel(r'observed fraction')
cx.legend(loc='upper left', frameon=False, handlelength=2.0, borderaxespad=0.3, labelspacing=0.25)
tag(cx, '(c)')

fig.savefig('fig2.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('fig2.png', dpi=230, bbox_inches='tight', pad_inches=0.02)
print('B0/M', B0M, 'limit hid', lim_hid, 'lo hi uni', lo, hi, uni)
