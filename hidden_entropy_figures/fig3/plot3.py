import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'font.family': 'serif', 'font.serif': ['cmr10'],
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
})
C_OBS = '#D55E00'; INK2 = '#555555'
RAMP = ['#FE9A66', '#E26A1D', '#B04D00', '#7D3502']      # r = 1e-2 ... 1e-8, light -> dark
EXP = ['-2', '-4', '-6', '-8']

d = json.load(open('fig3_data.json'))
A = d['a']; x = np.array([q['x'] for q in A])
SY = np.array([q['SY'] for q in A]); lo = np.array([q['lower'] for q in A]); hi = np.array([q['upper'] for q in A])
grid = np.array(d['grid']); YMAX = 2.35

fig, axs = plt.subplots(1, 3, figsize=(7.05, 2.45), sharey=True, gridspec_kw=dict(wspace=0.14))
ax, bx, cx = axs
def tag(a, s): a.text(-0.2 if a is ax else -0.08, 1.02, s, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom')

# (a) base detector versus ln(1/r)
ax.plot(x, hi, color=INK2, lw=0.9, ls=(0, (4, 2)), label='upper bound', zorder=2)
ax.plot(x, lo, color='#222222', lw=0.9, ls=(0, (1, 1.6)), label='lower bound', zorder=4)
ax.plot(x, SY, color=C_OBS, lw=1.4, label=r'$\Sigma_Y(\mathcal{T},0)/M$', zorder=3)
ax.set_xlim(0, 20); ax.set_ylim(0, YMAX)
ax.set_xlabel(r'$\ln(1/r)$'); ax.set_ylabel(r'$\Sigma_Y(\mathcal{T},0)/M$')
ax.legend(loc='lower right', frameon=False, handlelength=2.2, borderaxespad=0.3, labelspacing=0.25)
tag(ax, '(a)')

# (b) photon loss (theta = 1)   (c) gating (eta = 1)
for k, r in enumerate(d['rs']):
    yb = np.clip(d['b'][str(r)], 0, None); yc = np.clip(d['c'][str(r)], 0, None)
    bx.plot(grid, yb, color=RAMP[k], lw=1.4, label=r'$r=10^{%s}$' % EXP[k])
    cx.plot(grid, yc, color=RAMP[k], lw=1.4)
bx.set_xlim(0, 1); cx.set_xlim(0, 1)
bx.set_xlabel(r'collection efficiency $\eta$'); cx.set_xlabel(r'live-bin probability $\vartheta$')
bx.legend(loc='upper left', frameon=False, handlelength=2.2, borderaxespad=0.3, labelspacing=0.25)
tag(bx, '(b)'); tag(cx, '(c)')

fig.savefig('fig3.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('fig3.png', dpi=230, bbox_inches='tight', pad_inches=0.02)
