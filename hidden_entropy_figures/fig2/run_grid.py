import numpy as np, json, time
from compute import observed, Sfull_per_bin, M, r, th
xs = [0.75,1,1.25,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12,14,17,20,24,28,32,36]
rows = []
for x in xs:
    N = 34 if x < 1.5 else (28 if x < 3 else 22)
    t = time.time(); o = observed(np.exp(-x), N)
    sf, B = Sfull_per_bin(x)
    rows.append(dict(x=x, SY=o['SY']/M, Sfull=sf, Shid=sf-o['SY']/M, B=B, pstar=o['pstar'], N=N))
    print('x=%5.2f N=%d SY/M=%.6f Sfull/M=%.5f Shid/M=%.5f (%.1fs)' % (x, N, o['SY']/M, sf, sf-o['SY']/M, time.time()-t), flush=True)
o = observed(0.0, 22)
vac = dict(SY=o['SY']/M, pstar=o['pstar'], B0=Sfull_per_bin(60.0)[1])
# truncation check at two points
for x, Na, Nb in [(0.75, 34, 44), (3, 22, 30)]:
    print('trunc x=%g: %.3e' % (x, abs(observed(np.exp(-x), Na)['SY']-observed(np.exp(-x), Nb)['SY'])/M))
json.dump(dict(rows=rows, vac=vac, params=dict(Delta=1, g=0.6, gamma=1, delta=1, eta=1, theta=th, r=r, M=M)),
          open('fig2_data.json', 'w'), indent=1)
print('vac', vac)
