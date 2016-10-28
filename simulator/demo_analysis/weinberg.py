import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import jsonlines
import math
import numpy as np
costhetas = []
for i,e in enumerate(jsonlines.Reader(open(sys.argv[1]))):
    els = [p for p in e['particles'] if p['id'] == 11]
    mus = [p for p in e['particles'] if p['id'] == 13]
    assert len(mus) == 1
    assert len(els) == 1
    mu = mus[0]
    el = els[0]
    el_px, el_py, el_pz = [el[x] for x in ['px','py','pz']]
    mu_px, mu_py, mu_pz = [mu[x] for x in ['px','py','pz']]
    costheta = mu_pz/el_pz
    costhetas.append(costheta)

plt.hist(costhetas, bins = 100, histtype='stepfilled')
plt.savefig(sys.argv[2])
