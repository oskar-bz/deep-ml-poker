"""Per bot: how often they reach the last 4 / 3 / 2 and their place given that, plus chip share at each stage."""
import glob, json, sys
from collections import defaultdict, Counter
folder = sys.argv[1] if len(sys.argv) > 1 else "data/r4"
P = defaultdict(Counter); CH = defaultdict(lambda: defaultdict(list)); N = Counter()
for f in sorted(glob.glob(folder + "/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    seen = set()
    for h in g["hands"]:
        alive = [i for i, x in enumerate(h["s"]) if x > 0]
        k = len(alive)
        if k <= 4 and k not in seen:
            seen.add(k); tot = sum(h["s"])
            for i in alive:
                P[nm[i]][f"r{k}"] += 1; CH[nm[i]][k].append(h["s"][i] / tot * k)
    for i, n in enumerate(nm):
        N[n] += 1; P[n][f"p{g['places'][i]}"] += 1
print(f"{'bot':<17}{'reach4':>7}{'reach3':>7}{'reach2':>7}{'win':>6} | {'4->3':>6}{'3->2':>6}{'2->1':>6} | chip share x k at 4/3/2 (1.0 = average)")
for n in sorted(N, key=lambda n: -(10*P[n]['p1']+5*P[n]['p2']+2*P[n]['p3'])/N[n]):
    p = P[n]; t = N[n]
    r4, r3, r2 = p['r4'] or 1, p['r3'] or 1, p['r2'] or 1
    cs = " ".join(f"{sum(CH[n][k])/max(1,len(CH[n][k])):.2f}" for k in (4, 3, 2))
    print(f"{n[:16]:<17}{p['r4']/t:7.0%}{p['r3']/t:7.0%}{p['r2']/t:7.0%}{p['p1']/t:6.0%} | {p['r3']/r4:6.0%}{p['r2']/r3:6.0%}{p['p1']/r2:6.0%} | {cs}")
