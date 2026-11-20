"""Endgame: when we are in the final 2 / final 3, how do we convert? Compare with mw."""
import glob, json, sys
from collections import defaultdict
folder = sys.argv[1] if len(sys.argv) > 1 else "data/r2"
WHO = ["Oskar Behzadpour", "mw", "Daksh Malhan"]
HU = defaultdict(lambda: [0, 0.0, 0, 0.0, 0])  # (who, opp) -> [n, our pts, reached by cap, chip share at HU start, won]
TH = defaultdict(lambda: [0, 0.0])           # who -> [reached final 3, points]
for f in sorted(glob.glob(folder + "/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    hu_start = None; three_start = None
    for h in g["hands"]:
        alive = [i for i in range(8) if h["s"][i] > 0]
        if len(alive) == 3 and three_start is None: three_start = (h, alive)
        if len(alive) == 2 and hu_start is None: hu_start = (h, alive)
    if three_start:
        h, al = three_start
        for i in al:
            if nm[i] in WHO and not g["players"][i]["copy"]:
                TH[nm[i]][0] += 1; TH[nm[i]][1] += g["points"][i]
    if hu_start:
        h, al = hu_start
        for i in al:
            if nm[i] in WHO and not g["players"][i]["copy"]:
                o = [x for x in al if x != i][0]
                k = (nm[i], nm[o] if nm[o] in WHO else "other")
                st = HU[k]; st[0] += 1; st[1] += g["points"][i]
                st[2] += g["stacks"][o] > 0          # still 2 alive at the end -> cap reached
                st[3] += h["s"][i] / (h["s"][i] + h["s"][o]); st[4] += g["places"][i] == 1
print("FINAL 3 reached: avg points")
for k, v in TH.items(): print(f"  {k:<18} n {v[0]:4d}  avg pts {v[1]/v[0]:.2f}")
print("HEADS-UP: n, avg pts, won outright/at cap, avg chip share at HU start, cap reached")
for k, v in sorted(HU.items()):
    n = v[0]
    print(f"  {k[0][:12]:<13} vs {k[1][:12]:<13} n {n:4d}  pts {v[1]/n:5.2f}  1st {v[4]/n:4.0%}  start share {v[3]/n:4.0%}  cap {v[2]/n:4.0%}")
