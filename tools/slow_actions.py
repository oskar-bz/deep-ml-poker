"""Our slow decisions on the platform: by street / action / situation. python tools/slow_actions.py data/r3 [thr_ms]"""
import glob, json, sys
from collections import Counter
folder = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
ME = "Oskar Behzadpour"
slow = Counter(); allc = Counter(); maxes = []; ms_all = []
for f in glob.glob(folder + "/*.json"):
    g = json.load(open(f, encoding="utf-8"))["game"]
    me = {i for i, p in enumerate(g["players"]) if p["name"].startswith(ME)}
    for s in me: maxes.append(g["seat_stats"][s]["max_ms"])
    for h in g["hands"]:
        nlive_pre = sum(1 for x in h["s"] if x > 0)
        for a in h["a"]:
            if a[1] in me and len(a) > 4 and a[4] is not None:
                st = ["pre", "flop", "turn", "river"][a[0]]
                # number of raises before on this street
                key = (st, a[2])
                allc[key] += 1; ms_all.append(a[4])
                if a[4] >= thr: slow[key] += 1
maxes.sort(); ms_all.sort()
q = lambda L, p: L[int(p * (len(L) - 1))]
print(f"{folder}: decisions {len(ms_all)} avg {sum(ms_all)/len(ms_all):.3f} p99 {q(ms_all,.99):.3f} p99.9 {q(ms_all,.999):.3f} | per-game max median {q(maxes,.5):.2f} p95 {q(maxes,.95):.2f}")
print(f"slow (>= {thr} ms) by street/action:")
for k, v in sorted(slow.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {k}: {v} of {allc[k]} ({v/allc[k]:.2%})")
