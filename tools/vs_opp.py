"""Hands that ended heads-up between us and one opponent: where do the chips go? python tools/vs_opp.py data/r2 mw"""
import glob, json, sys
from collections import defaultdict
ME = "Oskar Behzadpour"; folder, OPP = sys.argv[1], sys.argv[2]
C = defaultdict(lambda: [0, 0.0])
for f in sorted(glob.glob(folder + "/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        acts = [a for a in h["a"] if a[2] != "b"]
        vol = {a[1] for a in acts if a[2] != "F"}
        for me in [i for i, n in enumerate(nm) if n == ME]:
            for o in [i for i, n in enumerate(nm) if n == OPP]:
                if me not in vol or o not in vol or len(vol) != 2:
                    continue
                pre = [a for a in acts if a[0] == 0]
                first_r = next((a[1] for a in pre if a[2] in "RA"), None)
                nraise = sum(a[2] in "RA" for a in pre)
                who = "we-open" if first_r == me else ("they-open" if first_r == o else "limp/none")
                maxst = max(a[0] for a in acts)
                allin = any(a[2] == "A" for a in acts)
                end = "pre" if maxst == 0 else ("allin" if allin else ("sd" if len(h["board"]) == 5 and not any(a[2] == "F" and a[0] == maxst for a in acts) else "post-fold"))
                key = (who, f"raises={min(nraise,3)}", end)
                C[key][0] += 1; C[key][1] += h["d"][me] / bb
tot = sum(v[1] for v in C.values())
print(f"total {tot:+.0f} bb over {sum(v[0] for v in C.values())} pots")
for k, v in sorted(C.items(), key=lambda kv: kv[1][1]):
    print(f"  {str(k):<45}{v[0]:5d} pots {v[1]:+8.0f} bb  {v[1]/v[0]:+6.2f}/pot")
