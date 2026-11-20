"""Our 3-bet jams over a single open (8-25bb): real outcome per opener, and openers' fold-to-rejam overall."""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models import top_of
ME = "Oskar Behzadpour"
OURS = defaultdict(lambda: [0, 0, 0.0, 0.0])     # opener -> [n, opener folded, bb result, sum our top%]
ALL = defaultdict(lambda: [0, 0])                 # opener -> [faced any rejam 8-25bb, folded]
for f in sorted(glob.glob("data/r2/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        op = None
        for i, a in enumerate(pre):
            if op is None:
                if a[2] == "R": op = a
                elif a[2] == "A": break
                continue
            if a[2] == "C": break
            if a[2] == "R": break
            if a[2] == "A":
                j = a[1]
                eff = min(h["s"][j], h["s"][op[1]]) / bb
                if not (8 <= eff <= 25): break
                resp = next((x for x in pre[i+1:] if x[1] == op[1]), None)
                if resp is None: break
                ALL[nm[op[1]]][0] += 1; ALL[nm[op[1]]][1] += resp[2] == "F"
                if nm[j] == ME:
                    o = OURS[nm[op[1]]]; o[0] += 1; o[1] += resp[2] == "F"; o[2] += h["d"][j] / bb; o[3] += top_of(h["hole"][j])
                break
tot = [0, 0, 0.0]
print(f"{'opener':<18}{'our rejams':>10}{'fold':>7}{'bb/rejam':>10}{'our hand':>10}   {'all rejams fold%':>16}")
for n, v in sorted(OURS.items(), key=lambda kv: -kv[1][0]):
    a = ALL[n]
    print(f"{n[:17]:<18}{v[0]:10d}{v[1]/v[0]:7.0%}{v[2]/v[0]:+10.2f}{v[3]/v[0]:10.2f}   {a[1]/max(1,a[0]):10.0%} (n{a[0]})")
    tot[0] += v[0]; tot[1] += v[1]; tot[2] += v[2]
print("TOTAL", tot[0], "fold %.0f%%" % (100 * tot[1] / tot[0]), "bb/rejam %+.2f" % (tot[2] / tot[0]), "total %+.0f bb" % tot[2])
