"""Openers facing a 3-bet jam: fold rate and called-hand strength, by opener and depth of the jam."""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models import top_of
T = defaultdict(lambda: [0, 0, []])
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        opener = None; jam = None
        for i, a in enumerate(pre):
            if opener is None and a[2] == "R":
                opener = a; continue
            if opener is not None and jam is None and a[2] in "RA":
                if a[2] == "A":
                    jam = a
                    resp = next((x for x in pre[i + 1:] if x[1] == opener[1]), None)
                    if resp is None: break
                    depth = min(h["s"][a[1]], h["s"][opener[1]]) / bb
                    db = "<12" if depth < 12 else ("12-25" if depth < 25 else "25+")
                    for key in ((nm[opener[1]], db), ("ALL", db)):
                        T[key][0] += 1
                        if resp[2] == "F": T[key][1] += 1
                        else: T[key][2].append(top_of(h["hole"][opener[1]]))
                break
print(f"{'opener':<18}" + "".join(f"{d:>22}" for d in ("<12", "12-25", "25+")))
names = sorted({k[0] for k in T}, key=lambda n: -sum(T[(n, d)][0] for d in ("<12", "12-25", "25+")))
for n in names[:20]:
    row = ""
    for d in ("<12", "12-25", "25+"):
        t, fo, L = T[(n, d)]
        row += f"   fold {fo/max(1,t):4.0%} n{t:4d}     " if t else f"{'-':>22}"
    print(f"{n[:17]:<18}{row}")
