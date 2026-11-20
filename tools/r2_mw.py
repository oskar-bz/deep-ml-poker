import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models import top_of
folder = sys.argv[1]; WHO = sys.argv[2]
r = defaultdict(lambda: [0, 0]); tb = defaultdict(list)
for f in sorted(glob.glob(folder + "/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        raises = []
        for i, a in enumerate(pre):
            if a[2] in "RA":
                raises.append(a)
                if len(raises) == 2 and nm[a[1]] == WHO:
                    eff = min(h["s"][a[1]], h["s"][raises[0][1]]) / bb
                    d = "<25" if eff < 25 else "25+"
                    tb[(d, a[2])].append(top_of(h["hole"][a[1]]))
                if len(raises) == 3 and nm[raises[1][1]] == WHO:
                    resp = next((x for x in pre[i+1:] if x[1] == raises[1][1]), None)
                    if resp:
                        eff = min(h["s"][raises[1][1]], h["s"][a[1]]) / bb
                        k = ("4bet-jam" if a[2] == "A" else "4bet-raise", "<25" if eff < 25 else "25+")
                        r[k][0] += 1; r[k][1] += resp[2] == "F"
print(WHO, "facing 4-bet after their 3-bet: fold rate")
for k, v in sorted(r.items()): print("  ", k, f"n={v[0]} fold {v[1]/max(1,v[0]):.0%}")
print(WHO, "3-bet hand top% quantiles 10/25/50/75/90")
for k, L in sorted(tb.items()):
    L.sort(); q = lambda p: L[int(p*(len(L)-1))]
    print("  ", k, len(L), " ".join(f"{q(p):.0%}" for p in (.1,.25,.5,.75,.9)))
