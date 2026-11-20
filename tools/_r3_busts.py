import glob, json, sys
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models import top_of
ME = "Oskar Behzadpour"
C = Counter(); V = Counter(); alive_c = Counter(); detail = defaultdict(list)
for f in sorted(glob.glob("data/r3/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for s, n in enumerate(nm):
        if n != ME or g["players"][s]["copy"] or g["busted_hand"][s] is None: continue
        h = g["hands"][g["busted_hand"][s]]; bb = h["b"][1]
        alive = sum(1 for x in h["s"] if x > 0); alive_c[alive] += 1
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        my = [a for a in h["a"] if a[1] == s and a[2] != "b"]
        st_bb = h["s"][s] / bb
        last_st = max(a[0] for a in my) if my else 0
        nr_before = 0; kind = None
        for a in pre:
            if a[1] == s and a[2] in "RAC" and kind is None:
                if a[2] in "RA":
                    kind = ["open-jam/raise", "3bet/rejam", "4bet+"][min(nr_before, 2)]
                else:
                    kind = ["limp", "call-open", "call-3bet+"][min(nr_before, 2)]
            if a[2] in "RA": nr_before += 1
        if last_st > 0: kind = "postflop(" + (kind or "?") + ")"
        depth = "<10" if st_bb < 10 else ("10-20" if st_bb < 20 else ("20-40" if st_bb < 40 else "40+"))
        C[(kind, depth)] += 1
        winners = [x for x in range(8) if h["d"][x] > 0]
        for w in winners: V[nm[w]] += 1
        detail[(kind, depth)].append((round(top_of(h["hole"][s]), 2), [round(top_of(h["hole"][w]), 2) for w in winners], alive))
print("busts by players alive:", sorted(alive_c.items()))
for k, v in sorted(C.items(), key=lambda kv: -kv[1]):
    d = detail[k]
    print(f"{str(k):<40}{v:4d}  our hand top% median {sorted(x[0] for x in d)[len(d)//2]:.2f}  winner hand median {sorted(x[1][0] for x in d if x[1])[len(d)//2] if d else 0:.2f}")
print("busted by:", V.most_common(9))
