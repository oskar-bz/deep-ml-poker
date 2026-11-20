import glob, json, sys
from collections import defaultdict, Counter
sys.path.insert(0, ".")
from poker_engine.range_engine import PREFLOP_RANK
R = "23456789TJQKA"
def cls(h):
    a, b = sorted(h, key=lambda c: -R.index(c[0]))
    return a[0] + b[0] + ("" if a[0] == b[0] else ("s" if a[1] == b[1] else "o"))
WHO = sys.argv[1:] or ["mw", "Daksh Malhan", "Shresth Rana"]
out = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        nr = 0
        for a in pre:
            n = nm[a[1]]
            if n in WHO and nr == 1 and a[2] in "RAF":
                eff = h["s"][a[1]] / bb
                d = "<15" if eff < 15 else ("15-40" if eff < 40 else ">40")
                out[n][(d, a[2])].append(cls(h["hole"][a[1]]))
            if a[2] in "RA": nr += 1
print(type(PREFLOP_RANK), list(PREFLOP_RANK.items())[:3] if hasattr(PREFLOP_RANK, "items") else PREFLOP_RANK[:3])
from poker_engine.preflop_multiway import MULTIWAY_EQ
def key2cls(k):
    hi, lo, s = k
    return R[hi] + R[lo] + ("" if hi == lo else ("s" if s else "o"))
# rank index may be 0..12 or 2..14
ks = list(MULTIWAY_EQ)
off = min(k[0] for k in ks)
order = sorted(ks, key=lambda k: -MULTIWAY_EQ[k][0])
TOP = {}; cum = 0
for k in order:
    w = 6 if k[0] == k[1] else (4 if k[2] else 12)
    cum += w; TOP[R[k[0]-off] + R[k[1]-off] + ("" if k[0] == k[1] else ("s" if k[2] else "o"))] = cum / 1326
for n in WHO:
    print("\n==", n)
    tot = Counter()
    for key, L in out[n].items(): tot[key[0]] += len(L)
    for key in sorted(out[n]):
        L = [TOP[c] for c in out[n][key]]
        if len(L) < 10: continue
        L.sort()
        q = lambda p: L[int(p * (len(L) - 1))]
        print(f"  facing open {key[0]:>6} {key[1]}: {len(L):5d} ({len(L)/tot[key[0]]:4.0%})  hand top% quantiles 10/25/50/75/90: "
              + " ".join(f"{q(p):.0%}" for p in (.1, .25, .5, .75, .9)))
