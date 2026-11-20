"""Bet-size signatures per bot: open raise-to (in bb), 3-bet ratio, postflop first-bet fraction."""
import glob, json, sys
from collections import defaultdict, Counter
folder = sys.argv[1]
O = defaultdict(Counter); T = defaultdict(Counter); P = defaultdict(Counter); L = defaultdict(Counter)
for f in sorted(glob.glob(folder + "/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        cur, nr = bb, 0
        for a in pre:
            n = nm[a[1]]
            if nr == 0 and a[2] == "C": L[n]["limp"] += 1
            if a[2] == "R":
                if nr == 0: O[n][round(a[3] / bb, 2)] += 1
                elif nr == 1: T[n][round(a[3] / cur, 2)] += 1
            if a[2] in "RA" and a[3] > cur: cur = a[3]; nr += 1
        prev = 0
        for st in range(4):
            com = defaultdict(int); bet = False
            for a in (x for x in h["a"] if x[0] == st):
                if st > 0 and not bet and a[2] == "B":
                    P[nm[a[1]]][round((a[3]) / max(1, prev), 2)] += 1
                if a[2] in "BRA": bet = True
                if a[2] in "CBRAb" and a[3]: com[a[1]] = max(com[a[1]], a[3])
            prev += sum(com.values())
for n in sorted(O, key=lambda n: -sum(O[n].values())):
    tot = sum(O[n].values())
    print(f"{n[:16]:<17} open-to(bb): " + ", ".join(f"{k}:{v/tot:.0%}" for k, v in O[n].most_common(4)))
    t3 = sum(T[n].values())
    if t3: print(f"{'':<17} 3bet-x:      " + ", ".join(f"{k}:{v/t3:.0%}" for k, v in T[n].most_common(4)))
    tp = sum(P[n].values())
    if tp: print(f"{'':<17} post bet/pot: " + ", ".join(f"{k}:{v/tp:.0%}" for k, v in P[n].most_common(5)))
