"""Do opponents 3-bet OUR opens more than others' opens? (per opponent, deep vs short)"""
import glob, json
from collections import defaultdict
ME = "Oskar Behzadpour"
T = defaultdict(lambda: [0, 0, 0, 0])   # opp -> [faced our open, 3bet ours, faced other open, 3bet other]
for f in sorted(glob.glob("data/r2/*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        opener = None; seen = set()
        for a in pre:
            if opener is None:
                if a[2] == "R": opener = a[1]
                elif a[2] == "A": break
                continue
            s = a[1]
            if s in seen or s == opener: break
            seen.add(s)
            n = nm[s]
            if n == ME: 
                if a[2] in "RA": break
                continue
            ours = nm[opener] == ME
            T[n][0 if ours else 2] += 1
            if a[2] in "RA":
                T[n][1 if ours else 3] += 1
                break
print(f"{'opponent':<18}{'vs our opens':>18}{'vs other opens':>20}")
for n, v in sorted(T.items(), key=lambda kv: -kv[1][0]):
    if v[0] < 50: continue
    print(f"{n[:17]:<18}{v[1]/v[0]:10.1%} (n{v[0]:4d}){v[3]/max(1,v[2]):10.1%} (n{v[2]:5d})")
