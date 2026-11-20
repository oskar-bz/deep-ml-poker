"""Real fold rate vs postflop bet size (heads-up, first bet of street)."""
import glob, json
from collections import defaultdict
ME = "Oskar Behzadpour"
T = defaultdict(lambda: [0, 0])
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        prev = 0; folded = set()
        for st in range(4):
            acts = [a for a in h["a"] if a[0] == st]
            com = defaultdict(int)
            bet = None
            for a in acts:
                s, c, amt = a[1], a[2], a[3]
                if c == "F":
                    folded.add(s)
                live = [x for x in range(8) if h["s"][x] > 0 and x not in folded]
                if st > 0 and bet is not None and s != bet[0] and len(live) + (c == "F") == 2:
                    frac = bet[1]
                    b = "<0.4" if frac < 0.4 else ("0.4-0.7" if frac < 0.7 else ("0.7-1.1" if frac < 1.1 else ">1.1"))
                    key = (nm[s], b)
                    T[key][0] += 1; T[key][1] += c == "F"
                    T[("rest", b)][0] += 1; T[("rest", b)][1] += c == "F"
                    bet = None
                if c in "CBRAb" and amt:
                    if st > 0 and c in "BA" and bet is None and all(v == 0 for v in com.values()):
                        pot = prev + sum(com.values())
                        bet = (s, (amt - com[s]) / max(1, pot))
                    com[s] = max(com[s], amt)
                if st > 0 and c in "RA" and bet is not None and s == bet[0]:
                    bet = None
            prev += sum(com.values())
for n in ("mw", "Shresth Rana", "Daksh Malhan", "thearly man", "Henrik Gombos", "rest"):
    print(f"{n:<15}" + "  ".join(f"{b}: {T[(n,b)][1]/max(1,T[(n,b)][0]):4.0%} (n{T[(n,b)][0]})" for b in ("<0.4", "0.4-0.7", "0.7-1.1", ">1.1")))

# write per-bot fold multipliers by size bucket into data/r1_models.json
models = json.load(open("data/r1_models.json", encoding="utf-8"))
B = ("<0.4", "0.4-0.7", "0.7-1.1", ">1.1")
def mult(n):
    tot = sum(T[(n, b)][0] for b in B); fl = sum(T[(n, b)][1] for b in B)
    base = fl / max(1, tot)
    out = []
    for b in B[:3]:
        t, f_ = T[(n, b)]
        r = ((f_ + 10 * base) / (t + 10)) / max(1e-6, base)
        out.append(round(r, 3))
    return out
rest = mult("rest")
for n in models:
    models[n]["fold_mult"] = mult(n) if (n, "<0.4") in T and T[(n, "<0.4")][0] > 0 else rest
json.dump(models, open("data/r1_models.json", "w"), indent=0)
print("rest mult", rest)
