"""Do real bots respond to ICM pressure? Fold-to-open and call-vs-jam by players alive / depth."""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models import top_of
ME = "Oskar Behzadpour"
F = defaultdict(lambda: [0, 0])      # (name, alive_bucket) -> [faced open (not all-in) as non-opener, folded]
J = defaultdict(lambda: [0, 0, []])  # (name, alive_bucket) -> [faced jam, called, called-hand top%]
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]; alive = sum(1 for x in h["s"] if x > 0)
        ab = "8-6" if alive >= 6 else ("5" if alive == 5 else ("4" if alive == 4 else ("3" if alive == 3 else "2")))
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        nr = 0; jam_open = False; seen = set()
        for a in pre:
            s = a[1]
            if s not in seen and nr == 1:
                seen.add(s)
                if jam_open:
                    st = J[(nm[s], ab)]; st[0] += 1
                    if a[2] in "CA": st[1] += 1; st[2].append(top_of(h["hole"][s]))
                else:
                    st = F[(nm[s], ab)]; st[0] += 1; st[1] += a[2] == "F"
            seen.add(s)
            if a[2] in "RA":
                nr += 1
                if nr == 1: jam_open = a[2] == "A"
names = sorted({k[0] for k in F if sum(F[(k[0], b)][0] for b in ("8-6",)) > 300})
print("FOLD vs open (non-allin)   by players alive:   8-6     5     4     3     2")
for n in names:
    print(f"  {n[:18]:<20}" + "".join(f"{F[(n,b)][1]/max(1,F[(n,b)][0]):6.0%}" for b in ("8-6", "5", "4", "3", "2")))
print("CALL vs open-jam (% called, median called hand top%)   8-6 / 5 / 4 / 3 / 2")
for n in names:
    row = ""
    for b in ("8-6", "5", "4", "3", "2"):
        st = J[(n, b)]
        L = sorted(st[2]); med = L[len(L)//2] if L else 0
        row += f"  {st[1]/max(1,st[0]):4.0%}({med:3.0%},n{st[0]:3d})"
    print(f"  {n[:18]:<20}{row}")
