"""How do WE respond to bets by size, and were we ahead/behind the bettor at that moment (current made hand)?"""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from tools.r1_tells import strength
ME = "Oskar Behzadpour"
R = defaultdict(lambda: [0, 0, 0, 0, 0.0])   # bucket -> [n, called, called_behind, folded_ahead, bb won on calls (hand delta)]
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    if ME not in nm: continue
    for h in g["hands"]:
        prev = 0
        for st in range(1, 4):
            pass
        prev = 0
        for st in range(4):
            com = defaultdict(int); bet = None
            for a in (x for x in h["a"] if x[0] == st):
                s, c, amt = a[1], a[2], a[3]
                if st > 0 and bet and nm[s] == ME and s != bet[0]:
                    board = h["board"][: {1: 3, 2: 4, 3: 5}[st]]
                    me, them = strength(h["hole"][s], board), strength(h["hole"][bet[0]], board)
                    b = bet[1]
                    k = "<.4" if b < 0.4 else (".4-.7" if b < 0.7 else (".7-1.1" if b < 1.1 else ">1.1"))
                    if bet[2]: k = "raise"
                    r = R[k]; r[0] += 1
                    if c in "CA":
                        r[1] += 1; r[2] += me < them; r[4] += h["d"][s] / h["b"][1]
                    elif c == "F":
                        r[3] += me > them
                    bet = None
                if st > 0 and c in "BRA" and nm[s] != ME:
                    pot = prev + sum(com.values())
                    bet = (s, (amt - com[s]) / max(1, pot), c == "R" or any(v > 0 for v in com.values()))
                elif c in "BRA":
                    bet = None
                if c in "CBRAb" and amt:
                    com[s] = max(com[s], amt)
            prev += sum(com.values())
print("bucket     n   call%  calls-behind%  folds-while-ahead%  avg hand result when calling (bb)")
for k in ("<.4", ".4-.7", ".7-1.1", ">1.1", "raise"):
    n, c, cb, fa, w = R[k]
    if n: print(f"{k:<8}{n:5d}  {c/n:5.0%}   {cb/max(1,c):10.0%}   {fa/max(1,n-c):14.0%}        {w/max(1,c):+.2f}")
