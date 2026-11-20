"""Per opponent: bb won per 100 hands dealt together (all pots, not only heads-up), for two heroes.
python tools/who_wins_from.py data/r3 "Oskar Behzadpour" mw"""
import glob, json, sys
from collections import defaultdict
folder, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
R = {A: defaultdict(lambda: [0.0, 0]), B: defaultdict(lambda: [0.0, 0])}
for f in glob.glob(folder + "/*.json"):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        for hero in (A, B):
            for i in [k for k, n in enumerate(nm) if n == hero and not g["players"][k]["copy"]]:
                if h["s"][i] <= 0: continue
                # attribute our net result to opponents in proportion to what they lost/won in this hand
                losers = {o: -h["d"][o] for o in range(8) if o != i and h["d"][o] < 0}
                winners = {o: h["d"][o] for o in range(8) if o != i and h["d"][o] > 0}
                d = h["d"][i]
                if d > 0 and losers:
                    tot = sum(losers.values())
                    for o, v in losers.items(): R[hero][nm[o]][0] += d * v / tot / bb
                elif d < 0 and winners:
                    tot = sum(winners.values())
                    for o, v in winners.items(): R[hero][nm[o]][0] += d * v / tot / bb
                for o in range(8):
                    if o != i and h["s"][o] > 0: R[hero][nm[o]][1] += 1
opps = sorted(set(R[A]) | set(R[B]), key=lambda o: -(R[A][o][1] + R[B][o][1]))
print(f"{'opponent':<18}{A[:14]+' bb/100':>22}{B[:14]+' bb/100':>22}{'diff':>8}")
for o in opps:
    if o in (A, B): continue
    a = R[A][o]; b = R[B][o]
    if a[1] < 2000 or b[1] < 2000: continue
    ra = 100 * a[0] / a[1]; rb = 100 * b[0] / b[1]
    print(f"{o[:17]:<18}{ra:22.2f}{rb:22.2f}{ra - rb:8.2f}")
a = R[A][B]; b = R[B][A]
print(f"head-to-head: {A[:14]} vs {B}: {100*a[0]/max(1,a[1]):.2f} bb/100 ; {B} vs us: {100*b[0]/max(1,b[1]):.2f}")
