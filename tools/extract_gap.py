"""Where does B extract more than A from each opponent? bb per 100 hands dealt together, by situation.
Situation from the hero's view: who opened (hero / opp / other / nobody) x hand ended preflop or postflop.
python tools/extract_gap.py data/r3 "Oskar Behzadpour" mw"""
import glob, json, sys
from collections import defaultdict
folder, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
R = defaultdict(float); N = defaultdict(int)
for f in glob.glob(folder + "/*.json"):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        opener = next((a[1] for a in pre if a[2] in "RA"), None)
        limper = next((a[1] for a in pre if a[2] == "C"), None) if opener is None else None
        ended_pre = max(a[0] for a in h["a"]) == 0
        for hero in (A, B):
            for i in [k for k, n in enumerate(nm) if n == hero and not g["players"][k]["copy"]]:
                if h["s"][i] <= 0: continue
                for o in range(8):
                    if o == i or h["s"][o] <= 0: continue
                    on = nm[o]; N[(hero, on)] += 1
                    d = h["d"][i]
                    if d > 0 and h["d"][o] < 0:
                        share = -h["d"][o] / sum(-h["d"][x] for x in range(8) if x != i and h["d"][x] < 0)
                    elif d < 0 and h["d"][o] > 0:
                        share = h["d"][o] / sum(h["d"][x] for x in range(8) if x != i and h["d"][x] > 0)
                    else:
                        continue
                    who = "hero-open" if opener == i else ("opp-open" if opener == o else ("other-open" if opener is not None else "limp/none"))
                    sit = who + ("/pre" if ended_pre else "/post")
                    R[(hero, on, sit)] += d * share / bb
opps = sorted({k[1] for k in N if k[0] == A} & {k[1] for k in N if k[0] == B} - {A, B})
sits = sorted({k[2] for k in R})
for o in opps:
    na, nb = N[(A, o)], N[(B, o)]
    if na < 2000 or nb < 2000: continue
    ta = 100 * sum(R[(A, o, s)] for s in sits) / na; tb = 100 * sum(R[(B, o, s)] for s in sits) / nb
    print(f"\n{o}:  {A[:10]} {ta:+.2f}  {B} {tb:+.2f}  bb/100 (gap {tb - ta:+.2f})")
    rows = []
    for s in sits:
        ra = 100 * R[(A, o, s)] / na; rb = 100 * R[(B, o, s)] / nb
        rows.append((rb - ra, s, ra, rb))
    for gap, s, ra, rb in sorted(rows, reverse=True)[:4]:
        print(f"    {s:<18} us {ra:+6.2f}   {B} {rb:+6.2f}   gap {gap:+.2f}")
