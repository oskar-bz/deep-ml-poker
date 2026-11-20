import glob, json, sys
from collections import defaultdict, Counter
ME = "Oskar Behzadpour"
R = "23456789TJQKA"
def cls(h):
    a, b = sorted(h, key=lambda c: -R.index(c[0]))
    return a[0] + b[0] + ("" if a[0] == b[0] else ("s" if a[1] == b[1] else "o"))
surv = Counter(); bust_by_ctx = Counter(); our_place = Counter(); our_bust_alive = Counter()
face3 = defaultdict(lambda: [0, 0, 0.0])  # 3bettor -> [times we faced, folded, bb lost by folding]
mw3 = Counter(); mwopen = defaultdict(Counter); calls_allin = defaultdict(list)
endgame = defaultdict(float)
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    final_alive = sum(1 for x in g["stacks"] if x > 0); surv[final_alive] += 1
    for s, n in enumerate(nm):
        if n == ME:
            our_place[g["places"][s]] += 1
            if g["busted_hand"][s] is not None:
                # players alive when we busted
                bh = g["hands"][g["busted_hand"][s]]
                our_bust_alive[sum(1 for x in bh["s"] if x > 0)] += 1
    for h in g["hands"]:
        bb = h["b"][1]; alive = sum(1 for x in h["s"] if x > 0)
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        raises = []
        for i, a in enumerate(pre):
            if a[2] in "RA": raises.append(a)
            if len(raises) == 2 and a is raises[1]:
                opener = raises[0][1]
                # opener's response
                resp = next((x for x in pre[i+1:] if x[1] == opener), None)
                if nm[opener] == ME and resp:
                    st = face3[nm[a[1]]]; st[0] += 1
                    if resp[2] == "F": st[1] += 1; st[2] += raises[0][3] / bb
        for a in pre[:]:
            pass
        # mw ranges
        for s, n in enumerate(nm):
            if h["s"][s] <= 0: continue
            if n == ME:
                endgame[(min(alive, 5), "bb")] += h["d"][s] / bb; endgame[(min(alive, 5), "n")] += 1
            if n not in ("mw",): continue
            eff = h["s"][s] / bb
            my = [a for a in pre if a[1] == s]
            if not my: continue
            before = [a for a in pre if pre.index(a) < pre.index(my[0])]
            nraise = sum(a[2] in "RA" for a in before)
            depth = "<15" if eff < 15 else ("15-40" if eff < 40 else ">40")
            if nraise == 0:
                mwopen[(depth, my[0][2])][cls(h["hole"][s])] += 1
            elif nraise == 1:
                mw3[(depth, my[0][2])] += 1
print("survivors at end of game:", sorted(surv.items()))
print("our places:", sorted(our_place.items()))
print("players alive when we busted:", sorted(our_bust_alive.items()))
print("our bb by players alive:", {k: round(endgame[(k,'bb')]/endgame[(k,'n')],2) for k in range(2,6) if endgame[(k,'n')]}, {k: int(endgame[(k,'n')]) for k in range(2,6)})
print("\nwe opened, got 3bet by X -> [n, folded, bb surrendered]:")
for k, v in sorted(face3.items(), key=lambda kv: -kv[1][0])[:12]: print(f"  {k:<20}{v[0]:5d} fold {v[1]/v[0]:4.0%}  {v[2]:6.0f}bb")
print("\nmw facing an open:", sorted(mw3.items()))
for key in sorted(mwopen):
    c = mwopen[key]; tot = sum(c.values())
    print("mw unopened", key, tot, c.most_common(12))
