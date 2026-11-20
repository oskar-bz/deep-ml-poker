import glob, json
from collections import Counter
ME = "Oskar Behzadpour"; R = "23456789TJQKA"
C = Counter(); ex = []
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for s, n in enumerate(nm):
        if n != ME or g["busted_hand"][s] is None: continue
        h = g["hands"][g["busted_hand"][s]]; bb = h["b"][1]
        my = [a for a in h["a"] if a[1] == s and a[2] != "b"]
        st = max(a[0] for a in my) if my else 0
        first_aggr = next((a for a in h["a"] if a[2] in "RA" and a[0] == st), None)
        we_aggr = any(a[2] in "RA" for a in my if a[0] == st)
        kind = ("pre" if st == 0 else "post") + ("_we_jam/raise" if we_aggr else "_we_call")
        stack_bb = h["s"][s] / bb
        C[(kind, "short" if stack_bb < 15 else "deep")] += 1
        winners = [x for x in range(8) if h["d"][x] > 0]
        ex.append((g["game_id"], kind, round(stack_bb, 1), h["hole"][s], [h["hole"][w] for w in winners], [nm[w] for w in winners], h["board"]))
for k, v in sorted(C.items(), key=lambda kv: -kv[1]): print(k, v)
for e in ex[:40]: print(e)
