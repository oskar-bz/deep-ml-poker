"""Round-1 tournament replay analysis (data/r1). Usage: python tools/r1_analyze.py [our name]"""
import glob, json, sys
from collections import defaultdict, Counter

ME = sys.argv[1] if len(sys.argv) > 1 else "Oskar Behzadpour"
S = defaultdict(lambda: defaultdict(float))
faults = []
place_ctr = defaultdict(Counter)
our = defaultdict(float)          # our chip flow by category
vs = defaultdict(lambda: defaultdict(float))   # our chips won/lost per opponent in 2-way pots
games = 0
for f in sorted(glob.glob("data/r1/r1-*.json")):
    g = json.load(open(f, encoding="utf-8"))["game"]
    games += 1
    nm = [p["name"] for p in g["players"]]
    for s, p in enumerate(g["players"]):
        st = S[p["name"]]
        st["tables"] += 1; st["pts"] += g["points"][s]
        place_ctr[p["name"]][g["places"][s]] += 1
        ss = g["seat_stats"][s]
        st["timeouts"] += ss["timeouts"]; st["errors"] += ss["errors"]; st["crash"] += ss["crashed"]
        st["maxms"] = max(st["maxms"], ss["max_ms"]); st["wallmax"] = max(st["wallmax"], ss["wall_max_ms"])
        if p["name"] == ME and (ss["timeouts"] or ss["errors"] or ss["crashed"]):
            faults.append((g["game_id"], ss))
    for h in g["hands"]:
        bb = h["b"][1]
        acts = h["a"]
        pre = [a for a in acts if a[0] == 0 and a[2] != "b"]
        # preflop sequence analysis
        first_raiser = None; raises = 0
        seen = set()
        for a in pre:
            n = nm[a[1]]; st = S[n]
            if a[1] not in seen:
                seen.add(a[1]); st["hands"] += 1
                if raises == 0:
                    st["unopened"] += 1
                    if a[2] in "RA": st["open"] += 1; st["open_sz"] += a[3] / bb
                    if a[2] == "A": st["open_jam"] += 1
                    if a[2] == "C": st["limp"] += 1
                elif raises == 1:
                    st["face_open"] += 1
                    if a[2] in "RA": st["threebet"] += 1
                    if a[2] == "C": st["flat"] += 1
                if a[2] in "CRA": st["vpip"] += 1
            if a[1] == first_raiser and raises >= 2 and not st.get("_done%d" % h["id"]):
                st["_done%d" % h["id"]] = 1
                st["face3b"] += 1
                st["fold3b"] += a[2] == "F"
                st["call3b"] += a[2] == "C"
                st["4b"] += a[2] in "RA"
            if a[2] in "RA":
                raises += 1
                if first_raiser is None: first_raiser = a[1]
        for k in list(S[nm[first_raiser]].keys()) if first_raiser is not None else []:
            if k.startswith("_done"): del S[nm[first_raiser]][k]
        # facing all-in calls: record caller hand strength
        # postflop: cbet and fold-to-bet
        for street in (1, 2, 3):
            sa = [a for a in acts if a[0] == street]
            bet_seen = False
            for a in sa:
                st = S[nm[a[1]]]
                if not bet_seen:
                    st["post_first"] += 1
                    if a[2] in "BA": st["post_bet"] += 1; bet_seen = True
                    elif a[2] in "RA": bet_seen = True
                else:
                    st["post_facebet"] += 1
                    st["post_foldbet"] += a[2] == "F"
                    st["post_raise"] += a[2] in "RA"
        # our chip flow
        if ME in nm:
            for s, n in enumerate(nm):
                if n != ME or h["s"][s] <= 0: continue
                d = h["d"][s] / bb
                my = [a for a in acts if a[1] == s and a[2] != "b"]
                players = {a[1] for a in acts if a[2] != "b" and a[2] != "F"} | {a[1] for a in acts if a[2] == "F" and a[0] > 0}
                if not my: continue
                mpre = [a for a in my if a[0] == 0]
                tag = "other"
                if any(a[2] == "A" for a in my): tag = "we_allin"
                elif any(a[0] == 0 and a[2] == "R" for a in my) and max(a[0] for a in acts) == 0: tag = "pre_raise_won/lost"
                elif all(a[2] == "F" for a in mpre[:1]) and mpre: tag = "pre_fold"
                elif max(a[0] for a in my) >= 1: tag = "postflop"
                our[tag] += d; our[tag + "_n"] += 1
                # 2-way pots
                live = [x for x in range(8) if any(a[1] == x and a[2] != "F" and a[2] != "b" for a in acts)]
                if len(live) == 2 and s in live:
                    o = nm[[x for x in live if x != s][0]]
                    vs[o]["bb"] += d; vs[o]["n"] += 1

print("games", games)
print(f"{'bot':<20}{'pts':>5}{'1st':>5}{'2nd':>5}{'3rd':>5}{'VPIP':>5}{'open':>5}{'osz':>5}{'jam':>5}{'limp':>5}{'3b':>5}{'flat':>5}{'f3b':>5}{'c3b':>5}{'bet1':>5}{'fbet':>5}{'rbet':>5}{'TO':>5}{'err':>4}{'maxms':>6}")
for n, st in sorted(S.items(), key=lambda kv: -kv[1]["pts"] / max(1, kv[1]["tables"])):
    if st["tables"] < 30: continue
    t = st["tables"]; pc = place_ctr[n]
    h = max(1, st["hands"]); u = max(1, st["unopened"]); fo = max(1, st["face_open"]); f3 = max(1, st["face3b"])
    print(f"{n[:19]:<20}{st['pts']/t:5.2f}{pc[1]/t:5.0%}{pc[2]/t:5.0%}{pc[3]/t:5.0%}{st['vpip']/h:5.0%}{st['open']/u:5.0%}"
          f"{st['open_sz']/max(1,st['open']):5.1f}{st['open_jam']/u:5.0%}{st['limp']/u:5.0%}{st['threebet']/fo:5.0%}{st['flat']/fo:5.0%}"
          f"{st['fold3b']/f3:5.0%}{st['call3b']/f3:5.0%}{st['post_bet']/max(1,st['post_first']):5.0%}"
          f"{st['post_foldbet']/max(1,st['post_facebet']):5.0%}{st['post_raise']/max(1,st['post_facebet']):5.0%}"
          f"{int(st['timeouts']):5d}{int(st['errors']):4d}{st['maxms']:6.2f}")
print("\nOUR faults:", len(faults))
for gid, ss in faults[:15]:
    print(" ", gid, {k: ss[k] for k in ("timeouts", "errors", "crashed", "max_ms", "wall_max_ms", "decisions")})
print("\nOUR chip flow (bb):")
for k in sorted(k for k in our if not k.endswith("_n")):
    print(f"  {k:<20}{our[k]:9.0f} bb over {int(our[k+'_n'])} hands ({our[k]/max(1,our[k+'_n']):+.2f}/hand)")
print("\nOUR 2-way pots per opponent (bb):")
for o, v in sorted(vs.items(), key=lambda kv: kv[1]["bb"]):
    if v["n"] >= 30: print(f"  {o[:20]:<22}{v['bb']:8.0f} bb  n={int(v['n'])}  {v['bb']/v['n']:+.2f}/pot")
