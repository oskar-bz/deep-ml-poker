"""Style stats of each entrant's bot from downloaded demo replays (data/replays)."""
import glob, json
from collections import defaultdict

S = defaultdict(lambda: defaultdict(float))
names = {}
for f in glob.glob("data/replays/v-*.json"):
    g = json.load(open(f))["game"]
    me = next(p for p in g["players"] if not p["house"])
    seat, key = me["seat"], me["bot_id"][:8]
    names[key] = me["name"]
    st = S[key]
    st["pts"] += g["points"][seat]; st["tables"] += 1
    for h in g["hands"]:
        if h["s"][seat] <= 0:
            continue
        bb = h["b"][1]
        acts = [a for a in h["a"] if a[1] == seat]
        pre = [a for a in acts if a[0] == 0 and a[2] not in "b"]
        if not pre:
            continue
        st["hands"] += 1
        st["vpip"] += any(a[2] in "CRA" for a in pre)
        st["pfr"] += any(a[2] in "RA" for a in pre)
        st["pre_allin"] += any(a[2] == "A" for a in pre)
        opens = [a for a in pre if a[2] == "R"]
        if opens:
            st["open_sz"] += opens[0][3] / bb; st["open_n"] += 1
        post = [a for a in acts if a[0] > 0]
        st["post_aggr"] += sum(a[2] in "BRA" for a in post)
        st["post_pass"] += sum(a[2] in "CX" for a in post)
        st["post_fold"] += sum(a[2] == "F" for a in post)
        st["ms"] += sum(a[4] for a in acts if len(a) > 4)
print(f"{'entrant':<18}{'pts':>6}{'VPIP':>6}{'PFR':>6}{'AI%':>6}{'open':>6}{'AF':>6}{'pfold':>6}")
for k, st in sorted(S.items(), key=lambda kv: -kv[1]["pts"]):
    n = max(1, st["hands"])
    print(f"{names[k][:17]:<18}{st['pts']/st['tables']:6.2f}{st['vpip']/n:6.0%}{st['pfr']/n:6.0%}{st['pre_allin']/n:6.0%}"
          f"{st['open_sz']/max(1,st['open_n']):6.1f}{st['post_aggr']/max(1,st['post_pass']):6.2f}"
          f"{st['post_fold']/max(1,st['post_aggr']+st['post_pass']+st['post_fold']):6.0%}")
