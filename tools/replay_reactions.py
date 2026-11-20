"""How each entrant reacts to aggression (from data/replays, hole cards known)."""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, ".")
from poker_engine.evaluator import parse_card
from poker_engine.preflop_tables import PREFLOP_PERCENTILE

S = defaultdict(lambda: defaultdict(list))
names = {}
for f in glob.glob("data/replays/v-*.json"):
    g = json.load(open(f))["game"]
    me = next(p for p in g["players"] if not p["house"])
    seat, key = me["seat"], me["bot_id"][:8]
    names[key] = me["name"]
    for h in g["hands"]:
        if h["s"][seat] <= 0:
            continue
        c0, c1 = (parse_card(x) for x in h["hole"][seat])
        pct = PREFLOP_PERCENTILE[c0][c1]
        pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
        raised_by_me = False
        for i, a in enumerate(pre):
            if a[1] == seat and a[2] in "RA" and not raised_by_me:
                raised_by_me = True
                # next action of mine after someone re-raised
                rr = [b for b in pre[i + 1:] if b[1] != seat and b[2] in "RA"]
                if rr:
                    nxt = [b for b in pre[pre.index(rr[0]) + 1:] if b[1] == seat]
                    if nxt:
                        S[key]["vs3bet"].append((nxt[0][2], pct))
            if a[1] != seat and a[2] == "A":
                nxt = [b for b in pre[i + 1:] if b[1] == seat]
                if nxt:
                    S[key]["vs_allin"].append((nxt[0][2], pct))
        # unopened-pot open range
        before = []
        for a in pre:
            if a[1] == seat:
                if not any(b[2] in "RAC" for b in before):
                    S[key]["open"].append((a[2], pct))
                break
            before.append(a)

print(f"{'entrant':<18}{'n3b':>5}{'fold3b':>7}{'n_ai':>5}{'callAI':>7}{'callAI_minpct':>14}{'open%':>7}{'open_minpct':>12}")
for k in sorted(names, key=lambda k: names[k]):
    v3 = S[k]["vs3bet"]; va = S[k]["vs_allin"]; op = S[k]["open"]
    f3 = sum(1 for a, _ in v3 if a == "F") / max(1, len(v3))
    calls = [p for a, p in va if a in "CA"]
    ca = len(calls) / max(1, len(va))
    opens = [p for a, p in op if a in "RA"]
    print(f"{names[k][:17]:<18}{len(v3):5d}{f3:7.0%}{len(va):5d}{ca:7.0%}{(min(calls) if calls else float('nan')):14.2f}"
          f"{len(opens)/max(1,len(op)):7.0%}{(sorted(opens)[len(opens)//20] if opens else float('nan')):12.2f}")
