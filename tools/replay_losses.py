"""Lists a bot's biggest losing hands in downloaded demo replays (data/replays)."""
import glob, json, sys

bot = sys.argv[1]
thr = int(sys.argv[2]) if len(sys.argv) > 2 else 400
for f in sorted(glob.glob(f"data/replays/v-{bot}-*.json")):
    g = json.load(open(f))["game"]
    seat = next(p["seat"] for p in g["players"] if p.get("bot_id") == bot)
    names = {p["seat"]: p["name"] for p in g["players"]}
    print(f"== table {g['game_id'][-7]} place {g['places'][seat]} busted_hand {g['busted_hand'][seat]}")
    for h in g["hands"]:
        d = h["d"][seat]
        if d <= -thr:
            acts = " ".join(f"{'PFTR'[a[0]]}:{names[a[1]][:6]}{'*' if a[1]==seat else ''}:{a[2]}{a[3]}" for a in h["a"])
            shown = {names[int(s)][:6] + str(s): h["hole"][int(s)] for s in h.get("show", [])}
            print(f"  hand {h['id']} lvl bb={h['b'][1]} start={h['s'][seat]} d={d} hole={h['hole'][seat]} board={h['board']}")
            print(f"     {acts}")
            print(f"     shown {shown}")
