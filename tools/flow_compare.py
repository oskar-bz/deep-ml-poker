import sys, glob, json, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
from tools.flowdefs import categorize
ME = "Oskar Behzadpour"
KIND = {"F": "fold", "X": "check", "C": "call", "B": "raise", "R": "raise", "A": "raise", "b": "blind"}


def real():
    S = defaultdict(float)
    for f in glob.glob("data/r1/r1-*.json"):
        g = json.load(open(f, encoding="utf-8"))["game"]
        nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
        for me in [i for i, n in enumerate(nm) if n == ME]:
            for h in g["hands"]:
                if h["s"][me] <= 0: continue
                acts = []
                contrib = 0
                for st in range(4):
                    mx = 0
                    for a in (x for x in h["a"] if x[0] == st):
                        acts.append((st, a[1], KIND[a[2]], a[3]))
                        if a[1] == me and a[3]: mx = max(mx, a[3])
                    contrib += mx
                c = categorize(acts, me, h["s"][me], contrib, h["d"][me] / h["b"][1])
                if c: add(S, c)
    return S


def add(S, c):
    S["hands"] += 1
    S[c["cat"]] += 1; S[c["cat"] + "_bb"] += c["d"]
    for k in ("open", "open_steal", "open_3bet"):
        S[k] += c.get(k, 0)
    if "pb" in c:
        S["o" + c["pb"]] += 1; S["s" + c["pb"]] += c["open_steal"]; S["t" + c["pb"]] += c["open_3bet"]


def sim_run(seeds):
    from tools.calib import _load
    from poker_engine.simulator import SitAndGoTable
    from pokerbot_sdk import SDKBotDriver
    from bots.sdk_field import R2ModelBot
    mod = _load("contest/round1_submission_bot.py")
    S = defaultdict(float)

    class Spy(SDKBotDriver):
        def act(self, state):
            self._bb = state["big_blind"]; return super().act(state)

        def on_hand_end(self, idx, summ):
            super().on_hand_end(idx, summ)
            st0 = summ["starting_stacks"][idx]
            if st0 <= 0: return
            acts = [(a[0], a[1], {"sb": "blind", "bb": "blind"}.get(a[2], a[2]), a[3]) for a in summ["action_history"]]
            c = categorize(acts, idx, st0, summ["hand_contributions"][idx],
                           (summ["ending_stacks"][idx] - st0) / self._bb)
            if c: add(S, c)

    for sd in seeds:
        rng = random.Random(sd)
        bots = [Spy(mod.MyBot(), "hero")]
        while len(bots) < 8:
            b = R2ModelBot(None, rng.randrange(1 << 30)); bots.append(SDKBotDriver(b, b.name))
        SitAndGoTable(bots, seed=sd).play_tournament()
    return S


def show(tag, S):
    n = S["hands"]
    row = f"{tag:<6}"
    for k in ("pre_fold", "pre_raise_end", "postflop", "allin_pre", "allin_post"):
        row += f" {k}: {S[k]/n:5.1%} {S[k+'_bb']/max(1,S[k]):+6.2f}bb |"
    row += f" opens {S['open']/n:5.1%} steal-won {S['open_steal']/max(1,S['open']):4.0%} got-3bet {S['open_3bet']/max(1,S['open']):4.0%}"
    print(row)
    parts = []
    for b in ("1", "2-3", "4+"):
        o = max(1, S["o" + b])
        parts.append(f"{b}: n {100*S['o'+b]/n:4.1f}% won {S['s'+b]/o:4.0%} 3bet {S['t'+b]/o:4.0%}")
    print("       by players behind: " + "  ".join(parts))


if __name__ == "__main__":
    show("REAL", real())
    W = 16; seeds = list(range(9000, 9000 + int(sys.argv[1])))
    T = defaultdict(float)
    with ProcessPoolExecutor(W) as ex:
        for S in ex.map(sim_run, [seeds[i::W] for i in range(W)]):
            for k, v in S.items(): T[k] += v
    show("SIM", T)
