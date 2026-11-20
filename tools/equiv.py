"""Decision-equivalence check between two single-file builds.

Records every act(state) / on_hand_end(result) event while build A plays real simulated tables, then
replays the identical event sequence into fresh instances of A and B with the RNG re-seeded before every
decision and all time budgets lifted. Any differing decision is a real behaviour change.

python tools/equiv.py A.py B.py [tables] [field: v3|adv]
"""
import copy
import importlib.util
import random
import sys

sys.path.insert(0, ".")


def load(path, tag):
    spec = importlib.util.spec_from_file_location("equiv_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fresh_bot(mod):
    bot = mod.MyBot()
    core = bot.core
    for attr in ("MC_BUDGET_NS",):
        if hasattr(core, attr):
            setattr(core, attr, 10 ** 12)
    if hasattr(core, "BR_BUDGET_NS"):
        core.BR_BUDGET_NS = 10 ** 12
    return bot


def record(mod, tables, kinds):
    from pokerbot_sdk import SDKBotDriver
    from poker_engine.simulator import SitAndGoTable
    import bots.sdk_field as F
    logs = []

    class Rec:
        def __init__(self, bot):
            self.bot = bot; self.ev = []

        def act(self, state):
            self.ev.append(("act", copy.deepcopy(state)))
            return self.bot.act(state)

        def on_hand_end(self, result):
            self.ev.append(("end", copy.deepcopy(result)))
            return self.bot.on_hand_end(result)

    for sd in range(tables):
        random.seed(sd)
        rec = Rec(fresh_bot(mod))
        opp = [random.choice(kinds)(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json")
               for _ in range(7)]
        SitAndGoTable([SDKBotDriver(rec, "h")] + [SDKBotDriver(o, o.name) for o in opp], seed=sd).play_tournament()
        logs.append(rec.ev)
    return logs


def replay(mod, ev):
    bot = fresh_bot(mod)
    out = []
    for k, (kind, payload) in enumerate(ev):
        if kind == "act":
            random.seed(1000003 * k + 7)
            a = bot.act(copy.deepcopy(payload))
            out.append((a.kind, getattr(a, "amount", None)))
        else:
            bot.on_hand_end(copy.deepcopy(payload))
    return out


if __name__ == "__main__":
    import bots.sdk_field as F
    A = load(sys.argv[1], "a"); B = load(sys.argv[2], "b")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    kinds = [F.R2ModelBot] if (len(sys.argv) < 5 or sys.argv[4] == "v3") else \
        [F.R2ModelBot, F.Adv3BetBot, F.Adv4BetBot, F.AdvJamBot, F.AdvCallerBot, F.AdvRaiseBot]
    logs = record(A, n, kinds)
    tot = diff = 0
    for t, ev in enumerate(logs):
        ra, rb = replay(A, ev), replay(B, ev)
        for i, (x, y) in enumerate(zip(ra, rb)):
            tot += 1
            if x != y:
                diff += 1
                if diff <= 5:
                    print("DIFF table", t, "decision", i, x, y)
    print("decisions compared: %d   differing: %d" % (tot, diff))
