"""Builds vs platform-style house bots (2x tight, call, 2x random, raise, fold), like the demo tables."""
import importlib.util, random, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")


def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(args):
    path, fp_off, seeds = args
    from pokerbot_sdk import SDKBotDriver, Action, Bot
    from poker_engine.simulator import SitAndGoTable
    import bots.sdk_field as F
    S = load(path, "b_%d" % abs(hash((path, fp_off))))
    if fp_off:
        S.MyBot.core_factory = type("NoFP", (S.ContestCore,), {"P_FP_ON": 0.0})

    class CallBot(Bot):
        def act(self, st): return Action.call() if st.to_call > 0 else Action.check()

    class FoldBot(Bot):
        def act(self, st): return Action.check() if st.to_call == 0 else Action.fold()

    class RandomBot(Bot):
        def __init__(self, seed): self.r = random.Random(seed)
        def act(self, st):
            x = self.r.random()
            if x < 0.3: return Action.check() if st.to_call == 0 else Action.fold()
            if x < 0.75 or "raise" not in st.legal: return Action.call() if st.to_call > 0 else Action.check()
            return Action.raise_to(min(st.max_raise, max(st.min_raise, st.current_bet + st.pot // 2)))

    class RaiseBot(Bot):
        def act(self, st):
            if "raise" in st.legal: return Action.raise_to(min(st.max_raise, max(st.min_raise, st.current_bet * 2 + st.big_blind)))
            return Action.call() if st.to_call > 0 else Action.check()

    pts = []; ids = Counter()
    for sd in seeds:
        rng = random.Random(sd)
        house = [F.StarterTightBot(), F.StarterTightBot(), CallBot(), RandomBot(rng.randrange(1 << 30)),
                 RandomBot(rng.randrange(1 << 30)), RaiseBot(), FoldBot()]
        rng.shuffle(house)
        hero = S.MyBot()
        res = SitAndGoTable([SDKBotDriver(hero, "hero")] + [SDKBotDriver(b, type(b).__name__) for b in house], seed=sd).play_tournament()
        pts.append(res.points[0])
        for s, who in getattr(hero.core, "fp_ident", {}).items():
            ids[(type(house[s - 1]).__name__, who)] += 1
    return pts, ids


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 480
    seeds = list(range(5000, 5000 + n)); W = 16
    for label, path, off in [("round5 sub", "contest/round5_submission_bot.py", False),
                             ("round4", "contest/round4_submission_bot.py", False),
                             ("round4 FP off", "contest/round4_submission_bot.py", True)]:
        allp = []; ids = Counter()
        with ProcessPoolExecutor(W) as ex:
            for p, i in ex.map(run, [(path, off, seeds[k::W]) for k in range(W)]):
                allp += p; ids.update(i)
        m = sum(allp) / len(allp)
        sd = (sum((x - m) ** 2 for x in allp) / (len(allp) - 1)) ** 0.5 / len(allp) ** 0.5
        print(f"{label:<14} {m:.2f} +- {sd:.2f} pts/table   misidentified house bots: {sum(ids.values())} {ids.most_common(4)}")
