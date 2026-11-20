"""Hero chip flow by category vs v2 models (compare with tools/r1_analyze.py real numbers)."""
import sys, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
from tools.calib import _load


def run(seeds):
    from poker_engine.simulator import SitAndGoTable
    from pokerbot_sdk import SDKBotDriver
    from bots.sdk_field import R2ModelBot
    mod = _load("contest/round1_submission_bot.py")
    flow = defaultdict(float)

    class Spy(SDKBotDriver):
        def on_hand_end(self, idx, summ):
            super().on_hand_end(idx, summ)
            bb = summ.get("big_blind") or summ.get("bb") or self._bb
            d = (summ["ending_stacks"][idx] - summ["starting_stacks"][idx]) / bb
            if summ["starting_stacks"][idx] <= 0:
                return
            my = [a for a in summ["action_history"] if a[1] == idx and a[2] not in ("sb", "bb")]
            if not my:
                return
            allin = any(a[3] >= 0 and False for a in my) or summ["ending_stacks"][idx] == 0 or \
                abs(summ["ending_stacks"][idx] - summ["starting_stacks"][idx]) >= 0.95 * summ["starting_stacks"][idx]
            maxst = max(a[0] for a in summ["action_history"])
            if allin:
                tag = "we_allin~"
            elif my[0][0] == 0 and my[0][2] == "fold":
                tag = "pre_fold"
            elif maxst == 0:
                tag = "pre_raise_won/lost" if any(a[2] == "raise" for a in my) else "other"
            elif max(a[0] for a in my) >= 1:
                tag = "postflop"
            else:
                tag = "other"
            flow[tag] += d; flow[tag + "_n"] += 1

        def act(self, state):
            self._bb = state.get("big_blind", 20)
            return super().act(state)

    for sd in seeds:
        rng = random.Random(sd)
        bots = [Spy(mod.MyBot(), "hero")]
        while len(bots) < 8:
            b = R2ModelBot(None, rng.randrange(1 << 30)); bots.append(SDKBotDriver(b, b.name))
        SitAndGoTable(bots, seed=sd).play_tournament()
    return dict(flow), len(seeds)


if __name__ == "__main__":
    W = 16; seeds = list(range(9000, 9000 + int(sys.argv[1]) if len(sys.argv) > 1 else 9480))
    tot = defaultdict(float); T = 0
    with ProcessPoolExecutor(W) as ex:
        for f, n in ex.map(run, [seeds[i::W] for i in range(W)]):
            T += n
            for k, v in f.items(): tot[k] += v
    for k in sorted(k for k in tot if not k.endswith("_n")):
        print(f"{k:<20}{tot[k]/T:8.2f} bb/table  {tot[k+'_n']/T:6.1f} hands/table  {tot[k]/max(1,tot[k+'_n']):+6.2f}/hand")
