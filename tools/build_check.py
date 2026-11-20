"""Exceptions + unloaded think time of a built submission file against all adversary kinds.
python tools/build_check.py FILE [tables]"""
import importlib.util, random, sys, time
sys.path.insert(0, ".")
from pokerbot_sdk import SDKBotDriver
from poker_engine.simulator import SitAndGoTable
import bots.sdk_field as F
spec = importlib.util.spec_from_file_location("built", sys.argv[1]); S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
errs = []; lat = []
class Core(S.ContestCore):
    def act(self, d):
        try:
            return super().act(d)
        except Exception as e:
            errs.append(repr(e)); raise
    def on_hand_end(self, i, s):
        try:
            return super().on_hand_end(i, s)
        except Exception as e:
            errs.append(repr(e)); raise
class Timed(SDKBotDriver):
    def act(self, st):
        t = time.perf_counter(); r = super().act(st); lat.append((time.perf_counter() - t) * 1000); return r
kinds = [F.R2ModelBot, F.Adv3BetBot, F.Adv4BetBot, F.AdvJamBot, F.AdvCallerBot, F.AdvRaiseBot]
for sd in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
    random.seed(sd)
    bot = S.MyBot(); bot.core = Core()
    opp = [random.choice(kinds)(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json") for _ in range(7)]
    SitAndGoTable([Timed(bot, "h")] + [SDKBotDriver(o, o.name) for o in opp], seed=sd).play_tournament()
lat.sort()
print("exceptions:", len(errs), errs[:3], "| decisions", len(lat),
      "| avg %.3f ms  p99.9 %.3f ms  max %.3f ms  >0.9ms: %d" % (sum(lat) / len(lat), lat[int(len(lat) * 0.999)], lat[-1],
                                                              sum(x > 0.9 for x in lat)))
