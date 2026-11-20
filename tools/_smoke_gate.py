"""Run tables with a given core config and report any exception raised inside core.act / on_hand_end."""
import random, sys, traceback
from collections import Counter
sys.path.insert(0, ".")
from bots.apex_plus import ApexPlus
from bots.sdk_adapter import MyBot
from pokerbot_sdk import SDKBotDriver
from poker_engine.simulator import SitAndGoTable
import bots.sdk_field as F
P = dict(P_RESTEAL_ON=1, P_W3_ON=1, P_XJ_ON=1, P_STAB_ON=1, P_STAB_FRAC=0.5, P_XR_ON=1, P_XR_MARGIN=1.5,
         P_XR_PRIOR=0.7, P_D3_ON=1, P_X4_ON=1, P_W4_ON=1, P_D3_GATE=1)
errs = Counter(); first = {}
class Core(type("C", (ApexPlus,), P)):
    def act(self, d):
        try:
            return super().act(d)
        except Exception as e:
            k = repr(e)[:120]; errs["act " + k] += 1; first.setdefault(k, traceback.format_exc()); raise
    def on_hand_end(self, hero_idx, summ):
        try:
            return super().on_hand_end(hero_idx, summ)
        except Exception as e:
            k = repr(e)[:120]; errs["end " + k] += 1; first.setdefault(k, traceback.format_exc()); raise
kinds = [F.R2ModelBot, F.Adv3BetBot, F.Adv4BetBot, F.AdvJamBot, F.AdvCallerBot, F.AdvRaiseBot]
fires = Counter()
for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    random.seed(sd)
    core = Core()
    for nm_ in ("_w4_try", "_d3_try", "_x4_try", "_xr_try"):
        f = getattr(core, nm_)
        def wrap(*a, _f=f, _n=nm_):
            r = _f(*a)
            if r is not None: fires[_n] += 1
            return r
        setattr(core, nm_, wrap)
    opp = [random.choice(kinds)(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json") for _ in range(7)]
    SitAndGoTable([SDKBotDriver(MyBot(core), "h")] + [SDKBotDriver(o, o.name) for o in opp], seed=sd).play_tournament()
print("exceptions:", dict(errs) or "none")
for k, tb in list(first.items())[:3]: print(tb)
print("feature fires:", dict(fires))
