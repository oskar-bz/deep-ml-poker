import random, sys
from collections import Counter
sys.path.insert(0, ".")
from poker_engine.simulator import SitAndGoTable
from bots.apex_plus import ApexPlus
from bots.sdk_adapter import MyBot
from pokerbot_sdk import SDKBotDriver
import bots.sdk_field as F
P = dict(P_RESTEAL_ON=1, P_W3_ON=1, P_XJ_ON=1, P_STAB_ON=1, P_STAB_FRAC=0.5, P_XR_ON=1, P_XR_MARGIN=1.5, P_XR_PRIOR=0.7, P_X4_ON=1)
fires = Counter(); f4 = Counter(); f4f = Counter(); obsmed = {}
class Core(type("C", (ApexPlus,), P)):
    def _x4_try(self, state, hero_idx, hero_stack, hand_num):
        r = super()._x4_try(state, hero_idx, hero_stack, hand_num)
        if r is not None:
            v = [s for st, s, a, _ in state["action_history"] if st == 0 and a == "raise"][1]
            fires[(self._names[v], "jam" if r[1] >= hero_stack else "4b")] += 1
        return r
for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 150):
    random.seed(sd)
    core = Core()
    opp = [F.R2ModelBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json") for _ in range(7)]
    core._names = ["HERO"] + [o.name for o in opp]
    SitAndGoTable([SDKBotDriver(MyBot(core), "hero")] + [SDKBotDriver(o, o.name) for o in opp], seed=sd).play_tournament()
    for s in range(1, 8):
        f4[core._names[s]] += core.f4_n[s]; f4f[core._names[s]] += core.f4_fold[s]
        if core.tb_deep[s]: obsmed.setdefault(core._names[s], []).extend(core.tb_deep[s])
print("fires:", fires.most_common())
for n in f4:
    L = sorted(obsmed.get(n, [0])); print(f"{n:<16} faced4 {f4[n]:4d} fold {f4f[n]/max(1,f4[n]):4.0%}  observed deep-3b median top {L[len(L)//2]:.0%} n{len(L)}")
