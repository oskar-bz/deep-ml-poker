"""Feed the bot end-of-hand results in the real platform's format (as seen in the round-2 probe) and compare
the learned counters with the local-format run. Shadow bot sees identical states."""
import random, sys
sys.path.insert(0, ".")
from bots.apex_plus import ApexPlus
from bots.sdk_adapter import MyBot
from pokerbot_sdk import SDKBotDriver, _sdk_history, card_str
from poker_engine.simulator import SitAndGoTable
import bots.sdk_field as F
P = dict(P_RESTEAL_ON=1, P_W3_ON=1, P_XJ_ON=1, P_STAB_ON=1, P_STAB_FRAC=0.5, P_XR_ON=1, P_XR_MARGIN=1.5,
         P_XR_PRIOR=0.7, P_D3_ON=1, P_D3_GATE=1, P_X4_ON=1, P_W4_ON=1)
C = type("C", (ApexPlus,), P)


def platform_result(summary, hero):
    start = summary["starting_stacks"]; end = summary["ending_stacks"]
    acts = []
    bet_on = set()
    for e in _sdk_history(summary.get("action_history", [])):
        a = e["action"]
        if a in ("small_blind", "big_blind"):
            a = "blind"
        elif a == "raise" and e["street"] != "preflop" and e["street"] not in bet_on:
            a = "bet"
        if e["action"] == "raise":
            bet_on.add(e["street"])
        acts.append({"street": e["street"], "seat": e["seat"], "action": a, "amount": e["amount"], "all_in": False})
    hole = [[card_str(c) for c in summary.get("all_hole", {}).get(i, ())] for i in range(len(start))]
    return {"hand_id": summary.get("hand_number"), "start_stacks": list(start), "hole": hole,
            "board": [card_str(c) for c in summary.get("board_dealt", [])], "actions": acts,
            "shown": {str(s): [card_str(c) for c in cs] for s, cs in summary.get("showdown_hands", {}).items()},
            "deltas": [e - s for s, e in zip(start, end)], "seat": hero}


class Twin(SDKBotDriver):
    def __init__(self, bot, shadow):
        super().__init__(bot, "hero"); self.shadow = shadow
    def act(self, state):
        from pokerbot_sdk import State
        self.shadow.act(State(state))
        return super().act(state)
    def on_hand_end(self, idx, summ):
        super().on_hand_end(idx, summ)
        self.shadow.on_hand_end(platform_result(summ, idx))


keys = ["faced_jam", "called_jam", "faced_3bet", "folded_3bet", "fourbet_count", "f4_n", "f4_fold", "faced_steal", "threebet_count"]
diff = 0; tot = {k: [0, 0] for k in keys + ["tb_deep", "tb4"]}
for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 15):
    random.seed(sd)
    main_core, shadow_core = C(), C()
    opp = [F.R2ModelBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json") for _ in range(7)]
    SitAndGoTable([Twin(MyBot(main_core), MyBot(shadow_core))] + [SDKBotDriver(o, o.name) for o in opp], seed=sd).play_tournament()
    for k in keys:
        a, b = getattr(main_core, k), getattr(shadow_core, k)
        tot[k][0] += sum(a); tot[k][1] += sum(b)
    for k in ("tb_deep", "tb4"):
        tot[k][0] += sum(len(x) for x in getattr(main_core, k)); tot[k][1] += sum(len(x) for x in getattr(shadow_core, k))
for k, (a, b) in tot.items():
    print(f"{k:<16} local {a:6d}  platform-format {b:6d}  {'OK' if abs(a - b) <= max(2, 0.03 * a) else 'MISMATCH'}")
