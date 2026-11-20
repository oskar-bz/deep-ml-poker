"""
Checks that bots/sdk_adapter.MyBot reconstructs the simulator's canonical state from
the (lossy) SDK State object: seat, button, blinds seats, bets, contributions,
active seats, raise bounds and the chips-added history.

    python tests/test_adapter_fidelity.py
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.apex_bot import ApexBot  # noqa: E402
from bots.sdk_adapter import MyBot  # noqa: E402
from bots.sdk_field import StarterBot, StarterAggroBot  # noqa: E402
from pokerbot_sdk import SDKBotDriver  # noqa: E402
from poker_engine.simulator import SitAndGoTable  # noqa: E402

KEYS = ("hero_idx", "button_idx", "sb_seat", "bb_seat", "street", "to_call", "pot",
        "street_bets", "hand_contributions", "active_seats", "min_raise_add",
        "max_raise_add", "action_history", "hole_cards", "board")
mismatch: Counter = Counter()
examples = {}
checked = 0


class CheckingCore(ApexBot):
    truth = None

    def act(self, d):
        global checked
        t = CheckingCore.truth
        checked += 1
        for k in KEYS:
            a, b = d[k], t[k]
            if k == "action_history":
                a = [tuple(x) for x in a]
                b = [tuple(x) for x in b]
            if k in ("street_bets", "hand_contributions", "active_seats"):
                a, b = list(a)[:8], list(b)[:8]
            if k == "hole_cards":
                a, b = tuple(a), tuple(b)
            if a != b:
                mismatch[k] += 1
                examples.setdefault(k, (a, b))
        return super().act(d)


class TruthDriver(SDKBotDriver):
    def act(self, d):
        CheckingCore.truth = d
        return super().act(d)


def main() -> None:
    rnd = random.Random(3)
    for t in range(40):
        bots = [TruthDriver(MyBot(CheckingCore()), "hero")]
        bots += [SDKBotDriver(StarterAggroBot(seed=t) if i % 2 else StarterBot(), f"o{i}") for i in range(4)]
        bots += [ApexBot() for _ in range(3)]
        rnd.shuffle(bots)
        SitAndGoTable(bots, seed=t).play_tournament()
    print(f"checked {checked} decisions")
    for k in KEYS:
        if mismatch[k]:
            print(f"  MISMATCH {k}: {mismatch[k]}  e.g. got={examples[k][0]} want={examples[k][1]}")
    if not mismatch:
        print("  all fields reconstructed exactly")


if __name__ == "__main__":
    main()
