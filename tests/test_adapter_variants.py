"""
Robustness of bots/sdk_adapter.MyBot against plausible variations of the real SDK's
undocumented details. For each variant, reports reconstruction mismatches and the
hero's average points vs a starter-heavy field.

    python tests/test_adapter_variants.py
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pokerbot_sdk as sdk  # noqa: E402
from bots.apex_bot import ApexBot  # noqa: E402
from bots.sdk_adapter import MyBot  # noqa: E402
from bots.sdk_field import StarterBot, StarterAggroBot  # noqa: E402
from poker_engine.simulator import SitAndGoTable  # noqa: E402

KEYS = ("hero_idx", "button_idx", "sb_seat", "bb_seat", "street_bets", "hand_contributions",
        "active_seats", "min_raise_add", "action_history")


class Hidden:
    """Proxy exposing only some attributes of a State."""

    def __init__(self, st, drop, hist_fn=None):
        self._st = st
        self._drop = drop
        self._hist_fn = hist_fn

    def __getattr__(self, k):
        if k in self._drop:
            raise AttributeError(k)
        if k == "history" and self._hist_fn:
            return self._hist_fn(self._st.history)
        return getattr(self._st, k)


def hist_added(h):
    """Convert raise amounts back to chips added."""
    out, sb, cur = [], {}, None
    for e in h:
        if e["street"] != cur:
            cur, sb = e["street"], {}
        e = dict(e)
        if e["action"] == "raise":
            add = e["amount"] - sb.get(e["seat"], 0)
            sb[e["seat"]] = e["amount"]
            e["amount"] = add
        else:
            sb[e["seat"]] = sb.get(e["seat"], 0) + e["amount"]
        out.append(e)
    return out


def hist_renamed(h):
    names = {"raise": "bet", "small_blind": "post_sb", "big_blind": "post_bb"}
    return [dict(e, action=names.get(e["action"], e["action"]).upper()) for e in h]


def hist_tuples(h):
    return [(e["street"], e["seat"], e["action"], e["amount"]) for e in h]


def hist_platform(h):
    """Observed on the real platform: both blinds logged as "blind", extra all_in key."""
    out = []
    for e in h:
        a = "blind" if e["action"] in ("small_blind", "big_blind") else e["action"]
        out.append(dict(e, action=a, all_in=False))
    return out


VARIANTS = {
    "platform blinds": ((), hist_platform),
    "platform+no seat/btn": (("seat", "button", "hand_number", "stack"), hist_platform),
    "documented+guessed": ((), None),
    "no seat": (("seat",), None),
    "no button": (("button",), None),
    "no hand_number": (("hand_number",), None),
    "no seat/button/hand": (("seat", "button", "hand_number"), None),
    "amounts=chips added": ((), hist_added),
    "renamed actions": ((), hist_renamed),
    "tuple history": ((), hist_tuples),
    "bare minimum": (("seat", "button", "hand_number", "stack"), hist_added),
}


def run_variant(name, drop, hist_fn, tables=40):
    mismatch: Counter = Counter()
    checked = [0]
    pts = []

    class CheckingCore(ApexBot):
        truth = None

        def act(self, d):
            checked[0] += 1
            t = CheckingCore.truth
            for k in KEYS:
                a, b = d[k], t[k]
                if k == "action_history":
                    a, b = [tuple(x) for x in a], [tuple(x) for x in b]
                elif isinstance(a, list):
                    a, b = list(a)[:8], list(b)[:8]
                if a != b:
                    mismatch[k] += 1
            return super().act(d)

    class VariantDriver(sdk.SDKBotDriver):
        def act(self, d):
            CheckingCore.truth = d
            st = sdk.State(d)
            a = self.bot.act(Hidden(st, drop, hist_fn))
            # reuse parent conversion by faking a bot that returns `a`
            inner = self.bot
            self.bot = type("Fixed", (), {"act": lambda _s, _st: a})()
            try:
                return super().act(d)
            finally:
                self.bot = inner

    rnd = random.Random(5)
    for t in range(tables):
        hero = VariantDriver(MyBot(CheckingCore()), "hero")
        bots = [hero] + [sdk.SDKBotDriver(StarterAggroBot(seed=t) if i % 2 else StarterBot(), f"o{i}")
                         for i in range(4)] + [ApexBot() for _ in range(3)]
        order = list(range(8))
        rnd.shuffle(order)
        res = SitAndGoTable([bots[i] for i in order], seed=t).play_tournament()
        pts.append(res.points[order.index(0)])
    bad = ", ".join(f"{k}:{v}" for k, v in mismatch.items()) or "exact"
    print(f"{name:<22} decisions {checked[0]:5d}  pts {sum(pts) / len(pts):5.2f}  mismatches: {bad}")


def main():
    for name, (drop, fn) in VARIANTS.items():
        run_variant(name, drop, fn)


if __name__ == "__main__":
    main()
