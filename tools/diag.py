"""
Leak finder: plays a hero (ApexBot by default) in a field and attributes chip results
(in big blinds) to the hero's line in each hand.

    python tools/diag.py --field llm --tables 400
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import FIELDS, _registry  # noqa: E402
from bots.apex_bot import ApexBot  # noqa: E402
from poker_engine.simulator import SitAndGoTable  # noqa: E402


class DiagBot(ApexBot):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self.line = []
        self.seat = 0
        self.info = {}

    def on_table_start(self, seat_idx, starting_stack):
        super().on_table_start(seat_idx, starting_stack)
        self.seat = seat_idx

    def act(self, d):
        r = super().act(d)
        st = d["street"]
        bb = d["big_blind"]
        if not self.line:
            self.info = {"bb": bb, "alive": sum(1 for s in d["starting_stacks"] if s > 0),
                         "eff": d["starting_stacks"][d["hero_idx"]] / bb}
        kind = r[0]
        if kind == "raise" and r[1] >= d["stacks"][d["hero_idx"]]:
            kind = "allin"
        if kind == "call" and d["to_call"] >= d["stacks"][d["hero_idx"]]:
            kind = "callallin"
        facing = "bet" if d["to_call"] > 0 else "open"
        self.line.append(f"{'PFTR'[st]}:{facing}-{kind}")
        return r

    def on_hand_end(self, hero_idx, summary):
        super().on_hand_end(hero_idx, summary)
        if self.line:
            s = self.seat
            delta = summary["ending_stacks"][s] - summary["starting_stacks"][s]
            bb = self.info["bb"]
            eff = self.info["eff"]
            depth = "short<=12" if eff <= 12 else ("mid<=30" if eff <= 30 else "deep")
            pre = [x for x in self.line if x.startswith("P")]
            post = [x for x in self.line if not x.startswith("P")]
            key_pre = pre[-1] if pre else "-"
            showdown = s in summary.get("showdown_hands", {})
            self.sink["pre:" + key_pre].append(delta / bb)
            self.sink["depth:" + depth + " " + key_pre].append(delta / bb)
            if post:
                last = post[-1]
                self.sink["post_last:" + last + (" SD" if showdown else "")].append(delta / bb)
        self.line = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="llm")
    ap.add_argument("--tables", type=int, default=300)
    a = ap.parse_args()
    reg = _registry()
    sink = defaultdict(list)
    pts = []
    for sd in range(a.tables):
        rng = random.Random(sd)
        hero = DiagBot(sink)
        bots = [hero] + [reg[n]() if n != "@hero" else ApexBot() for n in FIELDS[a.field]]
        order = list(range(8))
        rng.shuffle(order)
        seated = [bots[i] for i in order]
        hero.seat = order.index(0)
        res = SitAndGoTable(seated, seed=sd, enforce_1ms=False).play_tournament()
        pts.append(res.points[order.index(0)])
    print(f"field {a.field}: {sum(pts) / len(pts):.3f} pts over {len(pts)} tables")
    rows = sorted(sink.items(), key=lambda kv: sum(kv[1]))
    print(f"{'line':<45}{'n':>7}{'sum bb':>10}{'bb/hand':>9}")
    for k, v in rows:
        if len(v) >= 30:
            print(f"{k:<45}{len(v):7d}{sum(v):10.0f}{sum(v) / len(v):9.2f}")


if __name__ == "__main__":
    main()
