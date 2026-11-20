"""Opponent fingerprinting: naive-Bayes over bet-size / action signatures of known bots.

Each candidate bot has categorical distributions per feature (OPEN, UNOP, X3B, BET). An extra "unknown"
class with a flat likelihood guards against changed or new bots. A seat is identified once the best bot
beats every other class (including unknown) by FP_MARGIN nats and FP_MIN_OBS informative observations exist.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

FP_EPS = 0.002          # likelihood of a value a bot never showed in the data
FP_UNKNOWN_P = 0.06     # flat per-observation likelihood of the "unknown" class
FP_MARGIN = 6.0         # log-likelihood margin (nats) needed to identify
FP_MIN_OBS = 4
FP_MIN_SIZED = 2      # of which bet-size observations (OPEN/X3B/BET): limps/folds alone never identify


class Fingerprinter:
    def __init__(self, signatures: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        self.sig = signatures
        self.bots: List[str] = list(signatures)
        self.ll: Dict[int, List[float]] = {}
        self.unk: Dict[int, float] = {}
        self.nobs: Dict[int, int] = {}
        self.nsized: Dict[int, int] = {}
        self.ident: Dict[int, Optional[str]] = {}

    def reset(self) -> None:
        self.ll.clear(); self.unk.clear(); self.nobs.clear(); self.nsized.clear(); self.ident.clear()

    def observe(self, seat: int, feat: str, val) -> None:
        if seat in self.ident and self.ident[seat] is not None:
            return
        key = str(val)
        ll = self.ll.setdefault(seat, [0.0] * len(self.bots))
        for i, b in enumerate(self.bots):
            table = self.sig[b].get(feat)
            p = table.get(key, FP_EPS) if table else FP_EPS
            ll[i] += math.log(max(FP_EPS, p))
        if feat == "UNOP":   # first actions say little: "unknown" = population average, not a flat 0.06
            p_unk = sum(self.sig[b].get("UNOP", {}).get(key, FP_EPS) for b in self.bots) / max(1, len(self.bots))
        else:
            p_unk = FP_UNKNOWN_P
        self.unk[seat] = self.unk.get(seat, 0.0) + math.log(max(FP_EPS, p_unk))
        if feat != "UNOP" or key in ("C", "R", "A"):
            self.nobs[seat] = self.nobs.get(seat, 0) + 1
        if feat != "UNOP":
            self.nsized[seat] = self.nsized.get(seat, 0) + 1

    def identify(self, seat: int) -> Optional[str]:
        if self.ident.get(seat):
            return self.ident[seat]
        ll = self.ll.get(seat)
        if not ll or self.nobs.get(seat, 0) < FP_MIN_OBS or self.nsized.get(seat, 0) < FP_MIN_SIZED:
            return None
        # no "one seat per bot" exclusion: copies / changed bots make it unsafe (tested on round-3 data)
        order = sorted(range(len(ll)), key=lambda i: -ll[i])
        best = ll[order[0]]
        second = max(ll[order[1]] if len(order) > 1 else -1e9, self.unk.get(seat, -1e9))
        if best - second >= FP_MARGIN:
            self.ident[seat] = self.bots[order[0]]
            return self.ident[seat]
        return None


def fp_round(x: float, step: float, nd: int) -> float:
    return round(round(x / step) * step, nd)


def features_from_history(history, starting_stacks):
    """(seat, feature, value) from our core's canonical history (st, seat, act, chips_added); must match
    tools/build_profiles.features_from_hand on the replay format."""
    out = []
    bb = next((amt for st, s, a, amt in history if st == 0 and a == "bb"), 0)
    bb_seat = next((s for st, s, a, amt in history if st == 0 and a == "bb"), -1)
    if bb <= 0:
        return out
    commit_hand = {}
    street_commit = {}
    cur = -1
    nr = 0
    open_amt = 0
    pot_prev = 0
    pot_street = 0
    bet_on_street = False
    for st, s, a, amt in history:
        if st != cur:
            pot_prev += pot_street
            cur = st; street_commit = {}; pot_street = 0; bet_on_street = False
        before = street_commit.get(s, 0)
        if a in ("sb", "bb"):
            street_commit[s] = before + amt; pot_street += amt
            commit_hand[s] = commit_hand.get(s, 0) + amt
            continue
        add = amt if a in ("call", "raise") else 0
        after = before + add
        allin = starting_stacks and s < len(starting_stacks) and commit_hand.get(s, 0) + add >= starting_stacks[s] > 0
        if st == 0:
            if nr == 0 and s != bb_seat:
                v = {"fold": "F", "check": "F", "call": "C"}.get(a, "A" if allin else "R")
                out.append((s, "UNOP", v))
                if a == "raise" and not allin:
                    out.append((s, "OPEN", fp_round(after / bb, 0.05, 2)))
            if nr == 1 and a == "raise" and not allin and open_amt > 0:
                out.append((s, "X3B", fp_round(after / open_amt, 0.1, 1)))
            if a == "raise" or (allin and a == "call" and False):
                if nr == 0:
                    open_amt = after
                nr += 1
        else:
            if a == "raise" and not bet_on_street and not allin:
                out.append((s, "BET", fp_round(add / max(1, pot_prev), 0.02, 2)))
            if a == "raise":
                bet_on_street = True
        street_commit[s] = after
        pot_street += add
        commit_hand[s] = commit_hand.get(s, 0) + add
    return out
