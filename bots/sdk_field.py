"""
Opponents written against the real SDK API, modelled on what the contest page hands
every entrant (reference/starter_module.js). A large share of the real field will be
these bots or light edits of them, so they are the most important thing to beat.

- StarterBot:         the contest's STARTER_BOT, verbatim.
- TrainedExampleBot:  the TRAINED_BOT_EXAMPLE policy. Its net regresses Monte Carlo
                      equity vs `opponents` random hands; we use the true value
                      (table preflop, small MC postflop) instead of a fitted net.
- Starter variants:   typical "LLM-tweaked starter" edits (tighter, more aggressive,
                      position/pot-odds aware), to keep us from overfitting to one bot.
"""

from __future__ import annotations

import random

from pokerbot_sdk import Action, Bot, equity, parse_card


class StarterBot(Bot):
    def act(self, state):
        if state.street == "preflop":
            strength = 0.2 + 0.6 * state.preflop_rank()
        else:
            strength = {0: 0.25, 1: 0.45, 2: 0.65}.get(state.made_hand()["category"], 0.85)

        if state.stack_m < 5:
            if strength > 0.5:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()

        if state.to_call == 0:
            if strength > 0.6 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + state.pot // 2)
            return Action.check()

        if strength > 0.75 and "raise" in state.legal:
            return Action.raise_to(state.min_raise * 2)
        if strength > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()


class TrainedExampleBot(Bot):
    def __init__(self, samples: int = 60, seed: int = 0) -> None:
        self.samples = samples
        self.rng = random.Random(seed)

    def _eq(self, state) -> float:
        if state.street == "preflop":
            return state.preflop_equity()
        return equity(state.hole, state.board, max(1, state.opponents), self.samples, self.rng)

    def act(self, state):
        eq = self._eq(state)
        if state.stack_m < 5:
            if eq > 0.5:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()
        if state.to_call == 0:
            if eq > 0.6 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + state.pot // 2)
            return Action.check()
        if eq > 0.75 and "raise" in state.legal:
            return Action.raise_to(state.min_raise * 2)
        if eq > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()


class StarterTightBot(Bot):
    """Starter with tighter preflop and a wider short-stack shove (typical first edit)."""

    def act(self, state):
        if state.street == "preflop":
            strength = 0.1 + 0.8 * state.preflop_rank()
        else:
            cat = state.made_hand()["category"]
            strength = {0: 0.2, 1: 0.5, 2: 0.7}.get(cat, 0.9)

        if state.stack_m < 8:
            if strength > 0.55:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()

        if state.to_call == 0:
            if strength > 0.65 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + max(state.big_blind * 3, state.pot * 2 // 3))
            return Action.check()

        if strength > 0.8 and "raise" in state.legal:
            return Action.raise_to(max(state.min_raise, state.current_bet * 3))
        if strength > state.pot_odds() + 0.12:
            return Action.call()
        return Action.fold()


class StarterAggroBot(Bot):
    """Starter edited to open-raise and c-bet a lot (typical 'be aggressive' edit)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, state):
        bb = state.big_blind
        if state.street == "preflop":
            r = state.preflop_rank()
            if state.stack_m < 6:
                if r > 0.45:
                    return Action.all_in()
                return Action.check() if state.to_call == 0 else Action.fold()
            if state.current_bet <= bb and "raise" in state.legal and r > 0.55:
                return Action.raise_to(max(state.min_raise, 3 * bb))
            if r > 0.9 and "raise" in state.legal:
                return Action.raise_to(max(state.min_raise, state.current_bet * 3))
            if state.to_call == 0:
                return Action.check()
            if r > 0.4 + state.pot_odds() * 0.6:
                return Action.call()
            return Action.fold()

        cat = state.made_hand()["category"]
        if state.to_call == 0:
            if "raise" in state.legal and (cat >= 1 or self.rng.random() < 0.45):
                return Action.raise_to(state.current_bet + max(bb, state.pot * 2 // 3))
            return Action.check()
        if cat >= 2 and "raise" in state.legal:
            return Action.raise_to(max(state.min_raise, state.current_bet * 3))
        strength = {0: 0.2, 1: 0.5}.get(cat, 0.8)
        if strength > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()


class EquityPotOddsBot(Bot):
    """A careful 'math' bot: preflop rank by position, postflop equity vs pot odds."""

    def __init__(self, samples: int = 40, seed: int = 0) -> None:
        self.samples = samples
        self.rng = random.Random(seed)

    def act(self, state):
        bb = state.big_blind
        if state.street == "preflop":
            r = state.preflop_rank()
            eq = state.preflop_equity()
            if state.stack_m < 7:
                need = 0.55 if state.to_call > 0 else 0.45
                if r > need:
                    return Action.all_in()
                return Action.check() if state.to_call == 0 else Action.fold()
            if state.current_bet <= bb:
                if r > 0.7 and "raise" in state.legal:
                    return Action.raise_to(max(state.min_raise, int(2.5 * bb)))
                if state.to_call == 0:
                    return Action.check()
                return Action.fold()
            if r > 0.93 and "raise" in state.legal:
                return Action.raise_to(max(state.min_raise, state.current_bet * 3))
            if eq > state.pot_odds() + 0.08 and r > 0.75:
                return Action.call()
            return Action.check() if state.to_call == 0 else Action.fold()

        eq = equity(state.hole, state.board, max(1, state.opponents), self.samples, self.rng)
        if state.to_call == 0:
            if eq > 0.62 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + state.pot * 3 // 4)
            return Action.check()
        if eq > 0.82 and "raise" in state.legal:
            return Action.raise_to(max(state.min_raise, state.current_bet * 3))
        if eq > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()


class StarterShortBot(Bot):
    """The shorter starter from the official problem statement (min-raises, never bets postflop)."""

    def act(self, state):
        strength = 0.2 + 0.6 * state.preflop_rank() if state.street == "preflop" else 0.5
        if state.stack_m < 5:
            if strength > 0.5:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()
        if strength > 0.6 and "raise" in state.legal:
            return Action.raise_to(state.min_raise)
        return Action.call() if state.to_call == 0 or strength > state.pot_odds() + 0.05 else Action.fold()


class LooseRaiserBot(Bot):
    """Entrant-style bot modelled on the top demo entries (replay stats): open-raises
    a wide range to ~2.5bb, shoves short, plays postflop mostly passively by made hand."""

    def __init__(self, open_rank: float = 0.5, cbet: float = 0.3, seed: int = 0) -> None:
        self.open_rank = open_rank
        self.cbet = cbet
        self.rng = random.Random(seed)

    def act(self, state):
        bb = state.big_blind
        if state.street == "preflop":
            r = state.preflop_rank()
            if state.stack_m < 7:
                if r > 0.55 or (state.to_call == 0 and r > 0.35):
                    return Action.all_in()
                return Action.check() if state.to_call == 0 else Action.fold()
            if state.current_bet <= bb:
                if r > self.open_rank and "raise" in state.legal:
                    return Action.raise_to(max(state.min_raise, int(2.5 * bb)))
                return Action.check() if state.to_call == 0 else Action.fold()
            if r > 0.9 and "raise" in state.legal:
                return Action.raise_to(max(state.min_raise, state.current_bet * 3))
            if r > 0.6 + 0.6 * state.pot_odds() - 0.2:
                return Action.call()
            return Action.check() if state.to_call == 0 else Action.fold()
        cat = state.made_hand()["category"]
        if state.to_call == 0:
            if "raise" in state.legal and (cat >= 2 or (cat == 1 and self.rng.random() < 0.5)
                                           or self.rng.random() < self.cbet):
                return Action.raise_to(state.current_bet + max(bb, state.pot // 2))
            return Action.check()
        if cat >= 3 and "raise" in state.legal:
            return Action.raise_to(max(state.min_raise, state.current_bet * 3))
        need = {0: 0.15, 1: 0.4, 2: 0.6}.get(cat, 0.9)
        if need > state.pot_odds() + 0.1 or cat >= 2:
            return Action.call()
        return Action.fold()


class EntrantModelBot(LooseRaiserBot):
    """Preflop model of a real entrant, fitted from their public demo replays
    (tools/replay_reactions.py): open frequency in unopened pots and fold rate
    after being re-raised. Postflop as LooseRaiserBot."""

    def __init__(self, open_freq: float, fold3b: float, cbet: float = 0.3, seed: int = 0) -> None:
        super().__init__(open_rank=1.0 - open_freq, cbet=cbet, seed=seed)
        self.fold3b = fold3b

    def act(self, state):
        if state.street == "preflop" and state.stack_m >= 7:
            seat = getattr(state, "seat", None)
            bb = state.big_blind
            if seat is not None and state.bets[seat] > bb and state.to_call > 0:
                # we opened and got re-raised
                r = state.preflop_rank()
                if r > 0.93:
                    return Action.all_in() if r > 0.97 else Action.call()
                if self.rng.random() < self.fold3b:
                    return Action.fold()
                return Action.call()
        return super().act(state)


# ---------------------------------------------------------------------------------------------
# Round-1 empirical models (tools/fit_r1_models.py -> data/r1_models.json)
_R1 = None
_R1_TOP = None


def _r1_load():
    global _R1, _R1_TOP
    if _R1 is None:
        import json, os
        from poker_engine.preflop_multiway import MULTIWAY_EQ
        _R1 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "r1_models.json"), encoding="utf-8"))
        off = min(k[0] for k in MULTIWAY_EQ)
        order = sorted(MULTIWAY_EQ, key=lambda k: -MULTIWAY_EQ[k][0])
        _R1_TOP, cum = {}, 0
        for k in order:
            w = 6 if k[0] == k[1] else (4 if k[2] else 12)
            _R1_TOP[(k[0] - off, k[1] - off, k[2])] = (cum + w / 2) / 1326
            cum += w
    return _R1, _R1_TOP


_RK = "23456789TJQKA"


class R1ModelBot(Bot):
    """Plays the empirical round-1 policy of one real bot (preflop by situation/depth/decile)."""

    def __init__(self, name: str, seed: int = 0) -> None:
        models, self.top = _r1_load()
        self.m = models[name]
        self.rng = random.Random(seed)

    def _decile(self, hole):
        r = sorted((_RK.index(c[0]) for c in hole), reverse=True)
        return min(9, int(self.top[(r[0], r[1], hole[0][1] == hole[1][1] and r[0] != r[1])] * 10))

    def _made(self, hole, board):
        ranks = [c[0] for c in hole + board]
        hr = [c[0] for c in hole]
        br = [c[0] for c in board]
        if any(ranks.count(x) >= 2 for x in hr) and (sum(1 for x in set(hr) if x in br) == 2 or any(ranks.count(x) >= 3 for x in hr)):
            return 2
        if any(x in br for x in hr) or hr[0] == hr[1]:
            return 1
        return 0

    def _pass(self, state):
        return Action.check() if state.to_call == 0 else Action.fold()

    def act(self, state):
        bb = state.big_blind
        if state.street == "preflop":
            nr = sum(1 for h in state.history if h["street"] == "preflop" and h["action"] in ("raise", "all_in", "bet")
                     and h.get("amount", 0) > bb)
            st = state.stacks[state.seat] + state.bets[state.seat]
            d = 0 if st < 15 * bb else (1 if st < 40 * bb else 2)
            jam = nr == 1 and any(state.stacks[i] == 0 and state.bets[i] > bb for i in range(len(state.stacks)))
            key = f"{'j' if jam else min(nr, 2)}{d}"
            rows = self.m["pre"].get(key)
            if rows is None:
                return self._pass(state)
            p = rows[self._decile(state.hole)]
            x = self.rng.random() * sum(p)
            if x < p[0]:
                return self._pass(state)
            if x < p[0] + p[1]:
                return Action.call() if state.to_call > 0 else Action.check()
            if x < p[0] + p[1] + p[2] and "raise" in state.legal:
                mult = self.m["size"].get(key, 2.5 if nr == 0 else 3.0)
                tgt = int(mult * (bb if nr == 0 else state.current_bet))
                tgt = max(state.min_raise, min(state.max_raise, tgt))
                if tgt >= state.max_raise:
                    return Action.all_in()
                return Action.raise_to(tgt)
            return Action.all_in()
        mc = self._made(state.hole, state.board)
        if state.to_call == 0:
            v = self.m["post"].get(f"first{mc}")
            if v and self.rng.random() < v[0] and "raise" in state.legal:
                tgt = state.current_bet + max(bb, state.pot // 2)
                return Action.raise_to(max(state.min_raise, min(state.max_raise, tgt)))
            return Action.check()
        v = self.m["post"].get(f"face{mc}", [0.5, 0.05])
        pot_before = max(1, state.pot - state.to_call)
        frac = state.to_call / pot_before
        fm = self.m.get("fold_mult", [1.0, 1.0, 1.0])
        k = fm[0] if frac < 0.4 else (fm[1] if frac < 0.7 else fm[2])
        x = self.rng.random()
        if x < min(0.97, v[0] * k):
            return Action.fold()
        if x < v[0] + v[1] and "raise" in state.legal:
            tgt = max(state.min_raise, min(state.max_raise, state.current_bet * 3))
            return Action.all_in() if tgt >= state.max_raise else Action.raise_to(tgt)
        return Action.call()


# ---------------------------------------------------------------------------------------------
# Version-2 round-1 models (tools/fit_r1_models2.py -> data/r1_models2.json): postflop by real hand
# strength / draw / role / bet size, preflop by position; any of the 33 round-1 bots.
_R2 = {}


def _r2_load(fname="r1_models2.json"):
    if fname not in _R2:
        import json, os
        _R2[fname] = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", fname), encoding="utf-8"))
    return _R2[fname]


R2_EXCLUDE = ("Oskar Behzadpour",)
# measured round-2 fold rate of a 3-bettor facing a 4-bet: (<25 bb, 25-40 bb, 40+ bb)
F4_REAL = {"mw": (0.56, 0.80, 0.77), "Daksh Malhan": (0.04, 0.53, 0.45), "Shresth Rana": (0.14, 0.50, 0.30),
           "Henrik Gombos": (0.15, 0.66, 0.52), "Kashyap Patel": (0.04, 0.0, 0.07), "Lorenzo P.": (0.0, 0.17, 0.18),
           "Vaibhaav Tiwari": (0.0, 0.0, 0.0), "krishnah37": (0.15, 0.11, 0.02)}
# round-3 measured ratio: 3-bet rate vs OUR opens / vs others' opens
R4_TARGET_MULT = {"mw": 1.73, "Vaibhaav Tiwari": 1.73, "Shresth Rana": 1.54, "Henrik Gombos": 1.27,
                  "Lorenzo P.": 1.35, "Daksh Malhan": 1.16, "Kashyap Patel": 0.70}
R4_FIELD = ("mw", "Daksh Malhan", "Vaibhaav Tiwari", "Lorenzo P.", "Henrik Gombos", "Shresth Rana", "Kashyap Patel")
R3_FIELD = ("mw", "Daksh Malhan", "Shresth Rana", "Henrik Gombos", "Kashyap Patel", "Lorenzo P.",
            "Vaibhaav Tiwari", "krishnah37")
R2_FINALISTS = ("mw", "Shresth Rana", "Daksh Malhan", "thearly man", "Henrik Gombos", "Lorenzo P.", "Kashyap Patel",
                "Sujay", "krishnah37", "Scott Middlecamp", "Vaibhaav Tiwari", "Sam Thomas", "mohansriram",
                "Vedant Bhardwaj", "vidvuds calitis", "Elias Marlowe")


class R2ModelBot(Bot):
    def __init__(self, name=None, seed: int = 0, fname: str = "r1_models2.json") -> None:
        from tools import fit_r1_models2 as F2
        self.F2 = F2
        M = _r2_load(fname)
        self.rng = random.Random(seed)
        if name is None:
            pool = [(n, w) for n, w in M["_games"].items() if n not in R2_EXCLUDE]
            x = self.rng.random() * sum(w for _, w in pool)
            for n, w in pool:
                x -= w
                if x <= 0:
                    name = n
                    break
            else:
                name = pool[-1][0]
        self.name = name
        self.pre = M["pre"][name]; self.pre_size = M["pre_size"][name]
        self.fi = M["fi"][name]; self.fi_size = M["fi_size"][name]
        self.fa = M["fa"][name]; self.rsize = M["fa_rsize"][name]

    TARGET_MULT = R4_TARGET_MULT
    TARGET = False     # round-4 models: 3-bet our (seat 0) opens more often, as measured in round 3
    HERO_SEAT = 0
    F4_FIX = True  # pin facing-4-bet fold rates to measured round-2 values (F4_REAL)
    NOISE = 0.40   # 0 = pure threshold policy (rank within the strength cell picks the action)

    def _pick(self, p, frac=None):
        if frac is None:
            u = self.rng.random()
        else:
            # mixture (not a weighted sum): keeps u uniform so action frequencies match the data
            u = frac if self.rng.random() >= self.NOISE else self.rng.random()
        x = u * sum(p)
        for i, v in enumerate(p):
            x -= v
            if x <= 0:
                return i
        return len(p) - 1

    def _raise_to(self, state, tgt):
        tgt = int(max(state.min_raise, min(state.max_raise, tgt)))
        if tgt >= state.max_raise:
            return Action.all_in()
        return Action.raise_to(tgt)

    def _pass(self, state):
        return Action.check() if state.to_call == 0 else Action.fold()

    def act(self, state):
        F2 = self.F2
        hk = tuple(state.hole)
        if getattr(self, "_hk", None) != hk:
            self._hk = hk
            self._u = self.rng.random()
        bb = state.big_blind
        me = state.seat
        hist = state.history
        n = len(state.stacks)
        folded = {h["seat"] for h in hist if h["action"] == "fold"}
        in_hand = {h["seat"] for h in hist if h["street"] == "preflop"} | {i for i in range(n) if state.stacks[i] > 0}
        in_hand -= folded
        if state.street == "preflop":
            cur, nr, first = bb, 0, -1
            for h in hist:
                if h["street"] == "preflop" and h["action"] in ("raise", "bet", "all_in") and h.get("amount", 0) > cur:
                    cur = h["amount"]; nr += 1
                    if first < 0:
                        first = h["seat"]
            jam = nr == 1 and first >= 0 and state.stacks[first] == 0
            nk = "j" if jam else str(min(nr, 3))
            if nk == "1" and cur > 2.6 * bb:
                nk = "1L"
            d = F2.depth((state.stacks[me] + state.bets[me]) / bb)
            p = 0
            if nr <= 1:
                bbs = [h["seat"] for h in hist if h["action"] == "big_blind"]
                bb_seat = bbs[0] if bbs else me
                behind = 0
                if me != bb_seat:
                    x = me
                    for _ in range(n):
                        x = (x + 1) % n
                        if x in in_hand and x != me:
                            behind += 1
                        if x == bb_seat:
                            break
                p = F2.pos_bucket(behind)
                if nr == 1 and me == bb_seat:
                    p = 0
            rows = self.pre.get(f"{nk}{d}{p}")
            if rows is None and nk == "3":
                rows = self.pre.get(f"2{d}0")
            if rows is None:
                return self._pass(state)
            pd, pf = F2.pre_decile_frac(state.hole)
            row = rows[pd]
            tm = self.TARGET_MULT.get(self.name, 1.0) if self.TARGET else 1.0
            if tm != 1.0 and nk in ("1", "1L") and first == self.HERO_SEAT:
                r = list(row)
                agg = (r[2] + r[3]) * tm
                rest = r[0] + r[1]
                if agg < 0.98:
                    k_ = (1.0 - agg) / max(1e-9, rest)
                    row = [r[0] * k_, r[1] * k_, r[2] * tm, r[3] * tm]
            if nk == "3" and self.F4_FIX and self.name in F4_REAL:
                ff = F4_REAL[self.name][0 if d <= 1 else d - 1]
                cont = row[1:]
                tot = sum(cont) or 1.0
                # weakest hands fold first: map hand rank to fold below the ff quantile of this bot's 3-bet range
                premium = pd == 9 and pf >= 0.7          # ~top 3% never folds
                if not premium and self.rng.random() < min(1.0, ff * 1.05):
                    return self._pass(state)
                row = [0.0] + [c / tot for c in cont]
            a = self._pick(row, pf)
            if a == 0:
                return self._pass(state)
            if a == 1:
                return Action.call() if state.to_call > 0 else Action.check()
            if a == 2 and "raise" in state.legal:
                mult = self.pre_size.get(f"{nk}{d}", 2.5 if nr == 0 else 3.0)
                return self._raise_to(state, mult * (bb if nr == 0 else state.current_bet))
            return Action.all_in()
        hole_i = [parse_card(c) for c in state.hole]
        board_i = [parse_card(c) for c in state.board]
        sg = F2.strength(hole_i, board_i)
        idx = min(9, int(sg * 10)) * 2 + F2.has_draw(hole_i, board_i)
        sfrac = min(0.999, sg * 10 - min(9, int(sg * 10)))
        st = {"flop": 1, "turn": 2, "river": 3}.get(state.street, 3)
        mw = 1 if len(in_hand) > 2 else 0
        if state.to_call == 0:
            names = ("preflop", "flop", "turn", "river")
            prev_street = names[st - 1]
            last = -1
            for h in hist:
                if h["street"] == prev_street and h["action"] in ("raise", "bet", "all_in"):
                    last = h["seat"]
            role = 1 if last == me else 0
            a = self._pick(self.fi[f"{st}{role}{mw}"][idx], sfrac)
            if a == 0 or "raise" not in state.legal:
                return Action.check()
            if a == 4:
                return Action.all_in()
            frac = self.fi_size.get(str(a), {1: 0.33, 2: 0.55, 3: 0.85}[a])
            return self._raise_to(state, state.current_bet + max(bb, int(frac * state.pot)))
        cur_street = state.street
        nraise = sum(1 for h in hist if h["street"] == cur_street and h["action"] in ("raise", "bet", "all_in"))
        if any(state.stacks[i] == 0 and state.bets[i] >= state.current_bet > 0 for i in range(n) if i != me):
            sb = 5
        elif nraise >= 2:
            sb = 4
        else:
            cb = state.current_bet
            sb = F2.size_bucket(cb / max(1, state.pot - cb))
        a = self._pick(self.fa[f"{st}{sb}{mw}"][idx], sfrac)
        if a == 0:
            return Action.fold()
        if a == 1 or "raise" not in state.legal:
            return Action.call()
        if a == 3:
            return Action.all_in()
        return self._raise_to(state, self.rsize * state.current_bet)


# ---------------------------------------------------------------------------------------------
# Adversaries: plausible round-3 counter-strategies against us (built on the fitted models).
class Adv3BetBot(R2ModelBot):
    """3-bets any single (non-all-in) open with probability P3 when >= 15 bb deep; vs a 4-bet continues
    only with ~top 10% (jams them), else folds."""
    P3 = 0.45

    def act(self, state):
        if state.street == "preflop":
            bb = state.big_blind; me = state.seat
            raises = [h for h in state.history if h["street"] == "preflop" and h["action"] in ("raise", "bet", "all_in")
                      and h.get("amount", 0) > bb]
            my_raised = any(h["seat"] == me for h in raises)
            stack_bb = (state.stacks[me] + state.bets[me]) / bb
            if len(raises) == 1 and not my_raised and stack_bb >= 15 and "raise" in state.legal:
                opener = raises[0]["seat"]
                if state.stacks[opener] > 0:
                    if self.rng.random() < self.P3:
                        return self._raise_to(state, 3 * state.current_bet)
            if my_raised and len(raises) >= 3 and raises[-1]["seat"] != me:
                pd, pf = self.F2.pre_decile_frac(state.hole)
                if pd == 9:
                    return Action.all_in()
                return self._pass(state)
        return super().act(state)


class AdvCallerBot(R2ModelBot):
    """Calls any preflop all-in (open-jam or re-jam) with ~top 50% of hands."""
    def act(self, state):
        if state.street == "preflop" and state.to_call > 0:
            jammed = any(state.stacks[i] == 0 and state.bets[i] > state.big_blind
                         for i in range(len(state.stacks)) if i != state.seat)
            if jammed:
                pd, _ = self.F2.pre_decile_frac(state.hole)
                if pd >= 5:
                    return Action.call()
        return super().act(state)


class Adv4BetBot(R2ModelBot):
    """After opening and facing a 3-bet: 4-bets (jam if <= 40 bb, else 2.3x) with probability P4, any hand."""
    P4 = 0.40

    def act(self, state):
        if state.street == "preflop":
            bb = state.big_blind; me = state.seat
            raises = [h for h in state.history if h["street"] == "preflop" and h["action"] in ("raise", "bet", "all_in")
                      and h.get("amount", 0) > bb]
            if len(raises) == 2 and raises[0]["seat"] == me and raises[1]["seat"] != me and "raise" in state.legal:
                if self.rng.random() < self.P4:
                    stack_bb = (state.stacks[me] + state.bets[me]) / bb
                    if stack_bb <= 40:
                        return Action.all_in()
                    return self._raise_to(state, int(2.3 * state.current_bet))
        return super().act(state)


class AdvRaiseBot(R2ModelBot):
    """Postflop heads-up: raises any bet 3x with probability PR (flop/turn)."""
    PR = 0.30

    def act(self, state):
        if state.street in ("flop", "turn") and state.to_call > 0 and "raise" in state.legal:
            live = len({h["seat"] for h in state.history if h["street"] == "preflop"} |
                       {i for i in range(len(state.stacks)) if state.stacks[i] > 0}) - \
                   len({h["seat"] for h in state.history if h["action"] == "fold"})
            if live <= 2 and self.rng.random() < self.PR:
                return self._raise_to(state, 3 * state.current_bet)
        return super().act(state)


class AdvJamBot(Adv3BetBot):
    """Re-jams any single open with probability P3 when <= 30 bb deep (else behaves like the model)."""
    P3 = 0.35

    def act(self, state):
        if state.street == "preflop":
            bb = state.big_blind; me = state.seat
            raises = [h for h in state.history if h["street"] == "preflop" and h["action"] in ("raise", "bet", "all_in")
                      and h.get("amount", 0) > bb]
            stack_bb = (state.stacks[me] + state.bets[me]) / bb
            if len(raises) == 1 and raises[0]["seat"] != me and 8 <= stack_bb <= 30 and state.stacks[raises[0]["seat"]] > 0:
                if self.rng.random() < self.P3:
                    return Action.all_in()
        return R2ModelBot.act(self, state)



class R4ModelBot(R2ModelBot):
    """Round-3-fitted model of a round-4 finalist, targeting our seat like in round 3."""
    TARGET = True

    def __init__(self, name, seed=0):
        super().__init__(name, seed, "r3_models2.json")


# round-4 measured ratio: 3-bet rate vs OUR opens / vs others' opens (tools/_r4_targeting.py)
R5_TARGET_MULT = {"mw": 1.43, "Vaibhaav Tiwari": 1.63, "Shresth Rana": 1.66, "Henrik Gombos": 1.13,
                  "Lorenzo P.": 1.55, "Daksh Malhan": 1.28, "Kashyap Patel": 0.82}


class R5ModelBot(R2ModelBot):
    """Round-4-fitted model of a finalist (round 5 = final), targeting our seat like in round 4."""
    TARGET = True
    TARGET_MULT = R5_TARGET_MULT

    def __init__(self, name, seed=0):
        super().__init__(name, seed, "r4_models2.json")
