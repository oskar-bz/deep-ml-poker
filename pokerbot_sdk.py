"""
Local stand-in for the Deep-ML `pokerbot_sdk` (the real one exists only inside the
contest notebook / match workers).

Reconstructed from the contest page's documented API (see reference/starter_module.js):

    act(state) -> Action.fold() / check() / call() / raise_to(total) / all_in()
    state.hole, state.board         cards as strings like "As", "Td"
    state.street                    "preflop" / "flop" / "turn" / "river"
    state.to_call                   chips to add to call (0 = can check)
    state.min_raise / max_raise     smallest / largest legal raise-to TOTAL this street
    state.current_bet               highest street bet (raise-to totals are vs this)
    state.pot, state.stacks, state.bets
    state.big_blind, state.small_blind, state.level
    state.stack_m                   stack in orbits of blinds (stack / (sb + bb))
    state.legal                     e.g. ["fold", "call", "raise"]
    state.opponents                 opponents still in the hand
    state.history                   every action this hand: street, seat, action, amount
    state.pot_odds()
    state.preflop_equity() / preflop_rank()   vs the players still in
    state.made_hand()               {"name", "category" (0..8), "value"}
    state.equity(samples)           Monte Carlo pot share (~3 ms: too slow in matches)
    on_hand_end(result)             optional: deltas, stacks, cards shown

Fields NOT documented (seat, button, hand_number, ...) are provided here under the
names we guess; the submission's adapter reads them defensively and has fallbacks.

`SDKBotDriver` wraps an SDK-style bot so it can sit at a `poker_engine.simulator`
table, which lets us run the exact submission code path offline.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence

from poker_engine.evaluator import eval_cards, card_to_str, parse_card as _parse_card
from poker_engine.preflop_multiway import MULTIWAY_EQ

STREET_NAMES = ("preflop", "flop", "turn", "river")
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
)

# ---------------------------------------------------------------------------
# Card helpers (SDK exports parse_card / card_str)
# ---------------------------------------------------------------------------


def parse_card(c: Any) -> int:
    return _parse_card(c)


def card_str(c: int) -> str:
    return card_to_str(c)


def _class_key(c0: int, c1: int):
    r0, r1 = c0 >> 2, c1 >> 2
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    return (hi, lo, hi != lo and (c0 & 3) == (c1 & 3))


# preflop_rank: rank among the 169 classes by equity vs n opponents (1.0 = best)
_RANK_BY_N: List[Dict[Any, float]] = []
for _n in range(7):
    _order = sorted(MULTIWAY_EQ.keys(), key=lambda k: MULTIWAY_EQ[k][_n])
    _RANK_BY_N.append({k: i / (len(_order) - 1) for i, k in enumerate(_order)})


def preflop_equity_lookup(c0: int, c1: int, opponents: int) -> float:
    n = min(7, max(1, opponents))
    return MULTIWAY_EQ[_class_key(c0, c1)][n - 1]


def preflop_rank_lookup(c0: int, c1: int, opponents: int) -> float:
    n = min(7, max(1, opponents))
    return _RANK_BY_N[n - 1][_class_key(c0, c1)]


def equity(hole: Sequence[Any], board: Sequence[Any] = (), opponents: int = 1,
           samples: int = 200, rng: Optional[random.Random] = None) -> float:
    """Monte Carlo pot share vs `opponents` random hands."""
    rng = rng or random
    h = [parse_card(c) for c in hole]
    b = [parse_card(c) for c in board]
    dead = set(h) | set(b)
    rest = [c for c in range(52) if c not in dead]
    need = (5 - len(b)) + 2 * opponents
    share = 0.0
    for _ in range(samples):
        d = rng.sample(rest, need)
        full = b + d[: 5 - len(b)]
        me = eval_cards(h + full)
        ties, lost = 1, False
        off = 5 - len(b)
        for k in range(opponents):
            v = eval_cards([d[off + 2 * k], d[off + 2 * k + 1]] + full)
            if v > me:
                lost = True
                break
            if v == me:
                ties += 1
        if not lost:
            share += 1.0 / ties
    return share / max(1, samples)


# ---------------------------------------------------------------------------
# Weights helpers (no-op storage locally)
# ---------------------------------------------------------------------------
_SAVED: Dict[str, Any] = {}


def save_weights(model: Any) -> int:
    _SAVED["w"] = model
    return 0


def load_arrays():
    return _SAVED.get("w")


load_weights = load_arrays
load_state_dict = load_arrays


def clear_weights() -> None:
    _SAVED.pop("w", None)


# ---------------------------------------------------------------------------
# Action / Bot / State
# ---------------------------------------------------------------------------


class Action:
    __slots__ = ("kind", "amount")

    def __init__(self, kind: str, amount: int = 0) -> None:
        self.kind = kind
        self.amount = int(amount)

    @staticmethod
    def fold() -> "Action":
        return Action("fold")

    @staticmethod
    def check() -> "Action":
        return Action("check")

    @staticmethod
    def call() -> "Action":
        return Action("call")

    @staticmethod
    def raise_to(total: int) -> "Action":
        return Action("raise", total)

    @staticmethod
    def all_in() -> "Action":
        return Action("all_in")

    def __repr__(self) -> str:
        return f"Action({self.kind}, {self.amount})"


class Bot:
    def act(self, state: "State") -> Action:  # pragma: no cover - interface
        raise NotImplementedError

    def on_hand_end(self, result: Dict[str, Any]) -> None:
        pass


class State:
    """Snapshot handed to Bot.act(). Built from a simulator state dict."""

    def __init__(self, d: Dict[str, Any]) -> None:
        seat = d["hero_idx"]
        self.seat = seat
        self.button = d["button_idx"]
        self.hand_number = d.get("hand_number", 1)
        self.street = STREET_NAMES[d["street"]]
        h0, h1 = d["hole_cards"]
        self._hole_i = (h0, h1)
        self._board_i = list(d["board"])
        self.hole = [card_str(h0), card_str(h1)]
        self.board = [card_str(c) for c in d["board"]]
        self.stacks = list(d["stacks"])
        self.bets = list(d["street_bets"])
        self.stack = self.stacks[seat]
        self.pot = d["pot"]
        self.to_call = min(d["to_call"], self.stack)
        self.current_bet = max(self.bets)
        my_bet = self.bets[seat]
        self.max_raise = my_bet + self.stack
        self.min_raise = min(self.max_raise, my_bet + d["min_raise_add"])
        self.big_blind = d["big_blind"]
        self.small_blind = d["small_blind"]
        self.level = d.get("level", 0)
        self.stack_m = self.stack / max(1, self.big_blind + self.small_blind)
        active = d["active_seats"]
        self.active = list(active)
        self.opponents = sum(1 for i in range(len(active)) if active[i] and i != seat)
        self.players = sum(1 for s in self.stacks if s > 0) + sum(
            1 for i, s in enumerate(self.stacks) if s == 0 and active[i]
        )
        legal = ["fold", "call"] if self.to_call > 0 else ["check"]
        if self.stack > self.to_call:
            legal.append("raise")
        self.legal = legal
        self.history = _sdk_history(d["action_history"])

    def pot_odds(self) -> float:
        if self.to_call <= 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    def preflop_equity(self) -> float:
        return preflop_equity_lookup(self._hole_i[0], self._hole_i[1], self.opponents)

    def preflop_rank(self) -> float:
        return preflop_rank_lookup(self._hole_i[0], self._hole_i[1], self.opponents)

    def made_hand(self) -> Dict[str, Any]:
        cards = list(self._hole_i) + self._board_i
        if len(cards) < 5:
            r0, r1 = self._hole_i[0] >> 2, self._hole_i[1] >> 2
            cat = 1 if r0 == r1 else 0
            return {"name": CATEGORY_NAMES[cat], "category": cat, "value": 0}
        v = eval_cards(cards)
        cat = v >> 20
        return {"name": CATEGORY_NAMES[cat], "category": cat, "value": v}

    def equity(self, samples: int = 200) -> float:
        return equity(self.hole, self.board, max(1, self.opponents), samples)


# ---------------------------------------------------------------------------
# Simulator bridge
# ---------------------------------------------------------------------------


_SDK_ACTION_NAMES = {"sb": "small_blind", "bb": "big_blind"}


def _sdk_history(action_history) -> List[Dict[str, Any]]:
    """Simulator history (amount = chips added) -> SDK-style dicts.

    Guessed convention (the real one is undocumented): raises report the raise-to
    street total, calls/blinds the chips put in. The submission adapter detects
    either convention, so this only needs to be *a* plausible choice.
    """
    out = []
    street_bets: Dict[int, int] = {}
    cur = -1
    for (st, s, a, amt) in action_history:
        if st != cur:
            cur = st
            street_bets = {}
        street_bets[s] = street_bets.get(s, 0) + amt
        shown = street_bets[s] if a == "raise" else amt
        out.append({"street": STREET_NAMES[st], "seat": s,
                    "action": _SDK_ACTION_NAMES.get(a, a), "amount": shown})
    return out


class SDKBotDriver:
    """Adapts an SDK `Bot` to the poker_engine.simulator act(dict)->(type, add) interface.

    The simulator's history amounts are chips added; this driver converts them to the
    SDK-style list of dicts, and the SDK raise-to totals back into chips added.
    """

    def __init__(self, bot: Bot, name: Optional[str] = None, exempt_timing: bool = False) -> None:
        self.bot = bot
        self.name = name or bot.__class__.__name__
        self.exempt_timing = exempt_timing
        self._seat = 0

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self._seat = seat_idx
        if hasattr(self.bot, "on_table_start"):
            self.bot.on_table_start(seat_idx, starting_stack)

    def act(self, d: Dict[str, Any]):
        st = State(d)
        a = self.bot.act(st)
        seat = d["hero_idx"]
        stack = d["stacks"][seat]
        my_bet = d["street_bets"][seat]
        to_call = d["to_call"]
        if a is None:
            return ("check", 0) if to_call == 0 else ("fold", 0)
        k = a.kind
        if k == "fold":
            return ("fold", 0)
        if k == "check":
            return ("check", 0) if to_call == 0 else ("fold", 0)
        if k == "call":
            return ("call", to_call)
        if k == "all_in":
            if stack <= to_call:
                return ("call", to_call)
            return ("raise", stack)
        if k == "raise":
            total = max(st.min_raise, min(st.max_raise, a.amount))
            add = total - my_bet
            if add <= to_call:
                return ("call", to_call) if to_call > 0 else ("check", 0)
            return ("raise", add)
        return ("check", 0) if to_call == 0 else ("fold", 0)

    def on_hand_end(self, hero_idx: int, summary: Dict[str, Any]) -> None:
        start = summary["starting_stacks"]
        end = summary["ending_stacks"]
        result = {
            "hand_number": summary.get("hand_number"),
            "deltas": [e - s for s, e in zip(start, end)],
            "stacks": list(end),
            "shown": {s: [card_str(c) for c in cards]
                      for s, cards in summary.get("showdown_hands", {}).items()},
            "board": [card_str(c) for c in summary.get("board_dealt", summary.get("board", []))],
            "history": _sdk_history(summary.get("action_history", [])),
            # tournament reality (verified on round-3 replays): only showdown hands + our own are revealed
            "hole": [[card_str(c) for c in summary.get("all_hole", {}).get(i, ())]
                     if (i == hero_idx or i in summary.get("showdown_hands", {})) else []
                     for i in range(len(start))],
        }
        if hasattr(self.bot, "on_hand_end"):
            self.bot.on_hand_end(result)


def play_local(bot: Bot, tables: int = 20, seed: int = 0) -> None:
    """Rough analogue of the SDK's play_local: your bot vs 7 starter-style bots."""
    from poker_engine.simulator import SitAndGoTable
    from bots.sdk_field import StarterBot

    pts = 0.0
    t0 = time.time()
    for t in range(tables):
        seats = [SDKBotDriver(bot, "you")] + [SDKBotDriver(StarterBot(), f"house{i}") for i in range(7)]
        res = SitAndGoTable(seats, seed=seed + t).play_tournament()
        pts += res.points[0]
    print(f"{tables} tables: {pts / tables:.2f} avg points ({time.time() - t0:.1f}s)")
