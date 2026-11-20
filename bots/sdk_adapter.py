"""
Deep-ML SDK adapter: `MyBot(Bot)` turns the platform's `state` object into the
canonical simulator dict our core bots use, and their (type, chips_added) answer
back into an SDK `Action`.

Everything that is not documented on the contest page (seat index, button, hand
number, history amount convention, action names) is read defensively with fallbacks:
- seat:    state.seat / my_seat / player / me / position / index / hero
- button:  state.button / dealer / ...; else inferred from the small-blind poster
- history amounts: "raise" amounts are auto-detected as raise-to totals or chips
           added by replaying the street and matching state.bets
- hand number: state.hand_number / hand / ...; else counted locally
FIRST THING TO DO IN THE CONTEST NOTEBOOK: print(vars(state)) once and check these.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from pokerbot_sdk import Action, Bot
except Exception:  # pragma: no cover - lets the module import without the SDK
    class Bot:  # type: ignore[no-redef]
        pass

    class Action:  # type: ignore[no-redef]
        def __init__(self, kind: str, amount: int = 0) -> None:
            self.kind, self.amount = kind, amount

        @staticmethod
        def fold(): return Action("fold")

        @staticmethod
        def check(): return Action("check")

        @staticmethod
        def call(): return Action("call")

        @staticmethod
        def raise_to(total): return Action("raise", total)

        @staticmethod
        def all_in(): return Action("all_in")

from time import perf_counter_ns as _perf_ns

from poker_engine.evaluator import parse_card

_STREET_IDX = {"preflop": 0, "pre": 0, "flop": 1, "turn": 2, "river": 3,
               "pre-flop": 0, "pre_flop": 0, 0: 0, 1: 1, 2: 2, 3: 3}


def _norm_action(a: Any) -> str:
    s = str(a).lower()
    if "small" in s or s == "sb" or s == "post_sb":
        return "sb"
    if "big" in s or s == "bb" or s == "post_bb":
        return "bb"
    if "post" in s or "blind" in s or "ante" in s:
        return "bb"
    if "fold" in s:
        return "fold"
    if "check" in s:
        return "check"
    if "call" in s:
        return "call"
    if "all" in s or "raise" in s or "bet" in s or "shove" in s or "push" in s:
        return "raise"
    return s


def _attr(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default


def _entry(e: Any) -> Tuple[int, int, str, int]:
    if isinstance(e, dict):
        st = e.get("street", 0)
        seat = e.get("seat", e.get("player", e.get("pos", -1)))
        act = e.get("action", e.get("type", e.get("move", "")))
        amt = e.get("amount", e.get("size", e.get("to", 0))) or 0
    elif isinstance(e, (tuple, list)):
        st, seat, act, amt = (list(e) + [0, 0, "", 0])[:4]
    else:
        st = getattr(e, "street", 0)
        seat = getattr(e, "seat", -1)
        act = getattr(e, "action", "")
        amt = getattr(e, "amount", 0) or 0
    st = _STREET_IDX.get(st if not isinstance(st, str) else st.lower(), 0)
    return st, int(seat), _norm_action(act), int(amt)


def _fix_blinds(entries: List[Tuple[int, int, str, int]]) -> List[Tuple[int, int, str, int]]:
    """Platform logs both blinds as action "blind": the first preflop post is the small blind."""
    if any(a == "sb" for _, _, a, _ in entries):
        return entries
    idx = [i for i, (st, _, a, _) in enumerate(entries) if st == 0 and a == "bb"]
    if len(idx) >= 2:
        st, s, _, amt = entries[idx[0]]
        entries = list(entries)
        entries[idx[0]] = (st, s, "sb", amt)
    return entries


class MyBot(Bot):
    """SDK entry point. `core_factory` builds the decision engine (ApexBot-style)."""

    core_factory = None  # set by the submission file

    def __init__(self, core: Any = None) -> None:
        if core is None:
            core = type(self).core_factory() if type(self).core_factory else None
        self.core = core
        self._hand_key: Any = None
        self._local_hand = 0
        self._raise_as_total: Optional[bool] = None  # history amount convention
        self._last: Optional[Dict[str, Any]] = None
        self._recorded_hand = -1   # platform hand ids start at 0
        self._n_seats = 8
        self._prev_hand_no = 0
        self._seat_votes: List[int] = []

    # ------------------------------------------------------------------ helpers
    def _history(self, raw_hist: Any, bets: List[int], cur_street: int,
                 pot: int = -1) -> List[Tuple[int, int, str, int]]:
        """Normalized history with amounts converted to chips added.

        The raise-amount convention (raise-to total vs chips added) is detected once
        per table: the right one reproduces the current street bets and the pot.
        """
        entries = _fix_blinds([_entry(e) for e in (raw_hist or ())])
        if self._raise_as_total is None and entries:
            bt, pt = self._replay(entries, True, cur_street)
            ba, pa = self._replay(entries, False, cur_street)
            ok_total = bt == bets and (pot < 0 or pt == pot)
            ok_added = ba == bets and (pot < 0 or pa == pot)
            if ok_total != ok_added:
                self._raise_as_total = ok_total
            elif not ok_total:
                as_t = bt == bets  # neither fits fully: prefer the street-bets match
                if as_t != (ba == bets):
                    self._pending_as_total = as_t
        if self._raise_as_total is None:
            as_total = getattr(self, "_pending_as_total", True)
            self._pending_as_total = True
        else:
            as_total = self._raise_as_total
        out: List[Tuple[int, int, str, int]] = []
        sb: Dict[int, int] = {}
        high = 0
        cur = -1
        for st, s, a, amt in entries:
            if st != cur:
                cur, sb, high = st, {}, 0
            mine = sb.get(s, 0)
            if a in ("sb", "bb"):
                add = amt
            elif a == "call":
                add = amt if amt > 0 and amt <= max(0, high - mine) else max(0, high - mine)
            elif a == "raise":
                add = (amt - mine) if as_total else amt
                if add <= 0:
                    add = amt
            else:
                add = 0
            sb[s] = mine + add
            if sb[s] > high:
                high = sb[s]
            out.append((st, s, a, add))
        return out

    def _replay(self, entries, as_total: bool, cur_street: int) -> Tuple[List[int], int]:
        """(current street bets, total chips put in) under one amount convention."""
        n = self._n_seats
        sb = [0] * n
        cur = -1
        high = 0
        total = 0
        for st, s, a, amt in entries:
            if st != cur:
                cur, sb, high = st, [0] * n, 0
            if not 0 <= s < n:
                continue
            before = sb[s]
            if a in ("sb", "bb"):
                sb[s] += amt
            elif a == "call":
                sb[s] = max(sb[s], high) if amt == 0 or amt > high - sb[s] else sb[s] + amt
            elif a == "raise":
                sb[s] = amt if as_total else sb[s] + amt
            total += sb[s] - before
            high = max(high, sb[s])
        if cur != cur_street:
            return [0] * n, total
        return sb, total

    def _canonical(self, state: Any) -> Dict[str, Any]:
        hole_raw = _attr(state, ("hole", "hole_cards", "cards", "hand"))
        h0, h1 = parse_card(hole_raw[0]), parse_card(hole_raw[1])
        board = [parse_card(c) for c in (_attr(state, ("board", "community_cards"), ()) or ())]
        stacks = [int(x) for x in _attr(state, ("stacks", "chips"), [])]
        n = len(stacks) or 8
        self._n_seats = n
        bets = [int(x) for x in (_attr(state, ("bets", "street_bets"), None) or [0] * n)]
        street_raw = _attr(state, ("street",), None)
        if isinstance(street_raw, str):
            street = _STREET_IDX.get(street_raw.lower(), 0)
        elif isinstance(street_raw, int):
            street = street_raw
        else:
            nb = len(board)
            street = 0 if nb == 0 else (1 if nb == 3 else (2 if nb == 4 else 3))
        bb = int(_attr(state, ("big_blind", "bb"), 20))
        sbl = int(_attr(state, ("small_blind", "sb"), max(1, bb // 2)))
        to_call = int(_attr(state, ("to_call",), 0))
        pot = int(_attr(state, ("pot",), 0))

        raw_hist = _attr(state, ("history", "action_history", "actions"), ())
        pre_entries = _fix_blinds([_entry(e) for e in (raw_hist or ())])

        hero = _attr(state, ("seat", "my_seat", "seat_idx", "player", "player_idx", "me",
                             "hero", "hero_idx", "position", "index", "idx"), None)
        if not isinstance(hero, int) or not 0 <= hero < n:
            hero = self._infer_hero(state, stacks, bets, to_call, pre_entries, n)

        hist = self._history(raw_hist, bets, street, pot)

        # Seats involved this hand, folds, contributions
        folded = set()
        involved = set()
        contrib = [0] * n
        sb_seat = bb_seat = -1
        for st, s, a, add in hist:
            if not 0 <= s < n:
                continue
            involved.add(s)
            if a == "fold":
                folded.add(s)
            contrib[s] += add
            if a == "sb" and sb_seat < 0:
                sb_seat = s
            elif a == "bb" and bb_seat < 0:
                bb_seat = s
        # current street: state.bets is authoritative; history only for earlier streets
        prev = [0] * n
        for st, s, a, add in hist:
            if 0 <= s < n and st < street:
                prev[s] += add
        for i in range(n):
            contrib[i] = prev[i] + bets[i]
        active_raw = _attr(state, ("active", "in_hand", "active_seats", "live"), None)
        if isinstance(active_raw, (list, tuple)) and len(active_raw) == n:
            active = [bool(x) for x in active_raw]
        else:
            active = [((stacks[i] > 0) or (i in involved) or bets[i] > 0) and i not in folded
                      for i in range(n)]
        active[hero] = True

        alive = [i for i in range(n) if stacks[i] > 0 or contrib[i] > 0 or i == hero]
        button = _attr(state, ("button", "dealer", "button_seat", "dealer_seat", "btn",
                               "button_idx", "dealer_idx"), None)
        if not isinstance(button, int) or not 0 <= button < n:
            if sb_seat >= 0:
                if len(alive) == 2:
                    button = sb_seat
                else:
                    button = sb_seat
                    for k in range(1, n):
                        c = (sb_seat - k) % n
                        if c in alive:
                            button = c
                            break
            elif bb_seat >= 0:
                button = (bb_seat - 2) % n
            else:
                button = (hero - 3) % n
        if sb_seat < 0:
            sb_seat = next(((button + k) % n for k in range(1, n + 1)
                            if ((button + k) % n) in alive), button)
            if len(alive) == 2:
                sb_seat = button
        if bb_seat < 0:
            bb_seat = next(((sb_seat + k) % n for k in range(1, n + 1)
                            if ((sb_seat + k) % n) in alive), sb_seat)

        # Hand number
        hand_no = _attr(state, ("hand_number", "hand_num", "hand_id", "hand_index", "hand_no"), None)
        key = (h0, h1, button)
        if key != self._hand_key or len(hist) < (self._last or {}).get("_hlen", 0):
            if self._hand_key is not None:
                self._flush_hand()
            self._hand_key = key
            self._local_hand += 1
        if isinstance(hand_no, int):
            if hand_no < self._prev_hand_no and self.core is not None:
                # Same bot object reused at a new table: forget the old opponents.
                if hasattr(self.core, "on_table_start"):
                    self.core.on_table_start(hero, sum(stacks) // max(1, len(stacks)))
                self._recorded_hand = -1
                self._raise_as_total = None
                self._seat_votes = []
            self._prev_hand_no = hand_no
        if not isinstance(hand_no, int):
            level = _attr(state, ("level",), None)
            hand_no = self._local_hand
            if isinstance(level, int):
                hand_no = max(hand_no, level * 25 + 1)

        my_bet = bets[hero]
        min_raise = int(_attr(state, ("min_raise",), my_bet + to_call + bb))
        max_raise = int(_attr(state, ("max_raise",), my_bet + stacks[hero]))
        if n < 8:  # core bots assume 8 seats
            pad = 8 - n
            stacks = stacks + [0] * pad
            bets = bets + [0] * pad
            contrib = contrib + [0] * pad
            active = active + [False] * pad
        d = {
            "hand_number": hand_no,
            "street": street,
            "hero_idx": hero,
            "button_idx": button,
            "sb_seat": sb_seat,
            "bb_seat": bb_seat,
            "small_blind": sbl,
            "big_blind": bb,
            "hole_cards": (h0, h1),
            "board": board,
            "stacks": stacks,
            "starting_stacks": [stacks[i] + contrib[i] for i in range(len(stacks))],
            "street_bets": bets,
            "hand_contributions": contrib,
            "active_seats": active,
            "pot": pot,
            "to_call": to_call,
            "min_raise_add": max(0, min_raise - my_bet),
            "max_raise_add": max(0, max_raise - my_bet),
            "action_history": hist,
            "_legal": list(_attr(state, ("legal", "legal_actions"), ()) or ()),
            "_hlen": len(hist),
        }
        self._last = d
        return d

    def _infer_hero(self, state, stacks, bets, to_call, entries, n) -> int:
        my_stack = _attr(state, ("stack", "my_stack", "chips_left"), None)
        cands = list(range(n))
        if isinstance(my_stack, int):
            cands = [i for i in cands if stacks[i] == my_stack] or cands
        high = max(bets) if bets else 0
        cands = [i for i in cands if high - bets[i] == to_call or stacks[i] <= to_call] or cands
        guess = cands[0]
        if len(cands) > 1 and entries:
            # The seat to act is the next live seat after the last actor (or, on a new
            # street, after the button).
            folded = {s for _, s, a, _ in entries if a == "fold"}
            nb = len(_attr(state, ("board", "community_cards"), ()) or ())
            cur_street = 0 if nb == 0 else (1 if nb == 3 else (2 if nb == 4 else 3))
            start = entries[-1][1]
            if entries[-1][0] != cur_street:
                sb_seat = next((s for _, s, a, _ in entries if a == "sb"), -1)
                bb_seat = next((s for _, s, a, _ in entries if a == "bb"), -1)
                live = {s for _, s, _, _ in entries} | {i for i in range(n) if stacks[i] > 0}
                if sb_seat >= 0 and bb_seat >= 0:
                    heads_up = len(live) == 2
                    start = sb_seat if heads_up else (sb_seat - 1) % n
            for k in range(1, n + 1):
                c = (start + k) % n
                if c not in folded and stacks[c] > 0:
                    if c in cands:
                        guess = c
                    break
        # The seat never changes within a table: majority vote over decisions.
        if len(self._seat_votes) != n:
            self._seat_votes = [0] * n
        self._seat_votes[guess] += 1
        best = max(range(n), key=lambda i: self._seat_votes[i])
        return best if best in cands else guess

    def _flush_hand(self) -> None:
        d = self._last
        if not d or self.core is None or not hasattr(self.core, "on_hand_end"):
            return
        if d["hand_number"] <= self._recorded_hand:
            return
        self._recorded_hand = d["hand_number"]
        try:
            self.core.on_hand_end(d["hero_idx"], {
                "hand_number": d["hand_number"],
                "starting_stacks": d["starting_stacks"],
                "action_history": d["action_history"],
                "showdown_hands": {},
                "board": d["board"],
            })
        except Exception:
            pass

    # --------------------------------------------------------------- SDK hooks
    def act(self, state: Any):
        t0 = _perf_ns()
        try:
            d = self._canonical(state)
            d["_t0"] = t0
        except Exception:
            to_call = _attr(state, ("to_call",), 0) or 0
            return Action.check() if to_call == 0 else Action.fold()
        to_call = d["to_call"]
        try:
            kind, add = self.core.act(d)
        except Exception:
            kind, add = ("check", 0) if to_call == 0 else ("fold", 0)
        return self._to_action(d, kind, int(add))

    @staticmethod
    def _to_action(d: Dict[str, Any], kind: str, add: int):
        to_call = d["to_call"]
        hero = d["hero_idx"]
        stack = d["stacks"][hero]
        legal = d["_legal"]
        if kind == "fold":
            return Action.check() if to_call == 0 else Action.fold()
        if kind in ("check", "call"):
            return Action.check() if to_call == 0 else Action.call()
        # raise / bet
        if add >= stack:
            if stack <= to_call:
                return Action.call()
            return Action.all_in()
        if legal and "raise" not in legal and "bet" not in legal:
            return Action.check() if to_call == 0 else Action.call()
        my_bet = d["street_bets"][hero]
        total = my_bet + max(add, d["min_raise_add"])
        if total >= my_bet + d["max_raise_add"]:
            return Action.all_in()
        if total <= max(d["street_bets"]):
            return Action.check() if to_call == 0 else Action.call()
        return Action.raise_to(total)

    def on_hand_end(self, result: Any) -> None:
        d = self._last
        if not d or self.core is None:
            return
        try:
            shown_raw = _attr(result, ("shown", "showdown", "cards", "revealed", "hands"), {}) or {}
            shown = {}
            items = shown_raw.items() if isinstance(shown_raw, dict) else enumerate(shown_raw)
            for s, cards in items:
                if cards and len(cards) == 2:
                    shown[int(s)] = (parse_card(cards[0]), parse_card(cards[1]))
            hist_raw = _attr(result, ("history", "action_history", "actions"), None)
            hist = d["action_history"]
            if hist_raw:
                stacks = d["stacks"]
                hist = self._history(hist_raw, [0] * len(stacks), -1)
            board_raw = _attr(result, ("board", "community_cards"), None)
            board = [parse_card(c) for c in board_raw] if board_raw else d["board"]
            if d["hand_number"] > self._recorded_hand and hasattr(self.core, "on_hand_end"):
                self._recorded_hand = d["hand_number"]
                all_hole = {}
                try:
                    hr = _attr(result, ("hole", "holes", "hole_cards"), None)
                    if hr is not None:
                        it = hr.items() if isinstance(hr, dict) else enumerate(hr)
                        for s_, cards in it:
                            if cards and len(cards) == 2:
                                all_hole[int(s_)] = (parse_card(cards[0]), parse_card(cards[1]))
                except Exception:
                    all_hole = {}
                deltas = _attr(result, ("deltas", "delta", "chip_deltas", "net"), None)
                try:
                    deltas = [int(x) for x in deltas] if deltas is not None and len(deltas) == len(d["stacks"]) else None
                except Exception:
                    deltas = None
                self.core.on_hand_end(d["hero_idx"], {
                    "hand_number": d["hand_number"],
                    "starting_stacks": d["starting_stacks"],
                    "action_history": hist,
                    "showdown_hands": shown,
                    "deltas": deltas,
                    "all_hole": all_hole,
                    "bb": d.get("big_blind", 0),
                    "board": board,
                })
        except Exception:
            pass
        self._hand_key = None  # next act() starts a new hand
