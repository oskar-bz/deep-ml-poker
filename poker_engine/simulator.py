"""
Full 8-Player Sit-and-Go No-Limit Hold'em Tournament Simulator
matching the exact Deep-ML Pokerbot Tournament Rules:
- 8 players per table, 2,000 starting chips each.
- Blinds rising every 25 hands.
- 200-hand cap; top three score (10 / 5 / 2), with exact ICM at the 200-hand cap.
- 1 ms per-decision think time limit; late answer or error becomes check or fold.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from poker_engine.evaluator import eval_7
from poker_engine.icm import PAYOUTS, compute_icm_equities


DEFAULT_BLIND_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (10, 20),    # Hands 1..25   (official schedule from the problem statement)
    (20, 40),    # Hands 26..50
    (30, 60),    # Hands 51..75
    (50, 100),   # Hands 76..100
    (75, 150),   # Hands 101..125
    (100, 200),  # Hands 126..150
    (150, 300),  # Hands 151..175
    (200, 400),  # Hands 176..200
)

AGGRESSIVE_BLIND_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (10, 20),    # Hands 1..25
    (20, 40),    # Hands 26..50
    (40, 80),    # Hands 51..75
    (80, 160),   # Hands 76..100
    (150, 300),  # Hands 101..125
    (300, 600),  # Hands 126..150
    (500, 1000), # Hands 151..175
    (800, 1600), # Hands 176..200
)


@dataclass
class BotLatencyStats:
    decisions: int = 0
    total_ns: int = 0
    max_ns: int = 0
    timeouts: int = 0
    errors: int = 0
    samples_us: List[float] = field(default_factory=list)

    def record(self, elapsed_ns: int, timed_out: bool, errored: bool) -> None:
        self.decisions += 1
        self.total_ns += elapsed_ns
        if elapsed_ns > self.max_ns:
            self.max_ns = elapsed_ns
        if timed_out:
            self.timeouts += 1
        if errored:
            self.errors += 1
        if len(self.samples_us) < 20000:
            self.samples_us.append(elapsed_ns / 1000.0)

    @property
    def mean_us(self) -> float:
        return (self.total_ns / max(1, self.decisions)) / 1000.0

    @property
    def p99_us(self) -> float:
        if not self.samples_us:
            return 0.0
        s = sorted(self.samples_us)
        idx = min(len(s) - 1, int(0.99 * len(s)))
        return s[idx]


@dataclass
class TableResult:
    points: List[float]
    final_stacks: List[int]
    hands_played: int
    elimination_order: List[int]
    hit_hand_cap: bool


class SitAndGoTable:
    """Runs a single 8-handed Sit-and-Go table under Deep-ML Contest Rules."""

    def __init__(
        self,
        bots: Sequence[Any],
        starting_stack: int = 2000,
        hands_per_level: int = 25,
        max_hands: int = 200,
        blind_schedule: Sequence[Tuple[int, int]] = DEFAULT_BLIND_SCHEDULE,
        enforce_1ms: bool = True,
        time_limit_ns: int = 1_000_000,  # 1.0 ms = 1,000,000 ns
        seed: Optional[int] = None,
        latency_stats: Optional[Dict[str, BotLatencyStats]] = None,
    ) -> None:
        assert len(bots) == 8, "Sit-and-Go table requires 8 bots"
        self.bots = list(bots)
        self.starting_stack = starting_stack
        self.hands_per_level = hands_per_level
        self.max_hands = max_hands
        self.blind_schedule = list(blind_schedule)
        self.enforce_1ms = enforce_1ms
        self.time_limit_ns = time_limit_ns
        self.rng = random.Random(seed)
        self.latency_stats = latency_stats if latency_stats is not None else {}

    def play_tournament(self) -> TableResult:
        stacks = [self.starting_stack] * 8
        elimination_order: List[int] = []  # first busted to last busted
        button_idx = self.rng.randrange(8)
        deck = list(range(52))

        for b_idx, bot in enumerate(self.bots):
            if hasattr(bot, "on_table_start"):
                try:
                    bot.on_table_start(b_idx, self.starting_stack)
                except Exception:
                    pass

        hands_played = 0
        for hand_num in range(1, self.max_hands + 1):
            alive_seats = [i for i, s in enumerate(stacks) if s > 0]
            if len(alive_seats) <= 1:
                break

            hands_played = hand_num
            level_idx = min(len(self.blind_schedule) - 1, (hand_num - 1) // self.hands_per_level)
            sb_amt, bb_amt = self.blind_schedule[level_idx]

            # Advance button to next alive player
            button_idx = self._next_alive(stacks, button_idx)
            hand_start_stacks = list(stacks)

            self._play_single_hand(
                hand_num=hand_num,
                stacks=stacks,
                button_idx=button_idx,
                sb_amt=sb_amt,
                bb_amt=bb_amt,
                deck=deck,
            )

            # Check for newly eliminated players on this hand
            busted_this_hand = [
                i for i in alive_seats if stacks[i] == 0
            ]
            if busted_this_hand:
                # Standard poker rule: if multiple players bust on the same hand,
                # the player who started the hand with FEWER chips busts earlier (lower finish)
                busted_this_hand.sort(key=lambda idx: hand_start_stacks[idx])
                elimination_order.extend(busted_this_hand)

            if sum(1 for s in stacks if s > 0) <= 1:
                break

        alive_end = [i for i, s in enumerate(stacks) if s > 0]
        hit_cap = len(alive_end) > 1
        points = self._score_table(stacks, elimination_order)

        return TableResult(
            points=points,
            final_stacks=stacks,
            hands_played=hands_played,
            elimination_order=elimination_order,
            hit_hand_cap=hit_cap,
        )

    def _score_table(
        self,
        stacks: Sequence[int],
        elimination_order: Sequence[int],
    ) -> List[float]:
        """
        Computes the (10, 5, 2) tournament score for all 8 seats.
        - If 1 player remains: 1st gets 10, last busted gets 5, second-to-last busted gets 2.
        - If >1 player remains at the 200-hand cap:
          - If >=3 players remain: all 17 points (10, 5, 2) distributed by exact ICM!
          - If 2 players remain: 3rd place already busted and gets 2 points;
            the 2 survivors split (10, 5) via exact ICM!
        """
        points = [0.0] * 8
        alive = [i for i, s in enumerate(stacks) if s > 0]
        n_alive = len(alive)

        if n_alive == 1:
            points[alive[0]] = 10.0
            if len(elimination_order) >= 1:
                points[elimination_order[-1]] = 5.0
            if len(elimination_order) >= 2:
                points[elimination_order[-2]] = 2.0
            return points

        if n_alive == 2:
            if len(elimination_order) >= 1:
                points[elimination_order[-1]] = 2.0
            icm_pts = compute_icm_equities(stacks, payouts=(10.0, 5.0, 0.0))
            for i in alive:
                points[i] = icm_pts[i]
            return points

        # n_alive >= 3 at 200-hand cap: full ICM across surviving stacks
        return compute_icm_equities(stacks, payouts=PAYOUTS)

    @staticmethod
    def _next_alive(stacks: Sequence[int], current: int) -> int:
        for offset in range(1, 9):
            cand = (current + offset) & 7
            if stacks[cand] > 0:
                return cand
        return current

    def _play_single_hand(
        self,
        hand_num: int,
        stacks: List[int],
        button_idx: int,
        sb_amt: int,
        bb_amt: int,
        deck: List[int],
    ) -> None:
        alive_seats = [i for i in range(8) if stacks[i] > 0]
        n_alive = len(alive_seats)
        if n_alive <= 1:
            return

        # Shuffle only the needed prefix of deck (2 * n_alive + 5 cards)
        needed_cards = 2 * n_alive + 5
        for i in range(needed_cards):
            j = i + self.rng.randrange(52 - i)
            deck[i], deck[j] = deck[j], deck[i]

        hole_cards: Dict[int, Tuple[int, int]] = {}
        ptr = 0
        for s in alive_seats:
            hole_cards[s] = (deck[ptr], deck[ptr + 1])
            ptr += 2
        full_board = deck[ptr : ptr + 5]

        starting_stacks = list(stacks)
        active = [stacks[i] > 0 for i in range(8)]
        hand_contrib = [0] * 8
        street_bets = [0] * 8
        action_history: List[Tuple[int, int, str, int]] = []

        # Post blinds (Heads-up rule: Button posts SB, other posts BB)
        if n_alive == 2:
            sb_seat = button_idx
            bb_seat = self._next_alive(stacks, sb_seat)
        else:
            sb_seat = self._next_alive(stacks, button_idx)
            bb_seat = self._next_alive(stacks, sb_seat)

        sb_posted = min(stacks[sb_seat], sb_amt)
        stacks[sb_seat] -= sb_posted
        street_bets[sb_seat] = sb_posted
        hand_contrib[sb_seat] = sb_posted
        action_history.append((0, sb_seat, "sb", sb_posted))

        bb_posted = min(stacks[bb_seat], bb_amt)
        stacks[bb_seat] -= bb_posted
        street_bets[bb_seat] = bb_posted
        hand_contrib[bb_seat] = bb_posted
        action_history.append((0, bb_seat, "bb", bb_posted))

        pot = sb_posted + bb_posted

        for street in range(4):
            board = full_board[:0] if street == 0 else full_board[: street + 2]
            if street > 0:
                for i in range(8):
                    street_bets[i] = 0

            # If at most 1 active player has chips left to bet, or only 1 active player remains, skip betting
            n_active = sum(1 for i in range(8) if active[i])
            if n_active <= 1:
                break
            n_can_act = sum(1 for i in range(8) if active[i] and stacks[i] > 0)

            current_high_bet = max(street_bets)
            last_raise_size = bb_amt

            # Determine first seat to act on this street
            if street == 0:
                first_seat = self._next_alive_active(active, bb_seat)
            else:
                first_seat = self._next_alive_active(active, button_idx)

            # If everyone is all-in (or only 1 player has chips and faces no bet), no actions needed
            if n_can_act == 0 or (n_can_act == 1 and street_bets[first_seat] >= current_high_bet):
                continue

            # Track which active players with chips still need to act at current_high_bet
            needs_to_act = {i for i in range(8) if active[i] and stacks[i] > 0}
            cur_seat = first_seat

            safety_steps = 0
            while needs_to_act and sum(1 for i in range(8) if active[i]) > 1:
                safety_steps += 1
                if safety_steps > 64:
                    break

                if cur_seat in needs_to_act and active[cur_seat] and stacks[cur_seat] > 0:
                    needs_to_act.remove(cur_seat)
                    to_call = max(0, current_high_bet - street_bets[cur_seat])
                    min_raise_add = min(stacks[cur_seat], to_call + last_raise_size)
                    max_raise_add = stacks[cur_seat]

                    state = {
                        "hand_number": hand_num,
                        "street": street,
                        "hero_idx": cur_seat,
                        "button_idx": button_idx,
                        "sb_seat": sb_seat,
                        "bb_seat": bb_seat,
                        "small_blind": sb_amt,
                        "big_blind": bb_amt,
                        "hole_cards": hole_cards[cur_seat],
                        "board": board,
                        "stacks": list(stacks),
                        "starting_stacks": starting_stacks,
                        "street_bets": list(street_bets),
                        "hand_contributions": list(hand_contrib),
                        "active_seats": list(active),
                        "pot": pot,
                        "to_call": to_call,
                        "min_raise_add": min_raise_add,
                        "max_raise_add": max_raise_add,
                        "action_history": list(action_history),
                    }

                    act_type, act_amt = self._query_bot(self.bots[cur_seat], state)

                    if act_type == "fold":
                        if to_call == 0:
                            # Never fold when checking is free
                            action_history.append((street, cur_seat, "check", 0))
                        else:
                            active[cur_seat] = False
                            action_history.append((street, cur_seat, "fold", 0))
                    elif act_type in ("check", "call"):
                        put_in = min(stacks[cur_seat], to_call)
                        stacks[cur_seat] -= put_in
                        street_bets[cur_seat] += put_in
                        hand_contrib[cur_seat] += put_in
                        pot += put_in
                        action_history.append(
                            (street, cur_seat, "check" if put_in == 0 else "call", put_in)
                        )
                    elif act_type in ("raise", "bet"):
                        put_in = int(act_amt)
                        if put_in < min_raise_add:
                            put_in = min_raise_add
                        if put_in > stacks[cur_seat]:
                            put_in = stacks[cur_seat]

                        new_street_bet = street_bets[cur_seat] + put_in
                        raise_delta = new_street_bet - current_high_bet
                        if new_street_bet > current_high_bet:
                            if raise_delta >= last_raise_size:
                                last_raise_size = raise_delta
                            current_high_bet = new_street_bet
                            # Re-open action to other active players with chips
                            needs_to_act = {
                                i
                                for i in range(8)
                                if i != cur_seat and active[i] and stacks[i] > 0
                            }
                        stacks[cur_seat] -= put_in
                        street_bets[cur_seat] = new_street_bet
                        hand_contrib[cur_seat] += put_in
                        pot += put_in
                        action_history.append((street, cur_seat, "raise", put_in))

                cur_seat = (cur_seat + 1) & 7

        # Resolve pot / side-pots
        remaining_players = [i for i in range(8) if active[i]]
        if len(remaining_players) == 1:
            winner = remaining_players[0]
            stacks[winner] += pot
        else:
            # Evaluate 7-card hands for all remaining active players
            b0, b1, b2, b3, b4 = (
                full_board[0],
                full_board[1],
                full_board[2],
                full_board[3],
                full_board[4],
            )
            scores: Dict[int, int] = {}
            for p in remaining_players:
                h0, h1 = hole_cards[p]
                scores[p] = eval_7(h0, h1, b0, b1, b2, b3, b4)

            # Side-pot decomposition by distinct contribution levels
            levels = sorted({hand_contrib[p] for p in remaining_players if hand_contrib[p] > 0})
            prev_level = 0
            for lvl in levels:
                delta = lvl - prev_level
                if delta <= 0:
                    continue
                sub_pot = 0
                for i in range(8):
                    if hand_contrib[i] > prev_level:
                        sub_pot += min(delta, hand_contrib[i] - prev_level)
                eligible = [p for p in remaining_players if hand_contrib[p] >= lvl]
                best_score = max(scores[p] for p in eligible)
                winners = [p for p in eligible if scores[p] == best_score]
                share = sub_pot // len(winners)
                rem_chips = sub_pot - share * len(winners)
                for w in winners:
                    stacks[w] += share
                if rem_chips > 0:
                    # Odd chips go to first winner clockwise from button
                    winners.sort(key=lambda w: (w - button_idx - 1) & 7)
                    for idx_r in range(rem_chips):
                        stacks[winners[idx_r % len(winners)]] += 1
                prev_level = lvl

        # Notify bots of hand completion for opponent profiling
        hand_summary = {
            "hand_number": hand_num,
            "starting_stacks": starting_stacks,
            "ending_stacks": list(stacks),
            "hand_contributions": hand_contrib,
            "action_history": action_history,
            "active_at_end": remaining_players,
            "board": full_board if len(remaining_players) > 1 else [],
            "all_hole": {p: hole_cards[p] for p in hole_cards},
            "board_dealt": (full_board if len(remaining_players) > 1 else
                            full_board[: {0: 0, 1: 3, 2: 4, 3: 5}[max((x[0] for x in action_history), default=0)]]),
            "showdown_hands": {p: hole_cards[p] for p in remaining_players}
            if len(remaining_players) > 1
            else {},
        }
        for b_idx, bot in enumerate(self.bots):
            if hasattr(bot, "on_hand_end"):
                try:
                    bot.on_hand_end(b_idx, hand_summary)
                except Exception:
                    pass

    @staticmethod
    def _next_alive_active(active: Sequence[bool], current: int) -> int:
        for offset in range(1, 9):
            cand = (current + offset) & 7
            if active[cand]:
                return cand
        return current

    def _query_bot(self, bot: Any, state: Dict[str, Any]) -> Tuple[str, int]:
        to_call = state["to_call"]
        fallback = ("check", 0) if to_call == 0 else ("fold", 0)
        bot_name = getattr(bot, "name", bot.__class__.__name__)
        stats = self.latency_stats.setdefault(bot_name, BotLatencyStats())

        t0 = time.perf_counter_ns()
        timed_out = False
        errored = False
        try:
            res = bot.act(state)
            elapsed_ns = time.perf_counter_ns() - t0
        except Exception:
            elapsed_ns = time.perf_counter_ns() - t0
            errored = True
            stats.record(elapsed_ns, False, True)
            return fallback

        if elapsed_ns > self.time_limit_ns and not getattr(bot, "exempt_timing", False):
            timed_out = True
            stats.record(elapsed_ns, True, False)
            if self.enforce_1ms:
                return fallback
        else:
            stats.record(elapsed_ns, False, False)

        if not isinstance(res, (tuple, list)) or len(res) != 2:
            return fallback
        act_type, act_amt = res[0], int(res[1])
        if act_type not in ("fold", "check", "call", "raise", "bet"):
            return fallback
        return act_type, act_amt
