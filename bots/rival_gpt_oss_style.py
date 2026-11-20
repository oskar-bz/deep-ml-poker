from __future__ import annotations
import random
from typing import Dict, List, Sequence, Tuple
from poker_engine.preflop_tables import PREFLOP_PERCENTILE, NASH_PUSH_BB, NASH_CALL_BB
from poker_engine.icm import compute_bubble_factor
from poker_engine.evaluator import analyze_board_texture, conditional_postflop_profile


class RivalGPTOSSStyleBot:
    """
    Rival Team #7: RivalGPTOSSStyleBot
    Ultra-fast, formulaic pot-odds, Kelly/ICM, blocker-aware tournament poker bot.
    Designed by flash_lite model tier.
    """
    name: str = "RivalGPTOSSStyle"

    def __init__(self, name: str = "RivalGPTOSSStyle") -> None:
        self.name = name

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        pass

    def on_hand_end(self, seat_idx: int, summary: dict) -> None:
        pass

    def act(self, state: dict) -> Tuple[str, int]:
        try:
            return self._act_internal(state)
        except Exception:
            to_call = state.get("to_call", 0)
            if to_call <= 0:
                return ("check", 0)
            return ("fold", 0)

    def _act_internal(self, state: dict) -> Tuple[str, int]:
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)
        pot = state["pot"]
        street_bets = state["street_bets"]
        to_call = state["to_call"]
        min_raise_add = state["min_raise_add"]
        street = state["street"]
        hole = state["hole_cards"]
        h0, h1 = hole[0], hole[1]
        board = state["board"]
        bb = max(1, state["big_blind"])
        hand_number = state.get("hand_number", 1)
        active = state["active_seats"]
        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)
        num_opponents = len(active_opps)
        villain_idx = max(active_opps, key=lambda o: (street_bets[o], stacks[o]))
        button = state.get("button_idx", 0)
        dist_from_btn = (button - hero_idx) & 7

        # 1. Preflop Logic
        if street == 0:
            pct = PREFLOP_PERCENTILE[h0][h1]
            effective_bb = (hero_stack + street_bets[hero_idx]) / float(bb)

            # Short stack push/fold Nash strategy or ICM pressure
            if effective_bb <= 14.0 and max(street_bets) <= bb:
                push_bb = NASH_PUSH_BB[h0][h1]
                if effective_bb <= push_bb:
                    return ("raise", hero_stack)

            # Facing a raise / all-in
            if max(street_bets) > bb:
                call_bb = to_call / float(bb)
                nash_call = NASH_CALL_BB[h0][h1]
                bf, _ = compute_bubble_factor(stacks, hero_idx, villain_idx, max(bb, to_call), max(bb, pot), hand_number, 200)

                if call_bb <= (nash_call / max(1.0, bf)) and pct >= 0.78:
                    if pct >= 0.92 and hero_stack > to_call * 3:
                        raise_amt = min(hero_stack, max(min_raise_add, to_call * 3))
                        return ("raise", int(raise_amt))
                    return ("call", int(to_call))
                if to_call <= 3 * bb and pct >= 0.75 and bf <= 1.35:
                    return ("call", int(to_call))
                return ("fold", 0)

            # Unopened or limped pot preflop
            if pct >= 0.74 or (pct >= 0.58 and dist_from_btn <= 2):
                raise_amt = min(hero_stack, max(min_raise_add, int(2.5 * bb) - street_bets[hero_idx]))
                return ("raise", int(raise_amt))

            if to_call == 0:
                return ("check", 0)
            if to_call <= bb and pct >= 0.55:
                return ("call", int(to_call))
            return ("fold", 0)

        # 2. Postflop Logic (Flop, Turn, River)
        made_cat, wetness, has_strong_draw, is_op_tptk, has_nut_blocker = analyze_board_texture(hole, board)

        eq_cur, eq_sm, fold_sm, eq_bg, fold_bg, eq_jm, fold_jm = conditional_postflop_profile(
            hole, board, num_opponents, num_samples=32
        )

        pot_odds = to_call / max(1.0, pot + to_call)
        bf, _ = compute_bubble_factor(stacks, hero_idx, villain_idx, max(to_call, bb), max(pot, bb), hand_number, 200)

        # Facing a bet / raise
        if to_call > 0:
            required_eq = pot_odds * bf
            if eq_cur >= required_eq + 0.06 or (made_cat >= 2 and eq_cur >= 0.60) or (street < 3 and has_strong_draw and eq_cur >= pot_odds * 0.85):
                if eq_cur >= 0.72 and hero_stack > to_call * 2 and random.random() < 0.65:
                    raise_amt = min(hero_stack, max(min_raise_add, to_call + int(pot * 0.75)))
                    return ("raise", raise_amt)
                return ("call", int(to_call))

            if street == 3 and has_nut_blocker and to_call <= pot * 0.4 and eq_cur >= 0.35 and made_cat >= 1:
                return ("call", int(to_call))

            return ("fold", 0)

        # Checked to Hero (Initiative or continuation betting)
        if eq_cur >= 0.62 or is_op_tptk or made_cat >= 2:
            bet_size = min(hero_stack, max(min_raise_add, int(pot * (0.65 if wetness > 0.5 else 0.50))))
            return ("raise", bet_size)

        if (street < 3 and has_strong_draw) or (has_nut_blocker and wetness >= 0.4 and random.random() < 0.50):
            bet_size = min(hero_stack, max(min_raise_add, int(pot * 0.55)))
            return ("raise", bet_size)

        return ("check", 0)
