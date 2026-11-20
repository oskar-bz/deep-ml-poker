"""
Realistic Opponent Archetypes representing the full spectrum of competitors in the
Deep-ML No-Limit Hold'em Tournament — including Prepared LLM / Solver / Counter Bots:

Classical / Unprepared Field:
1. TimeoutFoldBot: Simulates bots that time out (>1ms) or ultra-nits (~5% VPIP).
2. CallingStationBot: Loose-passive bot that calls too wide preflop and postflop.
3. TightPassiveNitBot: Tight-passive bot (~16% VPIP) that folds to pressure without top pair+.
4. ManiacAggroBot: Hyper-aggressive bot that shoves/raises wide and blasts pots.
5. HeuristicEquityBot: In-session 60-minute LLM bot using raw equity vs pot odds without ICM.
6. SharkTAGBot: Strong position-aware Tight-Aggressive + Push/Fold bot.

Prepared LLM / Solver Competitor Field:
7. LLM_ICM_ExploiterBot: Prepared competitor with exact (10, 5, 2) ICM Bubble Factor math,
   online VPIP/PFR/Fold opponent tracking, min-raise blind stealing, and range-filtered equity.
8. LLM_CFR_GTO_Bot: Prepared competitor using Minimum Defense Frequency (MDF), 3-bet light
   re-steals with suited Ace blockers against late-position opens, check-raise traps vs small
   C-bets, and polarized multi-street barreling.
9. LLM_TrapCounterBot: Anti-meta counter-strategy bot designed to punish blind-stealers and
   small flop C-bettors via flat-floating, turn probe betting, and slow-playing monsters.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Sequence, Tuple

from poker_engine.evaluator import analyze_board_texture, eval_cards, fast_postflop_equity
from poker_engine.icm import compute_bubble_factor
from poker_engine.preflop_tables import NASH_CALL_BB, NASH_PUSH_BB, PREFLOP_HU_EQ, PREFLOP_PERCENTILE


class BaseBot:
    name: str = "BaseBot"

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        pass

    def on_hand_end(self, seat_idx: int, summary: Dict[str, Any]) -> None:
        pass

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        return ("check", 0) if state["to_call"] == 0 else ("fold", 0)


class TimeoutFoldBot(BaseBot):
    """Represents bots that time out or only play super-premium monsters (AA/KK/QQ/AK)."""
    name = "TimeoutFoldBot"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        pct = PREFLOP_PERCENTILE[h0][h1]
        to_call = state["to_call"]
        if pct >= 0.975:
            return ("raise", state["max_raise_add"])
        return ("check", 0) if to_call == 0 else ("fold", 0)


class CallingStationBot(BaseBot):
    """Loose-passive calling station: calls wide preflop and postflop, rarely raises."""
    name = "CallingStationBot"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        bb = state["big_blind"]
        stack = state["stacks"][state["hero_idx"]]
        pct = PREFLOP_PERCENTILE[h0][h1]

        if street == 0:
            if pct >= 0.96:
                return ("raise", min(stack, max(state["min_raise_add"], to_call + 2 * bb)))
            if to_call == 0:
                return ("check", 0)
            if pct >= 0.42 and to_call <= max(4 * bb, int(0.35 * stack)):
                return ("call", to_call)
            if pct >= 0.78:
                return ("call", to_call)
            return ("fold", 0)

        board = state["board"]
        score = eval_cards([h0, h1] + list(board))
        cat = score >> 20
        if cat >= 2:
            target = max(state["min_raise_add"], to_call + int(0.6 * state["pot"]))
            return ("raise", min(stack, target))
        if to_call == 0:
            return ("check", 0)
        if cat >= 1 or (PREFLOP_HU_EQ[h0][h1] > 0.60 and to_call <= state["pot"] // 2):
            return ("call", to_call)
        return ("fold", 0)


class TightPassiveNitBot(BaseBot):
    """Plays top ~16% of hands, gives up easily postflop when facing aggression."""
    name = "TightPassiveNitBot"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        bb = state["big_blind"]
        stack = state["stacks"][state["hero_idx"]]
        pct = PREFLOP_PERCENTILE[h0][h1]

        if street == 0:
            if pct >= 0.93:
                return ("raise", min(stack, max(state["min_raise_add"], to_call + int(2.5 * bb))))
            if pct >= 0.84 and to_call <= 3 * bb:
                if to_call == 0:
                    return ("raise", min(stack, max(state["min_raise_add"], int(2.2 * bb))))
                return ("call", to_call)
            return ("check", 0) if to_call == 0 else ("fold", 0)

        made_cat, _, has_draw, is_tptk, _ = analyze_board_texture((h0, h1), state["board"])
        if made_cat >= 2 or (is_tptk and to_call <= state["pot"]):
            if to_call == 0:
                return ("raise", min(stack, max(state["min_raise_add"], int(0.5 * state["pot"]))))
            if made_cat >= 2 or to_call <= int(0.65 * state["pot"]):
                return ("call", to_call)
        if to_call == 0:
            return ("check", 0)
        if has_draw and to_call <= state["pot"] // 3:
            return ("call", to_call)
        return ("fold", 0)


class ManiacAggroBot(BaseBot):
    """Hyper-aggressive bot that raises/shoves wide and applies constant big-bet pressure."""
    name = "ManiacAggroBot"

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        bb = state["big_blind"]
        stack = state["stacks"][state["hero_idx"]]
        pct = PREFLOP_PERCENTILE[h0][h1]

        if street == 0:
            eff_bb = stack / max(1, bb)
            if eff_bb <= 15 and pct >= 0.52:
                return ("raise", stack)
            if pct >= 0.55 or (to_call <= bb and self.rng.random() < 0.35):
                target = to_call + int(3.0 * bb)
                if pct >= 0.85 and to_call > 2 * bb:
                    target = stack
                return ("raise", min(stack, max(state["min_raise_add"], target)))
            if to_call <= 2 * bb and pct >= 0.38:
                return ("call", to_call)
            return ("check", 0) if to_call == 0 else ("fold", 0)

        made_cat, _, has_draw, _, _ = analyze_board_texture((h0, h1), state["board"])
        if made_cat >= 1 or has_draw or self.rng.random() < 0.42:
            if to_call == 0:
                bet = max(state["min_raise_add"], int(0.80 * state["pot"]))
                return ("raise", min(stack, bet))
            if made_cat >= 2 or (has_draw and self.rng.random() < 0.5):
                return ("raise", stack)
            if made_cat >= 1 and to_call <= state["pot"]:
                return ("call", to_call)
        return ("check", 0) if to_call == 0 else ("fold", 0)


class HeuristicEquityBot(BaseBot):
    """In-session 60-minute AI-assisted bot: raw equity vs pot odds without ICM."""
    name = "HeuristicEquityBot"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = state["big_blind"]
        stack = state["stacks"][state["hero_idx"]]
        n_active = sum(1 for a in state["active_seats"] if a)
        n_opp = max(1, n_active - 1)

        if street == 0:
            pct = PREFLOP_PERCENTILE[h0][h1]
            hu_eq = PREFLOP_HU_EQ[h0][h1]
            eff_bb = stack / max(1, bb)
            if eff_bb <= 10.0 and pct >= 0.72:
                return ("raise", stack)
            if to_call <= bb:
                if pct >= 0.72:
                    return ("raise", min(stack, max(state["min_raise_add"], int(2.5 * bb))))
                return ("check", 0) if to_call == 0 else ("fold", 0)
            pot_odds = to_call / max(1, pot + to_call)
            adj_eq = hu_eq ** (1.0 + 0.28 * (n_opp - 1))
            if pct >= 0.92:
                return ("raise", min(stack, max(state["min_raise_add"], to_call + int(2.5 * to_call))))
            if adj_eq >= pot_odds + 0.08:
                return ("call", to_call)
            return ("fold", 0)

        eq = fast_postflop_equity((h0, h1), state["board"], num_opponents=n_opp, num_samples=45)
        if to_call == 0:
            if eq >= 0.62:
                bet = max(state["min_raise_add"], int(0.65 * pot))
                return ("raise", min(stack, bet))
            return ("check", 0)

        pot_odds = to_call / max(1, pot + to_call)
        if eq >= 0.78:
            target = max(state["min_raise_add"], to_call + int(0.75 * pot))
            return ("raise", min(stack, target))
        if eq >= pot_odds + 0.06:
            return ("call", to_call)
        return ("fold", 0)


class SharkTAGBot(BaseBot):
    """Strong baseline tournament TAG bot with Nash push/fold and dry-flop C-bets."""
    name = "SharkTAGBot"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = state["big_blind"]
        hero_idx = state["hero_idx"]
        stack = state["stacks"][hero_idx]
        active = state["active_seats"]
        n_active = sum(1 for a in active if a)
        n_opp = max(1, n_active - 1)

        if street == 0:
            pct = PREFLOP_PERCENTILE[h0][h1]
            eff_bb = (stack + state["street_bets"][hero_idx]) / max(1, bb)
            button = state["button_idx"]
            dist_from_btn = (button - hero_idx) & 7

            if eff_bb <= 12.5:
                if to_call <= bb:
                    req_push_bb = NASH_PUSH_BB[h0][h1] / (1.0 + 0.14 * max(0, dist_from_btn - 1))
                    if eff_bb <= req_push_bb:
                        return ("raise", stack)
                    return ("check", 0) if to_call == 0 else ("fold", 0)
                else:
                    call_bb = to_call / max(1, bb)
                    if call_bb <= NASH_CALL_BB[h0][h1] * 0.85:
                        return ("call", to_call)
                    return ("fold", 0)

            open_threshold = 0.62 + 0.04 * min(5, dist_from_btn)
            if to_call <= bb:
                if pct >= open_threshold:
                    return ("raise", min(stack, max(state["min_raise_add"], int(2.2 * bb))))
                return ("check", 0) if to_call == 0 else ("fold", 0)

            if pct >= 0.92:
                return ("raise", min(stack, max(state["min_raise_add"], int(2.8 * to_call))))
            if pct >= max(0.78, open_threshold + 0.08) and to_call <= 4 * bb:
                return ("call", to_call)
            return ("fold", 0)

        made_cat, wetness, has_draw, is_tptk, _ = analyze_board_texture((h0, h1), state["board"])
        eq = fast_postflop_equity(
            (h0, h1), state["board"], num_opponents=n_opp, num_samples=50, villain_min_pct=0.60
        )

        if to_call == 0:
            if eq >= 0.62 or made_cat >= 2 or is_tptk:
                bet = max(state["min_raise_add"], int(0.55 * pot))
                return ("raise", min(stack, bet))
            if street == 1 and n_opp == 1 and wetness < 0.35 and (eq >= 0.42 or has_draw):
                bet = max(state["min_raise_add"], int(0.38 * pot))
                return ("raise", min(stack, bet))
            return ("check", 0)

        pot_odds = to_call / max(1, pot + to_call)
        if eq >= 0.76 or made_cat >= 3:
            target = max(state["min_raise_add"], to_call + int(0.70 * pot))
            return ("raise", min(stack, target))
        if eq >= pot_odds + 0.06:
            return ("call", to_call)
        return ("fold", 0)


class LLM_ICM_ExploiterBot(BaseBot):
    """
    Prepared LLM Competitor #1:
    Uses exact (10, 5, 2) ICM Bubble Factor math, VPIP/PFR/Fold opponent tracking,
    min-raise blind stealing, and range-filtered Monte Carlo equity (equivalent to ApexBot v2).
    """
    name = "LLM_ICM_Exploiter"

    def __init__(self) -> None:
        self.hands_seen = [0] * 8
        self.vpip = [0] * 8
        self.pfr = [0] * 8
        self.faced_raise = [0] * 8
        self.folded_raise = [0] * 8
        self._last_hand = -1

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self.hands_seen = [0] * 8
        self.vpip = [0] * 8
        self.pfr = [0] * 8
        self.faced_raise = [0] * 8
        self.folded_raise = [0] * 8
        self._last_hand = -1

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        to_call = state["to_call"]
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        hand_num = state.get("hand_number", 1)
        street_bets = state["street_bets"]
        bb = max(1, state["big_blind"])
        active = state["active_seats"]

        if hand_num != self._last_hand:
            self._last_hand = hand_num
            for i in range(8):
                if stacks[i] + street_bets[i] > 0:
                    self.hands_seen[i] += 1
                    if street_bets[i] > bb:
                        self.vpip[i] += 1
                        self.pfr[i] += 1
                    elif not active[i]:
                        self.faced_raise[i] += 1
                        self.folded_raise[i] += 1

        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))
        h0, h1 = state["hole_cards"]
        pct = PREFLOP_PERCENTILE[h0][h1]
        pot = state["pot"]
        min_raise_add = state["min_raise_add"]

        if state["street"] == 0:
            eff_chips = min(hero_stack + street_bets[hero_idx], max(stacks[o] + street_bets[o] for o in active_opps))
            eff_bb = eff_chips / bb
            bf, req_eq = compute_bubble_factor(
                total_stacks, hero_idx, primary_villain, max(bb, to_call), max(bb, pot), hand_num
            )
            bb_seat = state.get("bb_seat", (state["button_idx"] + 2) & 7)
            opps_behind = []
            if hero_idx != bb_seat:
                cur = (hero_idx + 1) & 7
                for _ in range(7):
                    if cur in active_opps and street_bets[cur] <= bb:
                        opps_behind.append(cur)
                    if cur == bb_seat:
                        break
                    cur = (cur + 1) & 7

            if max(street_bets) <= bb:
                avg_fold = (
                    sum((self.folded_raise[o] + 3.2) / (self.faced_raise[o] + 5.0) for o in opps_behind)
                    / max(1, len(opps_behind))
                )
                if len(opps_behind) <= 3 and avg_fold >= 0.74 and eff_bb >= 6.0 and pct >= 0.22:
                    return ("raise", min(hero_stack, max(min_raise_add, int(2.0 * bb) - street_bets[hero_idx])))
                if eff_bb <= 10.5 and eff_bb <= NASH_PUSH_BB[h0][h1] / (bf ** 1.2) and pct >= 0.58:
                    return ("raise", hero_stack)
                open_pct = 0.50 + 0.065 * len(opps_behind) + (bf - 1.1) * 0.16
                if pct >= open_pct:
                    return ("raise", min(hero_stack, max(min_raise_add, int(2.1 * bb) - street_bets[hero_idx])))
                return ("check", 0) if to_call == 0 else ("fold", 0)

            if pct >= 0.945:
                return ("raise", hero_stack if eff_bb <= 22 else min(hero_stack, max(min_raise_add, int(2.7 * to_call))))
            if to_call < int(0.22 * hero_stack) and pct >= 0.78 and bf <= 1.35:
                return ("call", to_call)
            return ("fold", 0)

        made_cat, wetness, has_draw, is_tptk, _ = analyze_board_texture((h0, h1), state["board"])
        eq = fast_postflop_equity((h0, h1), state["board"], len(active_opps), 55, villain_min_pct=0.50)
        if to_call == 0:
            if made_cat >= 2 or (is_tptk and eq >= 0.65) or eq >= 0.72:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.58 * pot))))
            if state["street"] == 1 and len(active_opps) == 1 and wetness <= 0.36 and eq >= 0.42:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.33 * pot))))
            return ("check", 0)

        _, req_eq = compute_bubble_factor(total_stacks, hero_idx, primary_villain, to_call, pot, hand_num)
        margin = 0.08 + 0.04 * state["street"]
        if made_cat >= 3 and eq >= 0.75:
            return ("raise", min(hero_stack, max(min_raise_add, to_call + int(0.70 * pot))))
        if eq + (0.05 if has_draw and state["street"] < 3 else 0.0) >= req_eq + margin:
            return ("call", to_call)
        return ("fold", 0)


class LLM_CFR_GTO_Bot(BaseBot):
    """
    Prepared LLM Competitor #2 (GTO / Pseudo-CFR / MDF & Blocker Bot):
    - Uses balanced positional opens + 3-bet light re-steals with suited Ace/King blockers
      against late-position stealers.
    - Defends Big Blind using Minimum Defense Frequency (MDF) against min-raises.
    - Check-raises dry flops with draws and two-pair+ against small C-bets.
    - Fires polarized Turn/River barrels when holding nut blockers.
    """
    name = "LLM_CFR_GTO"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = max(1, state["big_blind"])
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        street_bets = state["street_bets"]
        active = state["active_seats"]
        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        n_opp = len(active_opps)
        pct = PREFLOP_PERCENTILE[h0][h1]
        min_raise_add = state["min_raise_add"]
        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))
        hand_num = state.get("hand_number", 1)

        if street == 0:
            eff_chips = min(hero_stack + street_bets[hero_idx], max(stacks[o] + street_bets[o] for o in active_opps))
            eff_bb = eff_chips / bb
            bf, req_eq = compute_bubble_factor(
                total_stacks, hero_idx, primary_villain, max(bb, to_call), max(bb, pot), hand_num
            )
            button = state["button_idx"]
            bb_seat = state.get("bb_seat", (button + 2) & 7)
            dist_from_btn = (button - hero_idx) & 7

            # Short-stack Nash push/fold
            if eff_bb <= 12.0 and max(street_bets) <= bb:
                if eff_bb <= NASH_PUSH_BB[h0][h1] / ((1.0 + 0.13 * max(0, dist_from_btn - 1)) * (bf ** 1.15)):
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            # Unopened pot: positional GTO open
            if max(street_bets) <= bb:
                gto_open = 0.42 + 0.075 * min(5, dist_from_btn) + (bf - 1.1) * 0.14
                if pct >= gto_open:
                    return ("raise", min(hero_stack, max(min_raise_add, int(2.15 * bb) - street_bets[hero_idx])))
                return ("check", 0) if to_call == 0 else ("fold", 0)

            # Facing an open raise: check if opener is in late position (CO/BTN/SB)
            villain_dist = (button - primary_villain) & 7
            raise_bb = max(street_bets) / bb
            r0, r1 = h0 >> 2, h1 >> 2
            is_suited = (h0 & 3) == (h1 & 3)
            has_ace_blocker = (r0 == 12 or r1 == 12) and is_suited

            # 3-bet value + 3-bet light re-steal with suited Ace blockers vs late-position opens!
            if raise_bb <= 2.6 and villain_dist <= 2 and bf <= 1.30:
                if pct >= 0.87 or (has_ace_blocker and pct >= 0.68 and eff_bb >= 14.0):
                    if eff_bb <= 18.0:
                        return ("raise", hero_stack)
                    return ("raise", min(hero_stack, max(min_raise_add, int(2.8 * to_call))))

            if pct >= 0.945:
                return ("raise", hero_stack if eff_bb <= 22.0 else min(hero_stack, max(min_raise_add, int(2.8 * to_call))))

            # BB Minimum Defense Frequency (MDF) defense against small steals (<= 2.3x BB)
            if hero_idx == bb_seat and n_opp == 1 and raise_bb <= 2.3 and eff_bb >= 15.0 and bf <= 1.35:
                if pct >= 0.56 or (is_suited and pct >= 0.46) or r0 == r1:
                    return ("call", to_call)

            if to_call <= 3.5 * bb and pct >= 0.78 and bf <= 1.35:
                return ("call", to_call)
            return ("fold", 0)

        # Postflop GTO + Blocker play
        made_cat, wetness, has_draw, is_tptk, has_nut_blocker = analyze_board_texture((h0, h1), state["board"])
        bet_ratio = to_call / max(1, pot - to_call) if to_call > 0 else 0.0
        eq = fast_postflop_equity(
            (h0, h1), state["board"], n_opp, 64, villain_min_pct=0.52, villain_bet_ratio=bet_ratio
        )

        if to_call == 0:
            if made_cat >= 2 or (is_tptk and eq >= 0.64) or eq >= 0.70:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.62 * pot))))
            # Polarized bluff with nut blocker or strong draw heads-up
            if n_opp == 1 and (has_nut_blocker or (street <= 2 and has_draw)) and eq >= 0.36:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.45 * pot))))
            if street == 1 and n_opp == 1 and wetness <= 0.38 and eq >= 0.40:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.33 * pot))))
            return ("check", 0)

        # Facing a bet: Check-raise small flop bets (< 40% pot) with strong draws or 2-pair+
        _, req_eq = compute_bubble_factor(total_stacks, hero_idx, primary_villain, to_call, pot, hand_num)
        if street == 1 and n_opp == 1 and bet_ratio <= 0.42 and (made_cat >= 2 or (has_draw and eq >= 0.48)):
            return ("raise", min(hero_stack, max(min_raise_add, to_call + int(0.65 * pot))))
        if made_cat >= 3 and eq >= 0.74:
            return ("raise", min(hero_stack, max(min_raise_add, to_call + int(0.75 * pot))))

        draw_bonus = 0.05 if (street < 3 and has_draw and hero_stack >= 3 * to_call) else 0.0
        if eq + draw_bonus >= req_eq + 0.06 + 0.03 * street:
            if street == 3 and made_cat == 0:
                return ("fold", 0)
            return ("call", to_call)
        return ("fold", 0)


class LLM_TrapCounterBot(BaseBot):
    """
    Prepared LLM Competitor #3 (Anti-Steal / Anti-CBet Trap Bot):
    - Defends blinds and button by flat-calling steals and floating small flop C-bets.
    - Slow-plays monsters (Trips/Full House/Two Pair) on flop to check-raise turn.
    - Fires probe bets on turn/river whenever the preflop aggressor checks back.
    """
    name = "LLM_TrapCounter"

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        street = state["street"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = max(1, state["big_blind"])
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        street_bets = state["street_bets"]
        active = state["active_seats"]
        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        n_opp = len(active_opps)
        pct = PREFLOP_PERCENTILE[h0][h1]
        min_raise_add = state["min_raise_add"]
        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))
        hand_num = state.get("hand_number", 1)

        if street == 0:
            eff_chips = min(hero_stack + street_bets[hero_idx], max(stacks[o] + street_bets[o] for o in active_opps))
            eff_bb = eff_chips / bb
            bf, _ = compute_bubble_factor(
                total_stacks, hero_idx, primary_villain, max(bb, to_call), max(bb, pot), hand_num
            )
            if max(street_bets) <= bb:
                if eff_bb <= 11.0 and eff_bb <= NASH_PUSH_BB[h0][h1] / (bf ** 1.2) and pct >= 0.56:
                    return ("raise", hero_stack)
                if pct >= 0.64:
                    return ("raise", min(hero_stack, max(min_raise_add, int(2.1 * bb) - street_bets[hero_idx])))
                return ("check", 0) if to_call == 0 else ("fold", 0)

            # Trap with AA/KK in position when stacks > 20 BB by flat-calling preflop!
            if pct >= 0.985 and eff_bb >= 20.0 and n_opp == 1 and to_call <= 3 * bb:
                return ("call", to_call)
            if pct >= 0.94:
                return ("raise", hero_stack if eff_bb <= 20.0 else min(hero_stack, max(min_raise_add, int(2.8 * to_call))))
            if to_call <= 2.5 * bb and pct >= 0.62 and eff_bb >= 14.0 and bf <= 1.35:
                return ("call", to_call)
            return ("fold", 0)

        made_cat, wetness, has_draw, is_tptk, _ = analyze_board_texture((h0, h1), state["board"])
        bet_ratio = to_call / max(1, pot - to_call) if to_call > 0 else 0.0
        eq = fast_postflop_equity(
            (h0, h1), state["board"], n_opp, 60, villain_min_pct=0.50, villain_bet_ratio=bet_ratio
        )

        if to_call == 0:
            # Slow-play monsters on dry flop to trap
            if street == 1 and made_cat >= 3 and wetness <= 0.35 and n_opp == 1:
                return ("check", 0)
            if made_cat >= 2 or is_tptk or eq >= 0.66:
                return ("raise", min(hero_stack, max(min_raise_add, int(0.65 * pot))))
            # Probe bet on Turn when checked to in heads-up pot
            if street == 2 and n_opp == 1 and (eq >= 0.44 or has_draw):
                return ("raise", min(hero_stack, max(min_raise_add, int(0.48 * pot))))
            return ("check", 0)

        _, req_eq = compute_bubble_factor(total_stacks, hero_idx, primary_villain, to_call, pot, hand_num)
        # Spring the trap on Turn/River with Two Pair+!
        if street >= 2 and made_cat >= 2 and eq >= 0.72:
            return ("raise", min(hero_stack, max(min_raise_add, to_call + int(0.85 * pot))))
        # Float small flop C-bets (< 38% pot) wider in heads-up pots
        if street == 1 and n_opp == 1 and bet_ratio <= 0.38 and (made_cat >= 1 or has_draw or eq >= 0.42):
            return ("call", to_call)
        if eq + (0.05 if street < 3 and has_draw else 0.0) >= req_eq + 0.08 + 0.03 * street:
            if street == 3 and made_cat == 0:
                return ("fold", 0)
            return ("call", to_call)
        return ("fold", 0)
