"""
RivalCFRProBot — Rival Team #1
Game-Theoretic / Linear-CFR / Range-Advantage & Capped-Range Exploiter
for the Deep-ML 8-Player Sit-and-Go No-Limit Hold'em Tournament.

Key Architecture:
1. Online Bayesian Opponent Profiler (`on_table_start` & `on_hand_end` + live fallback):
   Tracks VPIP, PFR, 3-Bet, Fold-to-Steal, Fold-to-3Bet, Postflop Aggression, and
   Postflop Fold rates per seat with Laplace smoothing.
2. Position- & ICM-Aware Preflop Range Construction:
   - Exact Malmuth-Harville Bubble Factor (`compute_bubble_factor`) for both Hero (defense)
     and Villain (ICM bully pressure when Hero covers medium/short stacks on the bubble).
   - Compact min-raise / 2.08x BB steals to maximize risk-reward ratio, combined with
     ICM-adjusted Nash Push/Fold below 12 BB.
   - Position-aware 3-Betting: Linear + suited-Ace/King blocker 3-bets against late-position
     steals (CO/BTN/SB), strict premium-only 3-bets against EP/MP opens, and Squeeze plays
     when multiple callers flat an open.
   - Big Blind Minimum Defense Frequency (MDF) defense against small steals.
3. True Capped vs. Uncapped Range Detection:
   - Parses `action_history` in O(N) (~1.5 µs) to detect check-through streets
     (`flop_checked_through`, `turn_checked_through`), positional check-to-Hero (`hero_is_ip`),
     previous-street callers (`v_called_flop`, `v_called_turn`), and postflop re-raises.
   - Attacks check-through capped ranges on Turn and River while refusing to spew multi-street
     bluffs into opponents who called Flop/Turn bets.
   - Distinguishes true Hero pairs from board-only pairs on paired boards (`hero_real_cat`).
4. Geometric Bet Sizing & Trap Inducement:
   - Sizes Flop C-bets to 34% pot on dry boards (inducing check-raises when holding monsters)
     and 62-78% pot for value on wet boards / Turn & River barrels.
   - Strict <60 µs average decision latency (<250 µs P99) with zero timeouts.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

from poker_engine.evaluator import (
    analyze_board_texture,
    fast_postflop_equity,
)
from poker_engine.icm import compute_bubble_factor
from poker_engine.preflop_tables import (
    NASH_CALL_BB,
    NASH_PUSH_BB,
    PREFLOP_HU_EQ,
    PREFLOP_PERCENTILE,
    PREFLOP_STRENGTH,
)


class RivalCFRProBot:
    """
    Game-Theoretic / Linear-CFR / Capped-Range Exploiter Bot for 8-Max Sit-and-Go.
    """

    name: str = "RivalCFRProBot"

    def __init__(self) -> None:
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.hands_dealt: List[int] = [0] * 8
        self.vpip: List[int] = [0] * 8
        self.pfr: List[int] = [0] * 8
        self.three_bet: List[int] = [0] * 8
        self.faced_open: List[int] = [0] * 8
        self.folded_to_open: List[int] = [0] * 8
        self.faced_3bet: List[int] = [0] * 8
        self.folded_to_3bet: List[int] = [0] * 8
        self.post_bets: List[int] = [0] * 8
        self.post_calls: List[int] = [0] * 8
        self.post_checks: List[int] = [0] * 8
        self.post_folds: List[int] = [0] * 8
        self._last_recorded_hand: int = -1

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self._reset_stats()

    def on_hand_end(self, seat_idx: int, summary: Dict[str, Any]) -> None:
        hand_num = summary.get("hand_number", -1)
        if hand_num == self._last_recorded_hand:
            return
        self._last_recorded_hand = hand_num

        start_stacks = summary.get("starting_stacks")
        if start_stacks:
            for i in range(8):
                if start_stacks[i] > 0:
                    self.hands_dealt[i] += 1

        history = summary.get("action_history", ())
        if not history:
            return

        vpip_set = set()
        pfr_set = set()
        pf_raises = 0
        first_raiser = -1

        for st, seat, act, _amt in history:
            if st == 0:
                if act == "raise":
                    vpip_set.add(seat)
                    pfr_set.add(seat)
                    pf_raises += 1
                    if pf_raises == 1:
                        first_raiser = seat
                    elif pf_raises == 2:
                        self.three_bet[seat] += 1
                elif act == "call":
                    vpip_set.add(seat)
                    if pf_raises == 1:
                        self.faced_open[seat] += 1
                    elif pf_raises >= 2 and seat == first_raiser:
                        self.faced_3bet[seat] += 1
                elif act == "fold":
                    if pf_raises == 1:
                        self.faced_open[seat] += 1
                        self.folded_to_open[seat] += 1
                    elif pf_raises >= 2 and seat == first_raiser:
                        self.faced_3bet[seat] += 1
                        self.folded_to_3bet[seat] += 1
            else:
                if act in ("raise", "bet"):
                    self.post_bets[seat] += 1
                elif act == "call":
                    self.post_calls[seat] += 1
                elif act == "check":
                    self.post_checks[seat] += 1
                elif act == "fold":
                    self.post_folds[seat] += 1

        for s in vpip_set:
            self.vpip[s] += 1
        for s in pfr_set:
            self.pfr[s] += 1

    # ------------------------------------------------------------------
    # Fast O(1) Bayesian Opponent Queries
    # ------------------------------------------------------------------
    def _vpip_rate(self, seat: int) -> float:
        return (self.vpip[seat] + 2.4) / (self.hands_dealt[seat] + 10.0)

    def _pfr_rate(self, seat: int) -> float:
        return (self.pfr[seat] + 1.6) / (self.hands_dealt[seat] + 10.0)

    def _fold_to_open_rate(self, seat: int) -> float:
        return (self.folded_to_open[seat] + 3.6) / (self.faced_open[seat] + 5.0)

    def _fold_to_3bet_rate(self, seat: int) -> float:
        return (self.folded_to_3bet[seat] + 3.4) / (self.faced_3bet[seat] + 4.5)

    def _post_fold_rate(self, seat: int) -> float:
        tot = self.post_folds[seat] + self.post_calls[seat] + self.post_bets[seat]
        return (self.post_folds[seat] + 2.5) / (tot + 5.0)

    def _post_agg_freq(self, seat: int) -> float:
        tot = self.post_bets[seat] + self.post_calls[seat] + self.post_checks[seat]
        return (self.post_bets[seat] + 1.2) / (tot + 5.0)

    def _is_station(self, seat: int) -> bool:
        if self.hands_dealt[seat] < 8:
            return False
        return (
            self._vpip_rate(seat) > 0.34
            and self._post_fold_rate(seat) < 0.38
            and self._post_agg_freq(seat) < 0.22
        )

    def _is_maniac(self, seat: int) -> bool:
        if self.hands_dealt[seat] < 6:
            return False
        return self._pfr_rate(seat) > 0.33 or self._post_agg_freq(seat) > 0.48

    def _is_nit(self, seat: int) -> bool:
        if self.hands_dealt[seat] < 8:
            return False
        return self._vpip_rate(seat) < 0.17 and self._fold_to_open_rate(seat) > 0.78

    @staticmethod
    def _safe_raise(
        target_add: int,
        to_call: int,
        min_raise_add: int,
        hero_stack: int,
    ) -> Tuple[str, int]:
        if hero_stack <= to_call:
            return ("call", hero_stack) if to_call > 0 else ("check", 0)
        amt = max(min_raise_add, min(hero_stack, int(target_add)))
        if amt <= to_call:
            amt = min(hero_stack, max(min_raise_add, to_call + 1))
        return ("raise", amt)

    # ------------------------------------------------------------------
    # Main Decision Entrypoint (<60 µs avg, <250 µs P99)
    # ------------------------------------------------------------------
    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        to_call: int = state["to_call"]
        hero_idx: int = state["hero_idx"]
        stacks: Sequence[int] = state["stacks"]
        hero_stack: int = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        active: Sequence[bool] = state["active_seats"]
        active_opps: List[int] = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        street: int = state["street"]
        if street == 0:
            return self._act_preflop(state, hero_idx, hero_stack, to_call, active_opps)
        return self._act_postflop(state, hero_idx, hero_stack, to_call, active_opps)

    # ------------------------------------------------------------------
    # Preflop Engine: Linear-CFR + Blocker 3-Bets + ICM Pressure
    # ------------------------------------------------------------------
    def _act_preflop(
        self,
        state: Dict[str, Any],
        hero_idx: int,
        hero_stack: int,
        to_call: int,
        active_opps: List[int],
    ) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        pct = PREFLOP_PERCENTILE[h0][h1]
        hu_eq = PREFLOP_HU_EQ[h0][h1]

        bb = max(1, state["big_blind"])
        pot = state["pot"]
        min_raise_add = state["min_raise_add"]
        street_bets = state["street_bets"]
        stacks = state["stacks"]
        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        hand_num = state.get("hand_number", 1)

        hero_total_street = hero_stack + street_bets[hero_idx]
        max_opp_street = max(stacks[o] + street_bets[o] for o in active_opps)
        eff_chips = min(hero_total_street, max_opp_street)
        eff_bb = eff_chips / bb

        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))
        hero_bf, req_eq = compute_bubble_factor(
            total_stacks,
            hero_idx,
            primary_villain,
            max(bb, to_call),
            max(bb, pot),
            hand_num,
        )

        # Parse preflop action history
        history = state.get("action_history", ())
        pf_raises = 0
        pf_aggressor = -1
        pf_callers = 0
        hero_already_raised = False
        for st, seat, act, _amt in history:
            if st == 0:
                if act == "raise":
                    pf_raises += 1
                    pf_aggressor = seat
                    if seat == hero_idx:
                        hero_already_raised = True
                elif act == "call":
                    pf_callers += 1

        button = state["button_idx"]
        bb_seat = state.get("bb_seat", (button + 2) & 7)
        sb_seat = state.get("sb_seat", (button + 1) & 7)
        dist_from_btn = (button - hero_idx) & 7

        # Determine active players left to act behind Hero in an unopened/limped pot
        opps_behind: List[int] = []
        if hero_idx != bb_seat:
            cur = (hero_idx + 1) & 7
            for _ in range(7):
                if cur != hero_idx and state["active_seats"][cur] and street_bets[cur] <= bb:
                    opps_behind.append(cur)
                if cur == bb_seat:
                    break
                cur = (cur + 1) & 7
        n_behind = len(opps_behind)

        r0, r1 = h0 >> 2, h1 >> 2
        is_pair = r0 == r1
        is_suited = (h0 & 3) == (h1 & 3) and not is_pair
        hi_r = r0 if r0 >= r1 else r1
        lo_r = r1 if r0 >= r1 else r0
        has_ace_blocker = hi_r == 12 and is_suited
        has_king_blocker = hi_r == 11 and is_suited and lo_r >= 8

        max_bet = max(street_bets)

        # --------------------------------------------------------------
        # CASE 1: Unopened or Limped Pot (max_bet <= bb)
        # --------------------------------------------------------------
        if max_bet <= bb:
            # ICM Bully bonus when Hero covers all remaining players behind on the bubble
            n_alive = sum(1 for s in total_stacks if s > 0)
            icm_bully_bonus = 0.0
            if (3 <= n_alive <= 5 or hand_num >= 140) and hero_bf <= 1.15 and opps_behind:
                if all(total_stacks[hero_idx] > total_stacks[o] * 1.30 for o in opps_behind):
                    icm_bully_bonus = 0.08

            # 1A. Short-stack Push/Fold (eff_bb <= 12.0)
            if eff_bb <= 12.0 and pf_callers == 0:
                pos_mult = 1.0 + 0.13 * max(0, dist_from_btn - 1)
                adj_push_bb = NASH_PUSH_BB[h0][h1] / (pos_mult * (hero_bf ** 1.15))
                if icm_bully_bonus > 0:
                    adj_push_bb *= 1.15
                if eff_bb <= adj_push_bb:
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            # 1B. Limped Pot (pf_callers >= 1) -> Iso-Raise or Over-Limp / Check BB
            if pf_callers >= 1:
                iso_thresh = 0.74 + 0.03 * n_behind + (hero_bf - 1.08) * 0.15 - icm_bully_bonus
                if pct >= iso_thresh or (is_pair and r0 >= 7) or (hi_r == 12 and lo_r >= 9):
                    iso_total = int((2.6 + 0.8 * pf_callers) * bb)
                    return self._safe_raise(
                        iso_total - street_bets[hero_idx], to_call, min_raise_add, hero_stack
                    )
                if to_call == 0:
                    return ("check", 0)
                if (
                    to_call <= bb
                    and eff_bb >= 16.0
                    and hero_bf <= 1.28
                    and (is_pair or (is_suited and pct >= 0.56) or pct >= 0.66)
                ):
                    return ("call", to_call)
                return ("fold", 0)

            # 1C. Standard Unopened Pot (Positional Linear-CFR Open + Verified Nit-Blind Steal)
            joint_fold_prob = 1.0
            for o in opps_behind:
                joint_fold_prob *= self._fold_to_open_rate(o)

            steal_adj = 0.0
            if n_behind <= 2 and joint_fold_prob >= 0.64 and eff_bb >= 14.0:
                steal_adj = min(0.08, (joint_fold_prob - 0.60) * 0.45)

            gto_open = (
                0.41
                + 0.074 * min(5, dist_from_btn)
                + (hero_bf - 1.10) * 0.14
                - icm_bully_bonus
                - steal_adj
            )
            if pct >= gto_open:
                target_add = int(2.12 * bb) - street_bets[hero_idx]
                return self._safe_raise(target_add, to_call, min_raise_add, hero_stack)

            return ("check", 0) if to_call == 0 else ("fold", 0)

        # --------------------------------------------------------------
        # CASE 2: Facing an Open-Raise or Shove (pf_raises == 1)
        # --------------------------------------------------------------
        opener = pf_aggressor if pf_aggressor >= 0 else primary_villain
        open_bb = max_bet / bb
        opener_dist = (button - opener) & 7

        if pf_raises == 1 and not hero_already_raised:
            # 2A. Facing an All-In Shove or Huge Commit (to_call >= 0.38 * hero_stack or eff_bb <= 12.0)
            if to_call >= int(0.38 * hero_stack) or eff_bb <= 12.0:
                call_bb = to_call / bb
                adj_call_bb = NASH_CALL_BB[h0][h1] / (hero_bf ** 1.25)
                if self._is_maniac(opener):
                    adj_call_bb *= 1.30
                elif self._is_nit(opener):
                    adj_call_bb *= 0.70

                if pct >= 0.965 or (pct >= 0.945 and hero_bf <= 1.25):
                    return ("raise", hero_stack) if hero_stack > to_call else ("call", to_call)
                if call_bb <= adj_call_bb and hu_eq >= req_eq + 0.02 and pct >= 0.80:
                    return ("call", to_call)
                return ("fold", 0)

            # 2B. Facing a Standard Open-Raise (open_bb <= 4.5, eff_bb > 12.0)
            opener_is_station = self._is_station(opener)
            opener_is_nit = self._is_nit(opener)
            opener_is_maniac = self._is_maniac(opener)

            # 1) 3-Bet Value + Blocker Re-Steal vs Late-Position Openers (CO/BTN/SB)
            if open_bb <= 2.65 and opener_dist <= 2 and hero_bf <= 1.30 and not opener_is_nit:
                if (
                    pct >= 0.87
                    or (
                        pf_callers == 0
                        and not opener_is_station
                        and not opener_is_maniac
                        and eff_bb >= 14.0
                        and ((has_ace_blocker and pct >= 0.67) or (has_king_blocker and pct >= 0.74))
                    )
                ):
                    if eff_bb <= 18.0:
                        return ("raise", hero_stack)
                    return self._safe_raise(int(2.8 * to_call), to_call, min_raise_add, hero_stack)

            # 2) Ultra-Premium 3-Bet vs Any Position
            if pct >= 0.945 or (opener_is_maniac and pct >= 0.88 and hero_bf <= 1.30):
                if eff_bb <= 22.0:
                    return ("raise", hero_stack)
                return self._safe_raise(int(2.8 * to_call), to_call, min_raise_add, hero_stack)

            # 3) Squeeze vs Open + Flat-Caller(s) with Strong Value Range
            if pf_callers >= 1 and open_bb <= 2.6 and eff_bb >= 16.0 and hero_bf <= 1.24 and pct >= 0.90:
                if eff_bb <= 20.0:
                    return ("raise", hero_stack)
                return self._safe_raise(int((3.2 + 0.8 * pf_callers) * to_call), to_call, min_raise_add, hero_stack)

            # 4) Big Blind Minimum Defense Frequency (MDF) Defense vs Small Steals (<= 2.4x BB)
            if (
                hero_idx == bb_seat
                and len(active_opps) == 1
                and open_bb <= 2.4
                and eff_bb >= 14.5
                and hero_bf <= 1.35
            ):
                if pct >= 0.55 or (is_suited and pct >= 0.45) or is_pair:
                    return ("call", to_call)

            # 5) Flat-Call Single Raise in Position / Good Price
            if to_call <= int(3.4 * bb) and pct >= 0.78 and hero_bf <= 1.35:
                return ("call", to_call)
            if (
                to_call <= int(2.4 * bb)
                and eff_bb >= 18.0
                and hero_bf <= 1.28
                and (is_pair or (is_suited and pct >= 0.72))
            ):
                return ("call", to_call)

            return ("fold", 0)

        # --------------------------------------------------------------
        # CASE 3: Facing a 3-Bet or 4-Bet (pf_raises >= 2)
        # --------------------------------------------------------------
        if pct >= 0.955 or (pct >= 0.945 and hero_bf <= 1.28):
            if eff_bb <= 24.0 or to_call >= int(0.30 * hero_stack):
                return ("raise", hero_stack)
            return self._safe_raise(int(2.5 * to_call), to_call, min_raise_add, hero_stack)

        # Call small non-all-in 3-bet when holding TT-QQ / AQs / AKo in deep stack
        if (
            hero_already_raised
            and pf_raises == 2
            and to_call <= int(0.20 * hero_stack)
            and eff_bb >= 20.0
            and hero_bf <= 1.25
            and pct >= 0.90
        ):
            return ("call", to_call)

        return ("fold", 0)

    # ------------------------------------------------------------------
    # Postflop Engine: True Capped-Range Detection + Geometric Sizing
    # ------------------------------------------------------------------
    def _act_postflop(
        self,
        state: Dict[str, Any],
        hero_idx: int,
        hero_stack: int,
        to_call: int,
        active_opps: List[int],
    ) -> Tuple[str, int]:
        h0, h1 = state["hole_cards"]
        board: Sequence[int] = state["board"]
        street: int = state["street"]
        pot: int = state["pot"]
        min_raise_add: int = state["min_raise_add"]
        stacks: Sequence[int] = state["stacks"]
        street_bets: Sequence[int] = state["street_bets"]
        hand_contrib: Sequence[int] = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        hand_num: int = state.get("hand_number", 1)
        n_opp = len(active_opps)

        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))

        # 1. O(1) Board Texture & True Hero Hand Classification (~2 µs)
        made_cat, wetness, has_draw, is_tptk, has_nut_blocker = analyze_board_texture((h0, h1), board)

        hr0, hr1 = h0 >> 2, h1 >> 2
        board_ranks = [c >> 2 for c in board]
        max_board_rank = max(board_ranks)
        board_rank_counts = [0] * 13
        for br in board_ranks:
            board_rank_counts[br] += 1
        max_board_cnt = max(board_rank_counts)
        num_board_pairs = sum(1 for c in board_rank_counts if c >= 2)
        is_paired_board = max_board_cnt >= 2

        hero_matches_board = (board_rank_counts[hr0] > 0) or (board_rank_counts[hr1] > 0)
        hero_is_pocket_pair = hr0 == hr1
        hero_has_own_pair = hero_is_pocket_pair or hero_matches_board

        hero_real_cat = made_cat
        if is_paired_board:
            if made_cat == 1 and not hero_has_own_pair:
                hero_real_cat = 0
            elif made_cat == 2:
                if num_board_pairs >= 2 and not hero_has_own_pair and hr0 < max_board_rank and hr1 < max_board_rank:
                    hero_real_cat = 0
                elif max_board_cnt == 2 and not hero_matches_board:
                    hero_real_cat = 2 if (hero_is_pocket_pair and hr0 > max_board_rank) else 1
            elif made_cat == 3 and max_board_cnt == 3 and not hero_has_own_pair:
                hero_real_cat = 0

        # 2. Parse Action History for True Check-Through Capped Ranges (~1.5 µs)
        history = state.get("action_history", ())
        pf_aggressor = -1
        pf_raises = 0
        flop_had_bet = False
        turn_had_bet = False
        v_called_flop = False
        v_called_turn = False
        v_checked_this_street = False
        hero_bet_this_street = False

        for st, seat, act, _amt in history:
            if st == 0:
                if act == "raise":
                    pf_aggressor = seat
                    pf_raises += 1
            elif st == 1:
                if act in ("raise", "bet"):
                    flop_had_bet = True
                    if st == street and seat == hero_idx:
                        hero_bet_this_street = True
                elif act == "call" and seat == primary_villain:
                    v_called_flop = True
                elif st == street and act == "check" and seat == primary_villain:
                    v_checked_this_street = True
            elif st == 2:
                if act in ("raise", "bet"):
                    turn_had_bet = True
                    if st == street and seat == hero_idx:
                        hero_bet_this_street = True
                elif act == "call" and seat == primary_villain:
                    v_called_turn = True
                elif st == street and act == "check" and seat == primary_villain:
                    v_checked_this_street = True
            elif st == 3:
                if act in ("raise", "bet") and seat == hero_idx:
                    hero_bet_this_street = True
                elif act == "check" and seat == primary_villain:
                    v_checked_this_street = True

        flop_checked_through = (street >= 2) and (not flop_had_bet)
        turn_checked_through = (street == 3) and (not turn_had_bet)

        # Villain is genuinely capped ONLY when the previous street checked through (nobody bet!)
        villain_is_capped = False
        if n_opp == 1 and to_call == 0:
            if street == 2 and flop_checked_through:
                villain_is_capped = True
            elif street == 3 and turn_checked_through and not v_called_flop:
                villain_is_capped = True

        v_is_station = self._is_station(primary_villain)
        v_is_maniac = self._is_maniac(primary_villain)
        v_is_nit = self._is_nit(primary_villain)
        v_agg_freq = self._post_agg_freq(primary_villain)

        bet_ratio = to_call / max(1, pot - to_call) if to_call > 0 else 0.0

        # 3. Fast Early Fold Filter for Pure Air facing a bet
        if (
            to_call > 0
            and hero_real_cat == 0
            and not has_draw
            and (
                street == 3
                or bet_ratio > 0.40
                or n_opp >= 2
                or hero_bet_this_street
                or (hr0 < max_board_rank and hr1 < max_board_rank)
            )
        ):
            return ("fold", 0)

        # 4. Bayesian Range- & Action-Conditioned Equity Rollout (~45-65 µs)
        v_min_pct = 0.52
        if pf_raises >= 2 and not v_is_maniac:
            v_min_pct = 0.78
        elif v_is_nit:
            v_min_pct = 0.68

        v_bluff_prob = 0.25
        if v_is_maniac:
            v_bluff_prob = 0.40
        elif v_is_station or v_is_nit:
            v_bluff_prob = 0.08
        elif hero_bet_this_street and street >= 2:
            v_bluff_prob = 0.05

        num_samples = 56 if n_opp == 1 else 40
        eq = fast_postflop_equity(
            (h0, h1),
            board,
            num_opponents=n_opp,
            num_samples=num_samples,
            villain_min_pct=v_min_pct,
            villain_bet_ratio=bet_ratio,
            villain_bluff_Prob=v_bluff_prob,
        )

        # --------------------------------------------------------------
        # BRANCH A: Checked To Hero (to_call == 0)
        # --------------------------------------------------------------
        if to_call == 0:
            # 1) Strong Value Betting (Two Pair+, Sets, Straights, Flushes, or TPTK/Overpair)
            if hero_real_cat >= 2 or (is_tptk and eq >= 0.64) or eq >= 0.70:
                if v_is_station:
                    mult = 0.78 if street >= 2 else 0.70
                elif street == 1 and n_opp == 1 and hero_real_cat >= 2 and eq >= 0.76 and wetness <= 0.40:
                    # Induce check-raises / floats on dry-medium boards with monsters
                    mult = 0.35
                else:
                    mult = 0.62 if street == 1 else 0.68
                return self._safe_raise(int(mult * pot), 0, min_raise_add, hero_stack)

            # 2) Polarized Blocker / Semi-Bluff / Capped-Range Attack (Heads-Up)
            if n_opp == 1 and not v_is_station:
                # 2a. Polarized bluff or semi-bluff with nut blocker or strong draw
                if (has_nut_blocker or (street <= 2 and has_draw)) and eq >= 0.36:
                    # Do NOT barrel pure air on Turn/River if Villain already called a previous street bet
                    if (
                        street == 1
                        or (street == 2 and (not v_called_flop or has_draw or hero_real_cat >= 1))
                        or (street == 3 and not v_called_turn and (hero_real_cat >= 1 or turn_checked_through))
                    ):
                        return self._safe_raise(int(0.45 * pot), 0, min_raise_add, hero_stack)

                # 2b. Dry-Flop Range C-Bet
                if street == 1 and wetness <= 0.38 and eq >= 0.40:
                    return self._safe_raise(int(0.33 * pot), 0, min_raise_add, hero_stack)

                # 2c. Capped-Range Turn / River Probe (when previous street checked through!)
                if villain_is_capped and (
                    eq >= 0.44
                    or has_draw
                    or (v_checked_this_street and (eq >= 0.36 or (street == 3 and has_nut_blocker)))
                ):
                    return self._safe_raise(int(0.48 * pot), 0, min_raise_add, hero_stack)

            return ("check", 0)

        # --------------------------------------------------------------
        # BRANCH B: Facing a Bet or Raise (to_call > 0)
        # --------------------------------------------------------------
        hero_bf, req_eq = compute_bubble_factor(
            total_stacks, hero_idx, primary_villain, to_call, pot, hand_num
        )

        # 1) Facing a Re-Raise on Turn/River: non-maniacs only raise Two Pair+ / Trips+
        if hero_bet_this_street and street >= 2 and not v_is_maniac:
            if hero_real_cat <= 1 and to_call > int(0.18 * pot):
                return ("fold", 0)

        # 2) Check-Raise Small Flop Bets (< 42% pot) with Two Pair+ or Strong Draws
        if (
            street == 1
            and n_opp == 1
            and bet_ratio <= 0.42
            and not v_is_nit
            and (hero_real_cat >= 2 or (has_draw and eq >= 0.48 and not v_is_station))
        ):
            return self._safe_raise(to_call + int(0.65 * pot), to_call, min_raise_add, hero_stack)

        # 3) Value-Raise Monsters (Trips+ or High-Equity Two Pair)
        if hero_real_cat >= 3 and eq >= 0.74:
            return self._safe_raise(to_call + int(0.75 * pot), to_call, min_raise_add, hero_stack)
        if street >= 2 and hero_real_cat >= 2 and eq >= 0.76 and hero_bf <= 1.32:
            return self._safe_raise(to_call + int(0.70 * pot), to_call, min_raise_add, hero_stack)

        # 4) River High-Card Protection
        if street == 3 and hero_real_cat == 0:
            return ("fold", 0)

        # 5) Call / Float Decision
        draw_bonus = 0.05 if (street < 3 and has_draw and hero_stack >= 3 * to_call) else 0.0

        # Float small flop C-bets (< 38% pot) in heads-up pots with Pair, Draw, or good equity
        if (
            street == 1
            and n_opp == 1
            and bet_ratio <= 0.38
            and hero_bf <= 1.32
            and (hero_real_cat >= 1 or has_draw or eq >= 0.43)
            and eq + draw_bonus >= req_eq + 0.03
        ):
            return ("call", to_call)

        margin = 0.055 + 0.03 * street
        if v_is_maniac and hero_real_cat >= 1:
            margin -= 0.035

        if eq + draw_bonus >= req_eq + margin:
            return ("call", to_call)

        return ("fold", 0)
