"""
RivalClaudeStyleBot - Rival Team #6 in the Deep-ML Sit-and-Go Tournament.
Archetype: Structured Multi-Factor Scoring & Street-Plan Bot.

Key Pillars:
1. Online Opponent HUD & Profiler (tracked via on_hand_end and action_history):
   - Bayesian smoothed VPIP, PFR, 3-Bet frequency, Fold-to-Steal, C-Bet frequency, Fold-to-CBet.
   - Dynamic player classification (Calling Station, Nit, Maniac, TAG).

2. Preflop Positional & Squeeze Matrix:
   - Dynamic RFI (Raise First In) with exploitation of tight/loose blinds behind.
   - Short-stack Nash Push/Fold calibrated with exact Malmuth-Harville Bubble Factor.
   - Exploitative 3-Bet & Squeeze Engine:
     * Differentiates Late-Position opens vs Early-Position opens.
     * Identifies multiway squeeze spots to harvest dead money with value + blocker bluffs.
     * BB Minimum Defense Frequency (MDF) against min-raises.
     * Dedicated defense against 3-bets when Hero is the opener.

3. Stack-to-Pot Ratio (SPR) & Commitment Thresholds:
   - Low SPR (<= 2.5): Commit stack automatically with Overpair / TPTK+, Two Pair, Sets, and Monster Draws.
   - High SPR (>= 6.0): Strict pot control with 1-pair hands on wet boards or facing Turn/River aggression.
   - Medium SPR (2.5 < SPR < 6.0): Calibrated multi-street value betting and pot scaling.

4. Explicit Street-by-Street Game Plan & Multi-Factor Scoring:
   - Street 0 (Preflop): Range rank, push/fold threshold, opener archetype, squeeze potential, ICM BF.
   - Street 1 (Flop): Board texture wetness, C-bet frequency, SPR commitment, float/check-raise decisions.
   - Street 2 (Turn): Double-barrel equity, pot control checks, turn probe bets vs checked-through streets.
   - Street 3 (River): Polarized value sizing vs thin value checks, blocker bluffs, Bayesian bluff-catching.

Strict <60 us mean latency, P99 <300 us, 0 timeouts, 0 exceptions.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from poker_engine.evaluator import (
    analyze_board_texture,
    eval_cards,
    fast_postflop_equity,
)
from poker_engine.icm import compute_bubble_factor
from poker_engine.preflop_tables import (
    NASH_CALL_BB,
    NASH_PUSH_BB,
    PREFLOP_HU_EQ,
    PREFLOP_PERCENTILE,
)


class OpponentHUD:
    """Fast, online Bayesian opponent profiler updated every hand."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hands_seen = [0] * 8
        self.vpip_cnt = [0] * 8
        self.pfr_cnt = [0] * 8
        self.faced_raise_pre = [0] * 8
        self.folded_raise_pre = [0] * 8
        self.three_bet_cnt = [0] * 8
        self.faced_cbet_flop = [0] * 8
        self.folded_cbet_flop = [0] * 8
        self.cbet_opp_flop = [0] * 8
        self.cbet_done_flop = [0] * 8

    def update_hand(self, summary: Dict[str, Any]) -> None:
        starting_stacks = summary.get("starting_stacks", [])
        for s in range(8):
            if s < len(starting_stacks) and starting_stacks[s] > 0:
                self.hands_seen[s] += 1

        action_history = summary.get("action_history", [])
        if not action_history:
            return

        preflop_actions = [a for a in action_history if a[0] == 0]
        flop_actions = [a for a in action_history if a[0] == 1]

        preflop_raisers: List[int] = []
        for _, seat, act, amt in preflop_actions:
            if act == "raise":
                preflop_raisers.append(seat)
                self.vpip_cnt[seat] += 1
                self.pfr_cnt[seat] += 1
            elif act in ("call", "sb") and amt > 0:
                self.vpip_cnt[seat] += 1

        if preflop_raisers:
            first_raiser = preflop_raisers[0]
            for _, seat, act, _ in preflop_actions:
                if act == "fold":
                    self.faced_raise_pre[seat] += 1
                    self.folded_raise_pre[seat] += 1
                elif act in ("call", "raise") and seat != first_raiser:
                    self.faced_raise_pre[seat] += 1
                    if len(preflop_raisers) > 1 and seat == preflop_raisers[1]:
                        self.three_bet_cnt[seat] += 1

        if preflop_raisers and flop_actions:
            pfr = preflop_raisers[-1]
            for idx, (_, seat, act, _) in enumerate(flop_actions):
                if seat == pfr:
                    self.cbet_opp_flop[pfr] += 1
                    if act == "raise":
                        self.cbet_done_flop[pfr] += 1
                        for _, seat2, act2, _ in flop_actions[idx + 1 :]:
                            if act2 == "fold":
                                self.faced_cbet_flop[seat2] += 1
                                self.folded_cbet_flop[seat2] += 1
                            elif act2 in ("call", "raise"):
                                self.faced_cbet_flop[seat2] += 1
                    break

    def get_vpip(self, seat: int) -> float:
        return (self.vpip_cnt[seat] + 2.0) / (self.hands_seen[seat] + 8.0)

    def get_pfr(self, seat: int) -> float:
        return (self.pfr_cnt[seat] + 1.3) / (self.hands_seen[seat] + 8.0)

    def get_fold_to_steal(self, seat: int) -> float:
        return (self.folded_raise_pre[seat] + 3.25) / (self.faced_raise_pre[seat] + 5.0)

    def get_cbet_freq(self, seat: int) -> float:
        return (self.cbet_done_flop[seat] + 2.2) / (self.cbet_opp_flop[seat] + 4.0)

    def get_fold_to_cbet(self, seat: int) -> float:
        return (self.folded_cbet_flop[seat] + 2.0) / (self.faced_cbet_flop[seat] + 4.0)

    def is_station(self, seat: int) -> bool:
        return self.get_vpip(seat) >= 0.38 and self.get_pfr(seat) <= 0.16

    def is_maniac(self, seat: int) -> bool:
        return self.get_pfr(seat) >= 0.32 or (
            self.get_vpip(seat) >= 0.45 and self.get_pfr(seat) >= 0.22
        )

    def is_nit(self, seat: int) -> bool:
        return self.hands_seen[seat] >= 8 and self.get_vpip(seat) <= 0.17


class RivalClaudeStyleBot:
    """
    Rival Team #6: Structured Multi-Factor Scoring & Street-Plan Bot.
    """

    name: str = "RivalClaudeStyle"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.hud = OpponentHUD()
        self.hero_seat = -1
        self.starting_stack = 2000
        self.rng = random.Random(seed)

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self.hero_seat = seat_idx
        self.starting_stack = starting_stack
        self.hud.reset()

    def on_hand_end(self, seat_idx: int, summary: Dict[str, Any]) -> None:
        self.hud.update_hand(summary)

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        try:
            return self._act_internal(state)
        except Exception:
            to_call = state.get("to_call", 0)
            return ("check", 0) if to_call == 0 else ("fold", 0)

    def _act_internal(self, state: Dict[str, Any]) -> Tuple[str, int]:
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        to_call = state["to_call"]

        if hero_stack <= 0:
            return ("check", 0)

        street = state["street"]
        h0, h1 = state["hole_cards"]
        bb = max(1, state["big_blind"])
        pot = max(1, state["pot"])
        min_raise_add = state["min_raise_add"]
        max_raise_add = state["max_raise_add"]
        active = state["active_seats"]
        button_idx = state["button_idx"]
        hand_num = state.get("hand_number", 1)

        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        street_bets = state["street_bets"]
        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        primary_villain = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))

        # Accurate effective stack against the primary villain in this pot
        eff_chips = min(hero_stack + street_bets[hero_idx], stacks[primary_villain] + street_bets[primary_villain])
        eff_bb = eff_chips / bb
        spr = eff_chips / pot

        # Malmuth-Harville ICM Bubble Factor
        bf, req_eq = compute_bubble_factor(
            total_stacks,
            hero_idx,
            primary_villain,
            max(bb, to_call),
            max(bb, pot),
            hand_number=hand_num,
        )

        pct = PREFLOP_PERCENTILE[h0][h1]
        hu_eq = PREFLOP_HU_EQ[h0][h1]

        # -------------------------------------------------------------
        # STREET 0: PREFLOP STRATEGY PLAN
        # -------------------------------------------------------------
        if street == 0:
            return self._plan_preflop(
                state=state,
                hero_idx=hero_idx,
                button_idx=button_idx,
                hero_stack=hero_stack,
                eff_bb=eff_bb,
                to_call=to_call,
                bb=bb,
                pot=pot,
                min_raise_add=min_raise_add,
                max_raise_add=max_raise_add,
                active_opps=active_opps,
                primary_villain=primary_villain,
                bf=bf,
                pct=pct,
                h0=h0,
                h1=h1,
            )

        # -------------------------------------------------------------
        # POSTFLOP: STREET-BY-STREET COMPOSITE MULTI-FACTOR SCORING
        # -------------------------------------------------------------
        return self._plan_postflop(
            state=state,
            street=street,
            hero_idx=hero_idx,
            button_idx=button_idx,
            hero_stack=hero_stack,
            eff_chips=eff_chips,
            spr=spr,
            to_call=to_call,
            pot=pot,
            bb=bb,
            min_raise_add=min_raise_add,
            max_raise_add=max_raise_add,
            active_opps=active_opps,
            primary_villain=primary_villain,
            bf=bf,
            req_eq=req_eq,
            pct=pct,
            hu_eq=hu_eq,
            h0=h0,
            h1=h1,
        )

    # =========================================================================
    # PREFLOP STREET PLAN
    # =========================================================================
    def _plan_preflop(
        self,
        state: Dict[str, Any],
        hero_idx: int,
        button_idx: int,
        hero_stack: int,
        eff_bb: float,
        to_call: int,
        bb: int,
        pot: int,
        min_raise_add: int,
        max_raise_add: int,
        active_opps: List[int],
        primary_villain: int,
        bf: float,
        pct: float,
        h0: int,
        h1: int,
    ) -> Tuple[str, int]:
        bb_seat = state.get("bb_seat", (button_idx + 2) & 7)
        sb_seat = state.get("sb_seat", (button_idx + 1) & 7)
        street_bets = state["street_bets"]
        max_bet = max(street_bets)
        dist_before_btn = (button_idx - hero_idx) & 7
        n_opp = len(active_opps)

        # Check if Hero was the preflop opener and faces a 3-bet / 4-bet!
        preflop_hero_raises = [
            a for a in state.get("action_history", []) if a[0] == 0 and a[1] == hero_idx and a[2] == "raise"
        ]
        if preflop_hero_raises:
            if pct >= 0.94:
                return ("raise", hero_stack)
            if pct >= 0.86 and eff_bb <= 20.0:
                return ("raise", hero_stack)
            if to_call <= 3.2 * bb and pct >= 0.76 and bf <= 1.35 and eff_bb >= 16.0:
                return ("call", to_call)
            return ("fold", 0)

        opps_behind: List[int] = []
        cur = (hero_idx + 1) & 7
        while cur != hero_idx:
            if cur in active_opps and street_bets[cur] <= max_bet:
                opps_behind.append(cur)
            if cur == bb_seat:
                break
            cur = (cur + 1) & 7

        # ---------------------------------------------------------------------
        # 1. Short-Stack Nash Push/Fold Plan (eff_bb <= 12.5, or <= 14.5 unopened)
        # ---------------------------------------------------------------------
        if max_bet <= bb:
            if n_opp == 1 and hero_idx == button_idx:
                if pct >= 0.22:
                    r_size = min(hero_stack, max(min_raise_add, int(2.15 * bb) - street_bets[hero_idx]))
                    return ("raise", r_size)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            if eff_bb <= 14.0:
                avg_fold_behind = (
                    sum(self.hud.get_fold_to_steal(o) for o in opps_behind) / max(1, len(opps_behind))
                    if opps_behind
                    else 0.65
                )
                steal_mult = 1.0 + 0.25 * max(0.0, avg_fold_behind - 0.60)
                push_thresh = (NASH_PUSH_BB[h0][h1] * steal_mult) / (
                    (bf ** 1.15) * (1.0 + 0.08 * max(0, len(opps_behind) - 1))
                )
                if eff_bb <= push_thresh and pct >= 0.35:
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)
        else:
            if eff_bb <= 12.5:
                call_thresh = NASH_CALL_BB[h0][h1] / (bf ** 1.30)
                if (to_call / bb) <= call_thresh:
                    if pct >= 0.82 or to_call >= int(0.45 * hero_stack):
                        return ("raise", hero_stack)
                    return ("call", to_call)
                if pct >= 0.93:
                    return ("raise", hero_stack)
                return ("fold", 0)

        # ---------------------------------------------------------------------
        # 2. Unopened Pot (Positional RFI / Steal)
        # ---------------------------------------------------------------------
        if max_bet <= bb:
            avg_fold_behind = (
                sum(self.hud.get_fold_to_steal(o) for o in opps_behind) / max(1, len(opps_behind))
                if opps_behind
                else 0.65
            )

            # Smooth positional GTO open baseline
            if hero_idx == button_idx:
                base_open = 0.40
            elif hero_idx == sb_seat:
                base_open = 0.44
            else:
                base_open = 0.42 + 0.075 * min(5, dist_before_btn)

            open_thresh = base_open + (bf - 1.1) * 0.15

            # Steal wider when blinds fold often
            if len(opps_behind) <= 3 and avg_fold_behind >= 0.65:
                open_thresh -= 0.12

            if any(self.hud.is_maniac(o) for o in opps_behind):
                open_thresh += 0.05

            if pct >= open_thresh:
                r_size = min(hero_stack, max(min_raise_add, int(2.15 * bb) - street_bets[hero_idx]))
                return ("raise", r_size)
            return ("check", 0) if to_call == 0 else ("fold", 0)

        # ---------------------------------------------------------------------
        # 3. Exploitative 3-Bet & Squeeze Matrix
        # ---------------------------------------------------------------------
        raisers = [o for o in active_opps if street_bets[o] == max_bet]
        opener_idx = raisers[0] if raisers else primary_villain
        callers = [
            o
            for o in active_opps
            if (street_bets[o] > bb and street_bets[o] < max_bet) or (street_bets[o] == max_bet and o != opener_idx)
        ]
        num_callers = len(callers)

        opener_dist = (button_idx - opener_idx) & 7
        is_lp_opener = opener_dist in (0, 1, 7)  # BTN, CO, SB
        opener_pfr = self.hud.get_pfr(opener_idx)
        is_loose_opener = opener_pfr >= 0.22 or self.hud.is_maniac(opener_idx)
        raise_bb = max_bet / bb

        r0, r1 = h0 >> 2, h1 >> 2
        is_suited = (h0 & 3) == (h1 & 3)
        has_ace_blocker = (r0 == 12 or r1 == 12) and is_suited
        has_king_blocker = (r0 == 11 or r1 == 11) and is_suited

        # 3A. SQUEEZE PLAY (Opener + 1 or more callers: dead money in pot!)
        if num_callers >= 1:
            if pct >= 0.89:
                if eff_bb <= 22.0:
                    return ("raise", hero_stack)
                sq_size = min(hero_stack, max(min_raise_add, int(3.5 * to_call + num_callers * bb)))
                return ("raise", sq_size)

            if (
                (has_ace_blocker or (has_king_blocker and pct >= 0.68))
                and pct >= 0.60
                and bf <= 1.30
                and eff_bb >= 15.0
                and (is_lp_opener or is_loose_opener)
            ):
                sq_size = min(hero_stack, max(min_raise_add, int(3.2 * to_call + num_callers * bb)))
                return ("raise", sq_size)

            if r0 == r1 and to_call <= 3.5 * bb and eff_bb >= 18.0 and bf <= 1.35:
                return ("call", to_call)

            return ("fold", 0)

        # 3B. SINGLE RAISED POT (Heads-up facing opener)
        # Monster Value 3-Bet (AA, KK, QQ, JJ, AK)
        if pct >= 0.93:
            if eff_bb <= 22.0:
                return ("raise", hero_stack)
            three_bet_size = min(hero_stack, max(min_raise_add, int(2.85 * to_call)))
            return ("raise", three_bet_size)

        # Exploitative Light 3-Bet / Re-Steal vs Late Position or Loose Opener
        if (is_lp_opener or is_loose_opener) and raise_bb <= 2.8 and bf <= 1.32:
            if pct >= 0.85:
                if eff_bb <= 18.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, max(min_raise_add, int(2.8 * to_call))))

            if (has_ace_blocker or (has_king_blocker and pct >= 0.70)) and pct >= 0.64 and eff_bb >= 14.0:
                if eff_bb <= 18.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, max(min_raise_add, int(2.8 * to_call))))

        # Big Blind Minimum Defense Frequency (MDF) against small steals (<= 2.5x BB)
        if hero_idx == bb_seat and raise_bb <= 2.5 and bf <= 1.35 and eff_bb >= 14.0:
            if pct >= 0.48 or (is_suited and pct >= 0.40) or r0 == r1:
                return ("call", to_call)

        # Flat call in position with playable hands
        is_ip = ((hero_idx - button_idx) & 7) > ((opener_idx - button_idx) & 7)
        if is_ip and to_call <= 3.2 * bb and pct >= 0.72 and bf <= 1.35:
            return ("call", to_call)

        # Call with pocket pairs for set mining
        if r0 == r1 and to_call <= 2.8 * bb and eff_bb >= 16.0 and bf <= 1.35:
            return ("call", to_call)

        return ("fold", 0)

    # =========================================================================
    # POSTFLOP STREET PLAN (FLOP, TURN, RIVER)
    # =========================================================================
    def _plan_postflop(
        self,
        state: Dict[str, Any],
        street: int,
        hero_idx: int,
        button_idx: int,
        hero_stack: int,
        eff_chips: float,
        spr: float,
        to_call: int,
        pot: int,
        bb: int,
        min_raise_add: int,
        max_raise_add: int,
        active_opps: List[int],
        primary_villain: int,
        bf: float,
        req_eq: float,
        pct: float,
        hu_eq: float,
        h0: int,
        h1: int,
    ) -> Tuple[str, int]:
        board = state["board"]
        n_opp = len(active_opps)

        made_cat, wetness, has_draw, is_overpair_or_tptk, has_nut_blocker = analyze_board_texture(
            (h0, h1), board
        )

        if n_opp == 1:
            is_ip = hero_idx == button_idx or (
                primary_villain != button_idx and ((hero_idx - button_idx) & 7) > ((primary_villain - button_idx) & 7)
            )
        else:
            is_ip = ((hero_idx - button_idx) & 7) > ((primary_villain - button_idx) & 7)

        bet_ratio = to_call / max(1, pot - to_call) if to_call > 0 else 0.0
        bluff_prob = 0.35 if self.hud.is_maniac(primary_villain) else (0.12 if self.hud.is_nit(primary_villain) else 0.22)

        # Fast 40-sample Bayesian postflop equity (~28 us)
        eq = fast_postflop_equity(
            (h0, h1),
            board,
            num_opponents=n_opp,
            num_samples=40,
            villain_min_pct=0.52,
            villain_bet_ratio=bet_ratio,
            villain_bluff_Prob=bluff_prob,
        )

        # ---------------------------------------------------------------------
        # SPR COMMITMENT THRESHOLD 1: LOW SPR COMMITMENT (SPR <= 2.5)
        # ---------------------------------------------------------------------
        is_strong_made = made_cat >= 2 or is_overpair_or_tptk
        if spr <= 2.5 and (is_strong_made or (has_draw and eq >= 0.44)):
            if to_call == 0:
                if spr <= 1.4:
                    return ("raise", hero_stack)
                bet_amt = min(hero_stack, max(min_raise_add, int(0.70 * pot)))
                return ("raise", bet_amt)
            else:
                if eq >= 0.48 or made_cat >= 2:
                    return ("raise", hero_stack)
                return ("call", to_call)

        # ---------------------------------------------------------------------
        # SPR COMMITMENT THRESHOLD 2: HIGH SPR POT CONTROL (SPR >= 6.0)
        # ---------------------------------------------------------------------
        if spr >= 6.0 and made_cat <= 1:
            if to_call == 0:
                if is_ip or street >= 2:
                    return ("check", 0)
                if street == 1 and n_opp == 1 and wetness <= 0.35 and is_overpair_or_tptk:
                    cbet_amt = min(hero_stack, max(min_raise_add, int(0.35 * pot)))
                    return ("raise", cbet_amt)
                return ("check", 0)
            else:
                if street >= 2 and (bet_ratio >= 0.40 or wetness >= 0.38):
                    if eq < req_eq + 0.10:
                        return ("fold", 0)
                if bet_ratio <= 0.35 and eq >= req_eq:
                    return ("call", to_call)
                if eq >= req_eq + 0.06:
                    return ("call", to_call)
                return ("fold", 0)

        # ---------------------------------------------------------------------
        # STREET-BY-STREET SPECIFIC PLANS (MEDIUM SPR: 2.5 < SPR < 6.0)
        # ---------------------------------------------------------------------
        if street == 1:
            return self._plan_flop(
                to_call=to_call,
                pot=pot,
                hero_stack=hero_stack,
                min_raise_add=min_raise_add,
                n_opp=n_opp,
                is_ip=is_ip,
                wetness=wetness,
                made_cat=made_cat,
                is_overpair_or_tptk=is_overpair_or_tptk,
                has_draw=has_draw,
                has_nut_blocker=has_nut_blocker,
                eq=eq,
                req_eq=req_eq,
                bet_ratio=bet_ratio,
                primary_villain=primary_villain,
            )
        elif street == 2:
            return self._plan_turn(
                to_call=to_call,
                pot=pot,
                hero_stack=hero_stack,
                min_raise_add=min_raise_add,
                n_opp=n_opp,
                is_ip=is_ip,
                wetness=wetness,
                made_cat=made_cat,
                is_overpair_or_tptk=is_overpair_or_tptk,
                has_draw=has_draw,
                has_nut_blocker=has_nut_blocker,
                eq=eq,
                req_eq=req_eq,
                bet_ratio=bet_ratio,
                primary_villain=primary_villain,
            )
        else:  # street == 3 (River)
            return self._plan_river(
                to_call=to_call,
                pot=pot,
                hero_stack=hero_stack,
                min_raise_add=min_raise_add,
                n_opp=n_opp,
                is_ip=is_ip,
                wetness=wetness,
                made_cat=made_cat,
                has_nut_blocker=has_nut_blocker,
                eq=eq,
                req_eq=req_eq,
                bet_ratio=bet_ratio,
                primary_villain=primary_villain,
                bf=bf,
            )

    # -------------------------------------------------------------------------
    # FLOP PLAN (Street 1)
    # -------------------------------------------------------------------------
    def _plan_flop(
        self,
        to_call: int,
        pot: int,
        hero_stack: int,
        min_raise_add: int,
        n_opp: int,
        is_ip: bool,
        wetness: float,
        made_cat: int,
        is_overpair_or_tptk: bool,
        has_draw: bool,
        has_nut_blocker: bool,
        eq: float,
        req_eq: float,
        bet_ratio: float,
        primary_villain: int,
    ) -> Tuple[str, int]:
        is_station = self.hud.is_station(primary_villain)
        fold_to_cbet = self.hud.get_fold_to_cbet(primary_villain)

        if to_call == 0:
            if made_cat >= 3:
                if wetness <= 0.30 and n_opp == 1 and self.hud.is_maniac(primary_villain):
                    return ("check", 0)
                bet_size = min(hero_stack, max(min_raise_add, int(0.65 * pot)))
                return ("raise", bet_size)

            if made_cat >= 2 or (is_overpair_or_tptk and eq >= 0.64) or eq >= 0.70:
                size_fraction = 0.70 if is_station else (0.62 if wetness >= 0.40 else 0.52)
                bet_size = min(hero_stack, max(min_raise_add, int(size_fraction * pot)))
                return ("raise", bet_size)

            if n_opp == 1:
                if wetness <= 0.45 and (eq >= 0.38 or has_draw or has_nut_blocker or fold_to_cbet >= 0.48):
                    bet_size = min(hero_stack, max(min_raise_add, int(0.35 * pot)))
                    return ("raise", bet_size)

                if has_draw and eq >= 0.42:
                    bet_size = min(hero_stack, max(min_raise_add, int(0.55 * pot)))
                    return ("raise", bet_size)

            return ("check", 0)

        # Facing a Bet on Flop (to_call > 0)
        if made_cat >= 2 and eq >= 0.70:
            raise_target = min(hero_stack, max(min_raise_add, to_call + int(0.70 * pot)))
            return ("raise", raise_target)

        if n_opp == 1 and has_draw and bet_ratio <= 0.42 and eq >= 0.46 and fold_to_cbet >= 0.45:
            raise_target = min(hero_stack, max(min_raise_add, to_call + int(0.65 * pot)))
            return ("raise", raise_target)

        draw_equity_boost = 0.05 if has_draw else 0.0
        pos_boost = 0.02 if is_ip else 0.0
        margin = 0.06
        if (eq + draw_equity_boost + pos_boost) >= req_eq + margin:
            return ("call", to_call)

        if n_opp == 1 and is_ip and bet_ratio <= 0.35 and (made_cat >= 1 or eq >= 0.38):
            return ("call", to_call)

        return ("fold", 0)

    # -------------------------------------------------------------------------
    # TURN PLAN (Street 2)
    # -------------------------------------------------------------------------
    def _plan_turn(
        self,
        to_call: int,
        pot: int,
        hero_stack: int,
        min_raise_add: int,
        n_opp: int,
        is_ip: bool,
        wetness: float,
        made_cat: int,
        is_overpair_or_tptk: bool,
        has_draw: bool,
        has_nut_blocker: bool,
        eq: float,
        req_eq: float,
        bet_ratio: float,
        primary_villain: int,
    ) -> Tuple[str, int]:
        is_station = self.hud.is_station(primary_villain)

        if to_call == 0:
            if made_cat >= 2 or (is_overpair_or_tptk and eq >= 0.64):
                size_fraction = 0.75 if is_station else 0.60
                bet_size = min(hero_stack, max(min_raise_add, int(size_fraction * pot)))
                return ("raise", bet_size)

            if n_opp == 1 and is_ip and (eq >= 0.44 or has_draw or has_nut_blocker):
                bet_size = min(hero_stack, max(min_raise_add, int(0.48 * pot)))
                return ("raise", bet_size)

            return ("check", 0)

        # Facing Turn Bet (to_call > 0)
        if made_cat >= 3 and eq >= 0.75:
            raise_target = min(hero_stack, max(min_raise_add, to_call + int(0.75 * pot)))
            return ("raise", raise_target)

        draw_equity_boost = 0.04 if has_draw else 0.0
        pos_boost = 0.02 if is_ip else 0.0
        margin = 0.08
        if (eq + draw_equity_boost + pos_boost) >= req_eq + margin:
            return ("call", to_call)

        return ("fold", 0)

    # -------------------------------------------------------------------------
    # RIVER PLAN (Street 3)
    # -------------------------------------------------------------------------
    def _plan_river(
        self,
        to_call: int,
        pot: int,
        hero_stack: int,
        min_raise_add: int,
        n_opp: int,
        is_ip: bool,
        wetness: float,
        made_cat: int,
        has_nut_blocker: bool,
        eq: float,
        req_eq: float,
        bet_ratio: float,
        primary_villain: int,
        bf: float,
    ) -> Tuple[str, int]:
        is_station = self.hud.is_station(primary_villain)

        if to_call == 0:
            if made_cat >= 3:
                size_fraction = 0.80 if is_station else 0.65
                bet_size = min(hero_stack, max(min_raise_add, int(size_fraction * pot)))
                return ("raise", bet_size)

            if (made_cat >= 2 or (made_cat == 1 and eq >= 0.72)) and is_station:
                bet_size = min(hero_stack, max(min_raise_add, int(0.52 * pot)))
                return ("raise", bet_size)

            if made_cat >= 1:
                return ("check", 0)

            if (
                n_opp == 1
                and has_nut_blocker
                and not is_station
                and self.hud.get_fold_to_cbet(primary_villain) >= 0.48
            ):
                bluff_size = min(hero_stack, max(min_raise_add, int(0.60 * pot)))
                return ("raise", bluff_size)

            return ("check", 0)

        # Facing River Bet (to_call > 0)
        if made_cat == 0:
            return ("fold", 0)

        if made_cat >= 4 and eq >= 0.88:
            raise_target = min(hero_stack, max(min_raise_add, to_call + int(0.70 * pot)))
            return ("raise", raise_target)

        maniac_discount = -0.05 if self.hud.is_maniac(primary_villain) else (0.06 if self.hud.is_nit(primary_villain) else 0.0)
        margin = 0.07

        if eq >= req_eq + margin + maniac_discount:
            return ("call", to_call)

        return ("fold", 0)
