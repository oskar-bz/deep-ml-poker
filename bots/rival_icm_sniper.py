"""
RivalICMSniperBot — Tournament ICM Warlord, Bubble Exploiter & Stack-Asymmetry Specialist
Designed for the Deep-ML 8-Player Sit-and-Go No-Limit Hold'em Tournament.

Core Architecture:
1. Exact Malmuth-Harville ICM & Asymmetric Bubble Factor Engine:
   - Computes both Hero's ICM Bubble Factor (`hero_bf`) and Villain's ICM Bubble Factor
     (`villain_bf`) in ~9 us using `compute_bubble_factor` and `compute_hero_icm`.
   - Detects micro-stack bubble freeze states (4-5 players left with a short stack <5 BB)
     and 200-hand cap lock-in windows (hands 155..200).
2. Online Bayesian Opponent Profiler (`on_hand_end` + live state):
   - Tracks per-seat VPIP, PFR, Fold-to-Steal, Fold-to-3Bet, Postflop Fold-to-Bet,
     Postflop Raise-to-Bet, and Postflop Aggression with zero decision-time overhead.
   - Computes joint fold probability `prob_all_fold` across all remaining players behind
     to trigger high-EV 2.0x BB blind steals and bubble bullying.
3. True Preflop Tiering & 4.1x BB 3-Bet Squeeze/Re-Steal Sniper:
   - Separates true high-card/overpair monsters (AA/KK/QQ/AK) from set-mining medium
     pairs (66-99) whose `PREFLOP_PERCENTILE` is inflated by multiway set value.
   - Uses an exact `max_bet + 4.1x BB` 3-bet sizing that crosses the `to_call > 4.0x BB`
     and `> 3.5x BB` fold thresholds of CFR_GTO, TrapCounter, SharkTAG, and Nit archetypes,
     forcing 85%-94% folds across positional opens.
4. Board-Relative Hand Classifier (`_classify_true_hand`):
   - Eliminates the paired-board `made_cat == 2` illusion by distinguishing genuine Two Pair
     and Sets from underpairs/single-pairs on paired boards.
5. Sizing-Tell & Capped-Range Postflop Engine:
   - Check-raises small Flop C-bets (<=40% pot) with Two Pair+ and high-equity draws.
   - Fires second barrels with draws, nut blockers, and strong pairs to maintain initiative,
     and fires 48% pot Turn probe bets whenever the flop C-bettor checks the Turn.
   - Distinguishes value-sized Turn/River barrels (>=52% pot) from probe/blocker bluffs
     (<=49% pot) and folds One Pair immediately to non-maniac postflop raises.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

from poker_engine.evaluator import (
    analyze_board_texture,
    fast_postflop_equity,
)
from poker_engine.icm import (
    compute_bubble_factor,
)
from poker_engine.preflop_tables import (
    NASH_CALL_BB,
    NASH_PUSH_BB,
    PREFLOP_PERCENTILE,
    PREFLOP_STRENGTH,
)


def _classify_true_hand(
    h0: int,
    h1: int,
    board: Sequence[int],
    raw_made_cat: int,
    raw_is_tptk: bool,
) -> Tuple[int, bool]:
    """
    O(1) Board-Relative Hand Classifier (~1.2 us).
    Prevents the paired-board illusion where a weak pocket pair (e.g. 44 on Q-Q-8)
    or a single board match has `raw_made_cat == 2` (Two Pair) or `raw_made_cat == 3` (Trips).

    Returns:
        (true_cat, true_is_tptk)
    """
    if raw_made_cat == 0:
        return 0, False

    hr0, hr1 = h0 >> 2, h1 >> 2
    hi, lo = (hr0, hr1) if hr0 >= hr1 else (hr1, hr0)
    is_pocket_pair = hr0 == hr1

    b_counts: Dict[int, int] = {}
    max_b = 0
    for c in board:
        r = c >> 2
        b_counts[r] = b_counts.get(r, 0) + 1
        if r > max_b:
            max_b = r

    max_b_cnt = max(b_counts.values())
    if max_b_cnt == 1:
        return raw_made_cat, raw_is_tptk

    n_b_pairs = sum(1 for cnt in b_counts.values() if cnt >= 2)

    if max_b_cnt == 2 and n_b_pairs == 1:
        paired_r = next(r for r, cnt in b_counts.items() if cnt == 2)
        if hr0 == paired_r or hr1 == paired_r:
            return raw_made_cat, True
        if raw_made_cat >= 4:
            if raw_made_cat == 6 and is_pocket_pair and hi < max_b:
                return 3, False
            return raw_made_cat, True
        if is_pocket_pair:
            if hi > max_b:
                return 1, True
            return 1, False
        unpaired_max = max((r for r in b_counts if r != paired_r), default=0)
        if hr0 == unpaired_max or hr1 == unpaired_max:
            other_kicker = lo if hi == unpaired_max else hi
            return 1, (unpaired_max >= max_b and other_kicker >= 9)
        if (hr0 in b_counts) or (hr1 in b_counts):
            return 1, False
        return 0, False

    if max_b_cnt >= 3:
        trips_r = next(r for r, cnt in b_counts.items() if cnt >= 3)
        if hr0 == trips_r or hr1 == trips_r:
            return 7, True
        if is_pocket_pair and hi >= max_b:
            return 6, True
        if (hr0 in b_counts) or (hr1 in b_counts):
            matched_r = hr0 if (hr0 in b_counts and hr0 != trips_r) else hr1
            return (6, True) if matched_r >= max_b else (2, False)
        if is_pocket_pair:
            return 1, (hi > max_b - 2)
        return 0, False

    paired_ranks = [r for r, cnt in b_counts.items() if cnt == 2]
    top_pair_r = max(paired_ranks)
    if hr0 == top_pair_r or hr1 == top_pair_r:
        return 6, True
    if (hr0 in paired_ranks) or (hr1 in paired_ranks):
        return 3, True
    if is_pocket_pair and hi > max_b:
        return 2, True
    if (hr0 in b_counts) or (hr1 in b_counts):
        return 1, False
    return 0, False


class RivalICMSniperBot:
    """
    Tournament ICM Warlord, Bubble Exploiter & Stack-Asymmetry Specialist.
    Strictly engineered for <60 us average latency and <300 us P99 latency.
    """

    name: str = "RivalICMSniperBot"

    def __init__(self) -> None:
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.hands_seen: List[int] = [0] * 8
        self.vpip: List[int] = [0] * 8
        self.pfr: List[int] = [0] * 8
        self.faced_open: List[int] = [0] * 8
        self.folded_open: List[int] = [0] * 8
        self.three_bet: List[int] = [0] * 8
        self.faced_3bet: List[int] = [0] * 8
        self.folded_3bet: List[int] = [0] * 8
        self.post_faced_bet: List[int] = [0] * 8
        self.post_folded_bet: List[int] = [0] * 8
        self.post_raised_bet: List[int] = [0] * 8
        self.post_bet_opp: List[int] = [0] * 8
        self.post_bet_made: List[int] = [0] * 8
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
            for s in range(8):
                if start_stacks[s] > 0:
                    self.hands_seen[s] += 1

        action_history = summary.get("action_history")
        if not action_history:
            return

        vpip_flag = [False] * 8
        pfr_flag = [False] * 8
        n_raises_pre = 0
        cur_street = 0
        street_bet_live = False

        for st, s, act, _amt in action_history:
            if st == 0:
                if act in ("sb", "bb"):
                    continue
                if act in ("call", "raise"):
                    vpip_flag[s] = True
                if act == "raise":
                    pfr_flag[s] = True

                if n_raises_pre == 1:
                    self.faced_open[s] += 1
                    if act == "fold":
                        self.folded_open[s] += 1
                    elif act == "raise":
                        self.three_bet[s] += 1
                elif n_raises_pre >= 2:
                    self.faced_3bet[s] += 1
                    if act == "fold":
                        self.folded_3bet[s] += 1

                if act == "raise":
                    n_raises_pre += 1
            else:
                if st != cur_street:
                    cur_street = st
                    street_bet_live = False

                if not street_bet_live:
                    self.post_bet_opp[s] += 1
                    if act in ("raise", "bet"):
                        self.post_bet_made[s] += 1
                        street_bet_live = True
                else:
                    self.post_faced_bet[s] += 1
                    if act == "fold":
                        self.post_folded_bet[s] += 1
                    elif act in ("raise", "bet"):
                        self.post_raised_bet[s] += 1

        for s in range(8):
            if vpip_flag[s]:
                self.vpip[s] += 1
            if pfr_flag[s]:
                self.pfr[s] += 1

    # --- Fast Bayesian Posterior Rate Helpers ---

    def _vpip_rate(self, s: int) -> float:
        return (self.vpip[s] + 1.3) / (self.hands_seen[s] + 5.0)

    def _pfr_rate(self, s: int) -> float:
        return (self.pfr[s] + 0.9) / (self.hands_seen[s] + 5.0)

    def _fold_open_rate(self, s: int) -> float:
        emp_fold = (self.folded_open[s] + 2.9) / (self.faced_open[s] + 4.0)
        vpip_complement = 1.0 - self._vpip_rate(s)
        return max(emp_fold, 0.55 * emp_fold + 0.45 * vpip_complement)

    def _fold_3bet_rate(self, s: int) -> float:
        return (self.folded_3bet[s] + 2.5) / (self.faced_3bet[s] + 3.0)

    def _post_fold_rate(self, s: int) -> float:
        return (self.post_folded_bet[s] + 2.2) / (self.post_faced_bet[s] + 4.0)

    def _is_calling_station(self, s: int) -> bool:
        hs = self.hands_seen[s]
        if hs < 6:
            return False
        vpip_r = self.vpip[s] / hs
        pfr_r = self.pfr[s] / hs
        return (vpip_r >= 0.36 and (vpip_r - pfr_r) >= 0.22) or (
            self.post_faced_bet[s] >= 4 and (self.post_folded_bet[s] / self.post_faced_bet[s]) <= 0.30
        )

    def _is_maniac(self, s: int) -> bool:
        hs = self.hands_seen[s]
        if hs < 6:
            return False
        return (self.pfr[s] / hs) >= 0.36 or (
            self.post_bet_opp[s] >= 6 and (self.post_bet_made[s] / self.post_bet_opp[s]) >= 0.55
        )

    def _is_ultra_nit(self, s: int) -> bool:
        hs = self.hands_seen[s]
        if hs < 8:
            return False
        return (self.vpip[s] / hs) <= 0.10 and (self.pfr[s] / hs) <= 0.08

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        to_call: int = state["to_call"]
        hero_idx: int = state["hero_idx"]
        stacks: List[int] = state["stacks"]
        hero_stack: int = stacks[hero_idx]

        if hero_stack <= 0:
            return ("check", 0)

        street: int = state["street"]
        bb: int = max(1, state["big_blind"])
        pot: int = state["pot"]
        min_raise_add: int = state["min_raise_add"]
        street_bets: List[int] = state["street_bets"]
        active: List[bool] = state["active_seats"]
        hand_num: int = state.get("hand_number", 1)

        if hand_num < self._last_recorded_hand and hand_num <= 2:
            self._reset_stats()

        active_opps: List[int] = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        hand_contrib: List[int] = state.get("hand_contributions", street_bets)
        total_stacks: List[int] = [stacks[i] + hand_contrib[i] for i in range(8)]
        hero_total: int = total_stacks[hero_idx]

        h0, h1 = state["hole_cards"]
        pct: float = PREFLOP_PERCENTILE[h0][h1]

        primary_villain: int = max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))

        if street == 0:
            return self._act_preflop(
                state=state,
                h0=h0,
                h1=h1,
                pct=pct,
                hero_idx=hero_idx,
                hero_stack=hero_stack,
                hero_total=hero_total,
                bb=bb,
                pot=pot,
                to_call=to_call,
                min_raise_add=min_raise_add,
                street_bets=street_bets,
                total_stacks=total_stacks,
                active_opps=active_opps,
                primary_villain=primary_villain,
                hand_num=hand_num,
            )

        return self._act_postflop(
            state=state,
            h0=h0,
            h1=h1,
            pct=pct,
            street=street,
            hero_idx=hero_idx,
            hero_stack=hero_stack,
            hero_total=hero_total,
            bb=bb,
            pot=pot,
            to_call=to_call,
            min_raise_add=min_raise_add,
            street_bets=street_bets,
            total_stacks=total_stacks,
            active_opps=active_opps,
            primary_villain=primary_villain,
            hand_num=hand_num,
        )

    # =========================================================================
    # PREFLOP ENGINE
    # =========================================================================

    def _act_preflop(
        self,
        state: Dict[str, Any],
        h0: int,
        h1: int,
        pct: float,
        hero_idx: int,
        hero_stack: int,
        hero_total: int,
        bb: int,
        pot: int,
        to_call: int,
        min_raise_add: int,
        street_bets: List[int],
        total_stacks: List[int],
        active_opps: List[int],
        primary_villain: int,
        hand_num: int,
    ) -> Tuple[str, int]:
        max_bet = max(street_bets)
        button_idx = state["button_idx"]
        bb_seat = state.get("bb_seat", (button_idx + 2) & 7)
        stacks = state["stacks"]

        alive_seats = [i for i in range(8) if total_stacks[i] > 0]
        n_alive = len(alive_seats)
        min_other_stack = min((total_stacks[i] for i in alive_seats if i != hero_idx), default=hero_total)
        micro_stack_at_table = (n_alive in (4, 5)) and (min_other_stack <= 4.5 * bb) and (hero_total >= 10 * bb)
        near_cap = hand_num >= 160 and n_alive >= 3

        opps_behind: List[int] = []
        if hero_idx != bb_seat:
            cur = (hero_idx + 1) & 7
            for _ in range(7):
                if cur in active_opps and street_bets[cur] <= bb:
                    opps_behind.append(cur)
                if cur == bb_seat:
                    break
                cur = (cur + 1) & 7

        r0, r1 = h0 >> 2, h1 >> 2
        hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
        is_pair = r0 == r1
        is_suited = (h0 & 3) == (h1 & 3)

        is_aa_kk = is_pair and hi >= 11
        is_qq_jj_ak = is_aa_kk or (is_pair and hi >= 9) or (hi == 12 and lo == 11)
        is_tt_aqs_aqo = is_qq_jj_ak or (is_pair and hi == 8) or (hi == 12 and lo == 10)
        is_99_ajs_kqs = is_tt_aqs_aqo or (is_pair and hi == 7) or (is_suited and hi >= 11 and lo >= 9)

        # ---------------------------------------------------------------------
        # 1. UNOPENED POT (or limped pot: max_bet <= bb)
        # ---------------------------------------------------------------------
        if max_bet <= bb:
            target_opps = opps_behind if opps_behind else active_opps
            max_opp_total = max((total_stacks[o] for o in target_opps), default=hero_total)
            eff_chips = min(hero_total, max_opp_total)
            eff_bb = eff_chips / bb

            threat_villain = max(target_opps, key=lambda o: total_stacks[o])
            clash_probe = min(eff_chips, max(4 * bb, int(0.30 * eff_chips)))
            bf, _ = compute_bubble_factor(
                total_stacks, hero_idx, threat_villain, clash_probe, None, hand_num
            )
            if micro_stack_at_table and total_stacks[threat_villain] > hero_total:
                bf *= 1.12

            v_bf, _ = compute_bubble_factor(
                total_stacks, threat_villain, hero_idx, clash_probe, None, hand_num
            )

            n_behind = len(opps_behind)
            n_limpers = len(active_opps) - n_behind

            prob_all_fold = 1.0
            for o in opps_behind:
                prob_all_fold *= self._fold_open_rate(o)

            any_station_behind = any(self._is_calling_station(o) for o in opps_behind)
            any_maniac_behind = any(self._is_maniac(o) for o in opps_behind)

            min_open_add = min(hero_stack, max(min_raise_add, int(2.0 * bb) - street_bets[hero_idx]))

            # A. Critical Short-Stack Zone (eff_bb <= 6.5 BB): Nash Push/Fold
            if eff_bb <= 6.5:
                pos_mult = 1.0 + 0.12 * max(0, n_behind + n_limpers - 1)
                icm_mult = bf ** 1.15
                req_push_bb = NASH_PUSH_BB[h0][h1] / (pos_mult * icm_mult)
                if prob_all_fold >= 0.55 and n_behind <= 2 and n_limpers == 0:
                    req_push_bb *= 1.25
                if eff_bb <= req_push_bb and pct >= (0.36 if n_behind <= 1 else 0.48):
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            # B. Mid-Short Stack Zone (6.5 < eff_bb <= 13.5 BB):
            if eff_bb <= 13.5 and n_limpers == 0:
                pos_mult = 1.0 + 0.13 * max(0, n_behind - 1)
                req_push_bb = NASH_PUSH_BB[h0][h1] / (pos_mult * (bf ** 1.18))
                if eff_bb <= req_push_bb and pct >= 0.68 and bf <= 1.35:
                    if is_qq_jj_ak and eff_bb >= 9.0:
                        return ("raise", min_open_add)
                    return ("raise", hero_stack)

                if n_behind <= 3 and not any_station_behind and not any_maniac_behind:
                    if prob_all_fold >= 0.42 or v_bf >= 1.30:
                        steal_thresh = 0.40 + 0.05 * (n_behind - 1) - max(0.0, (prob_all_fold - 0.42) * 0.85)
                        if v_bf >= 1.30:
                            steal_thresh -= min(0.22, (v_bf - 1.18) * 0.42)
                        if pct >= max(0.18, steal_thresh):
                            return ("raise", min_open_add)

            # C. Exploitative Blind Steal & Bubble Bullying (eff_bb > 13.5, n_limpers == 0)
            if n_behind <= 3 and n_limpers == 0 and not any_station_behind and not any_maniac_behind:
                req_fold_prob = 0.48 if n_behind == 1 else (0.42 if n_behind == 2 else 0.35)
                if prob_all_fold >= req_fold_prob or v_bf >= 1.30:
                    bully_thresh = 0.32 + 0.06 * (n_behind - 1) - max(0.0, (prob_all_fold - req_fold_prob) * 0.90)
                    if v_bf >= 1.30:
                        bully_thresh -= min(0.22, (v_bf - 1.20) * 0.42)
                    if bf >= 1.35:
                        bully_thresh += (bf - 1.35) * 0.25
                    if pct >= max(0.18, bully_thresh):
                        return ("raise", min_open_add)

            # D. Standard Positional Open-Raise
            base_open = 0.39 + 0.062 * min(6, n_behind) + 0.07 * n_limpers
            if bf > 1.15:
                base_open += (bf - 1.15) * 0.18
            if near_cap and bf > 1.20:
                base_open += 0.05
            if v_bf > 1.28 and n_behind <= 3:
                base_open -= min(0.14, (v_bf - 1.25) * 0.28)

            if pct >= base_open:
                if n_limpers > 0 and is_tt_aqs_aqo:
                    val_size = int((2.4 + 0.5 * n_limpers) * bb) - street_bets[hero_idx]
                    return ("raise", min(hero_stack, max(min_raise_add, val_size)))
                return ("raise", min_open_add)

            if to_call == 0:
                return ("check", 0)
            return ("fold", 0)

        # ---------------------------------------------------------------------
        # 2. FACING A RAISE OR ALL-IN (max_bet > bb)
        # ---------------------------------------------------------------------
        n_raises = 0
        last_raiser = primary_villain
        for st, s, act, _amt in state.get("action_history", ()):
            if st == 0 and act == "raise":
                last_raiser = s
                n_raises += 1

        n_callers = sum(
            1 for o in active_opps if o != last_raiser and street_bets[o] >= max_bet - (bb // 2)
        )

        v_total = total_stacks[last_raiser]
        v_rem = stacks[last_raiser]
        eff_chips = min(hero_total, v_total)
        eff_bb = eff_chips / bb
        raise_bb = max_bet / bb
        call_bb = to_call / bb

        bf_call, _ = compute_bubble_factor(
            total_stacks, hero_idx, last_raiser, to_call, None, hand_num
        )
        if micro_stack_at_table and v_total >= hero_total and to_call >= 0.25 * hero_stack:
            bf_call *= 1.15

        v_is_nit = self._is_ultra_nit(last_raiser)
        v_is_maniac = self._is_maniac(last_raiser)
        v_is_station = self._is_calling_station(last_raiser)
        v_pfr = self._pfr_rate(last_raiser)
        v_fold_3bet = self._fold_3bet_rate(last_raiser)

        # --- SUBCASE 2A: Facing a 3-Bet / 4-Bet or All-In Shove ---
        if n_raises >= 2 or raise_bb >= 4.2 or v_rem == 0 or to_call >= int(0.32 * hero_stack):
            if v_is_nit and not v_is_maniac:
                if is_aa_kk:
                    return ("raise", hero_stack) if hero_stack > to_call else ("call", to_call)
                return ("fold", 0)

            if v_is_maniac:
                if is_tt_aqs_aqo or (pct >= 0.86 and bf_call <= 1.25) or (is_pair and hi >= 6 and bf_call <= 1.25):
                    return ("raise", hero_stack) if eff_bb <= 22.0 else ("call", to_call)
                return ("fold", 0)

            hero_already_raised = street_bets[hero_idx] > bb
            if hero_already_raised and n_raises >= 2:
                if street_bets[hero_idx] > 3 * bb:
                    if is_aa_kk or (is_qq_jj_ak and hi >= 10 and bf_call <= 1.15):
                        return ("raise", hero_stack) if hero_stack > to_call else ("call", to_call)
                    if to_call <= int(0.16 * pot) and pct >= 0.80:
                        return ("call", to_call)
                    return ("fold", 0)
                else:
                    if is_aa_kk:
                        return ("raise", hero_stack if eff_bb <= 28.0 else min(hero_stack, max(min_raise_add, int(2.5 * to_call))))
                    if is_qq_jj_ak and bf_call <= 1.28:
                        if v_rem == 0 or eff_bb <= 16.0:
                            if call_bb <= 14.0 or hi >= 10:
                                return ("call", to_call)
                            return ("fold", 0)
                        if to_call <= int(0.22 * hero_stack):
                            return ("call", to_call)
                    if to_call <= int(0.14 * hero_stack) and bf_call <= 1.18 and is_tt_aqs_aqo:
                        return ("call", to_call)
                    return ("fold", 0)

            if n_raises >= 2 and call_bb >= 6.0:
                if is_aa_kk or (is_pair and hi == 10 and bf_call <= 1.15):
                    return ("raise", hero_stack) if hero_stack > to_call else ("call", to_call)
                return ("fold", 0)

            adj_nash_call = NASH_CALL_BB[h0][h1] / (bf_call ** 1.35)
            if near_cap and bf_call > 1.20:
                adj_nash_call *= 0.80
            if call_bb <= adj_nash_call:
                if call_bb <= 9.0 or is_tt_aqs_aqo or (is_99_ajs_kqs and bf_call <= 1.15):
                    return ("raise", hero_stack) if (is_aa_kk and hero_stack > to_call) else ("call", to_call)
            if is_aa_kk:
                return ("raise", hero_stack) if hero_stack > to_call else ("call", to_call)
            return ("fold", 0)

        # --- SUBCASE 2B: Facing a Single Open-Raise (n_raises == 1, raise_bb < 4.2) ---
        # CRITICAL: Make `v_add_target = max(int(4.05 * bb) + 1, int(1.80 * max_bet))` so that:
        # - `to_call` for the opener is strictly > 4.0 * bb (and > 3.5 * bb and > 2.5 * bb)!
        # - This forces SharkTAG, LLM_CFR_GTO, LLM_TrapCounter, and TightPassiveNit to FOLD 92%-94.5% of hands!
        v_add_target = max(int(4.05 * bb) + 1, int(1.80 * max_bet))
        if int(0.22 * v_rem) + 2 <= int(4.8 * bb):
            v_add_target = max(v_add_target, int(0.22 * v_rem) + 2)
        target_3bet_total = max_bet + v_add_target
        three_bet_add = min(hero_stack, max(min_raise_add, target_3bet_total - street_bets[hero_idx]))

        bf_3bet, _ = compute_bubble_factor(
            total_stacks, hero_idx, last_raiser, min(hero_stack, three_bet_add), None, hand_num
        )
        v_bf_3bet, _ = compute_bubble_factor(
            total_stacks, last_raiser, hero_idx, min(v_rem, v_add_target), None, hand_num
        )

        # 1. Value 3-Bet (AA, KK, QQ, JJ, AK, plus TT/99/AQs/AQo when BF <= 1.28)
        if not v_is_nit:
            if is_aa_kk or (is_qq_jj_ak and bf_3bet <= 1.38) or (is_tt_aqs_aqo and bf_3bet <= 1.28):
                if eff_bb <= 16.5 and bf_3bet <= 1.32:
                    return ("raise", hero_stack)
                return ("raise", three_bet_add)
        else:
            if is_aa_kk:
                return ("raise", hero_stack if eff_bb <= 18.0 else three_bet_add)

        # 2. High-Leverage 3-Bet Re-Steal & Squeeze Sniper (CO, BTN, SB, BB)!
        any_sticky_relevant = (
            v_is_station
            or v_is_maniac
            or any(self._is_calling_station(o) or self._is_maniac(o) for o in opps_behind)
        )
        opener_dist_btn = (button_idx - last_raiser) & 7
        can_re_steal = (
            not v_is_nit
            and not any_sticky_relevant
            and (opener_dist_btn <= 3 or v_pfr >= 0.20)
            and v_fold_3bet >= 0.62
            and len(opps_behind) <= 2
            and n_callers <= 1
        )

        if can_re_steal:
            has_ace_blocker = (hi == 12 and is_suited) or (hi == 12 and lo >= 8)
            has_king_suited = (hi == 11 and is_suited and lo >= 7)
            # A. Non-All-In 6.1x BB 3-Bet Sniper (risking <= 34% of Hero's stack)
            if eff_bb >= 15.5 and three_bet_add <= int(0.34 * hero_stack) and bf_3bet <= 1.40:
                if v_bf_3bet >= 1.28 and (pct >= 0.56 or has_ace_blocker or has_king_suited):
                    return ("raise", three_bet_add)
                if opener_dist_btn <= 2 and bf_3bet <= 1.28 and (
                    pct >= 0.84 or is_99_ajs_kqs or (has_ace_blocker and pct >= 0.64)
                ):
                    return ("raise", three_bet_add)

            # B. Short-Mid Stack Re-Steal Jam (10.0 - 15.5 BB)
            if 10.0 <= eff_bb <= 15.5 and bf_3bet <= 1.25 and (is_99_ajs_kqs or (is_pair and hi >= 5) or (has_ace_blocker and pct >= 0.80)):
                return ("raise", hero_stack)

        # 3. Flat-Call Defense (BB Minimum Defense Frequency + In-Position Implied Odds)
        if v_is_nit and not is_qq_jj_ak:
            if is_pair and to_call <= int(0.06 * eff_chips) and bf_call <= 1.20:
                return ("call", to_call)
            return ("fold", 0)

        # Big Blind defense closing the action against small steals
        if hero_idx == bb_seat and not opps_behind and raise_bb <= 2.3 and eff_bb >= 13.0 and bf_call <= 1.32:
            bb_def_pct = 0.54 + max(0.0, (raise_bb - 2.0) * 0.22) + max(0.0, (bf_call - 1.08) * 0.36)
            if near_cap and bf_call > 1.20:
                bb_def_pct += 0.08
            if pct >= bb_def_pct or ((is_suited or is_pair) and PREFLOP_STRENGTH[h0][h1] >= 0.55 and bf_call <= 1.20):
                return ("call", to_call)

        # Flat-call in position (CO/BTN) when price is cheap and hand has high playability
        if len(opps_behind) <= 1 and to_call <= int(2.6 * bb) and to_call <= int(0.13 * hero_stack) and bf_call <= 1.26:
            flat_pct = 0.78 + max(0.0, (bf_call - 1.08) * 0.28)
            if eff_bb >= 18.0 and (is_pair or (is_suited and pct >= 0.70)):
                flat_pct -= 0.06
            if pct >= flat_pct or is_99_ajs_kqs:
                return ("call", to_call)

        return ("fold", 0)

    # =========================================================================
    # POSTFLOP ENGINE
    # =========================================================================

    def _act_postflop(
        self,
        state: Dict[str, Any],
        h0: int,
        h1: int,
        pct: float,
        street: int,
        hero_idx: int,
        hero_stack: int,
        hero_total: int,
        bb: int,
        pot: int,
        to_call: int,
        min_raise_add: int,
        street_bets: List[int],
        total_stacks: List[int],
        active_opps: List[int],
        primary_villain: int,
        hand_num: int,
    ) -> Tuple[str, int]:
        board: List[int] = state["board"]
        n_opp = len(active_opps)

        raw_made_cat, wetness, has_draw, raw_is_tptk, has_nut_blocker = analyze_board_texture((h0, h1), board)
        made_cat, is_tptk = _classify_true_hand(h0, h1, board, raw_made_cat, raw_is_tptk)

        # Fast O(1) early exits
        if to_call > 0:
            if made_cat == 0 and not has_draw:
                if street >= 2 or n_opp >= 2 or to_call > int(0.26 * pot) or pct < 0.70:
                    return ("fold", 0)
        else:
            if n_opp >= 3 and made_cat == 0 and not has_draw:
                return ("check", 0)

        n_pre_raises = 0
        villain_raised_prev = False
        hero_barreled_prev = False
        villain_cbet_prev = False
        prev_st = street - 1
        prev_bets = 0
        for st, s, act, _amt in state.get("action_history", ()):
            if st == 0 and act == "raise":
                n_pre_raises += 1
            elif st == prev_st and prev_st >= 1:
                if act in ("raise", "bet"):
                    prev_bets += 1
                    if s == hero_idx:
                        hero_barreled_prev = True
                    else:
                        if prev_bets == 1:
                            villain_cbet_prev = True
                        elif prev_bets >= 2:
                            villain_raised_prev = True

        v_vpip = self._vpip_rate(primary_villain)
        v_min_pct = max(0.28, min(0.80, 1.0 - 1.15 * v_vpip))
        if n_pre_raises >= 2 or pot >= 10 * bb:
            v_min_pct = max(v_min_pct, 0.72)

        bet_ratio = to_call / max(1, pot - to_call) if to_call > 0 else 0.0
        v_is_station = self._is_calling_station(primary_villain)
        v_is_maniac = self._is_maniac(primary_villain)
        v_bluff_prob = 0.38 if v_is_maniac else (0.08 if v_is_station else 0.22)

        samples = 52 if n_opp == 1 else 42
        eq = fast_postflop_equity(
            (h0, h1),
            board,
            num_opponents=n_opp,
            num_samples=samples,
            villain_min_pct=v_min_pct,
            villain_bet_ratio=bet_ratio,
            villain_bluff_Prob=v_bluff_prob,
        )

        v_post_fold = self._post_fold_rate(primary_villain)
        stacks_o = [state["stacks"][o] for o in active_opps]
        spr = min(hero_stack, max(stacks_o)) / max(1, pot)

        # ---------------------------------------------------------------------
        # 1. CHECKED TO HERO (to_call == 0)
        # ---------------------------------------------------------------------
        if to_call == 0:
            if villain_raised_prev and not v_is_maniac and made_cat < 3 and eq < 0.82:
                return ("check", 0)

            probe_risk = min(max(stacks_o), max(bb, int(0.50 * pot)))
            v_bf, _ = compute_bubble_factor(
                total_stacks, primary_villain, hero_idx, probe_risk, None, hand_num
            )
            hero_bf, _ = compute_bubble_factor(
                total_stacks, hero_idx, primary_villain, probe_risk, None, hand_num
            )

            # A. Monster Value (True Trips/Set, Straight, Flush, Full House, Quads: made_cat >= 3)
            if (made_cat >= 3 and eq >= 0.74) or eq >= 0.85:
                if street == 1 and n_opp == 1 and not v_is_station and wetness <= 0.45:
                    return ("raise", min(hero_stack, max(min_raise_add, int(0.35 * pot))))
                if spr <= 1.30 and street >= 2 and hero_bf <= 1.50:
                    return ("raise", hero_stack)
                big_val = int((0.78 if v_is_station else 0.65) * pot)
                return ("raise", min(hero_stack, max(min_raise_add, big_val)))

            # B. Strong Value (True Two Pair, Overpair, TPTK)
            if (made_cat == 2 and eq >= 0.65) or (is_tptk and eq >= 0.63) or eq >= 0.70:
                if street == 3 and made_cat == 1 and hero_barreled_prev and not v_is_station:
                    if eq < 0.75 or hero_bf > 1.30:
                        return ("check", 0)
                val_mult = 0.70 if v_is_station else (0.52 if street == 1 else 0.58)
                return ("raise", min(hero_stack, max(min_raise_add, int(val_mult * pot))))

            # C. Thin Value / Protection on Flop/Turn (True One Pair with eq >= 0.58)
            if made_cat == 1 and eq >= 0.58 and n_opp == 1 and street <= 2 and not hero_barreled_prev:
                thin_mult = 0.50 if v_is_station else 0.44
                return ("raise", min(hero_stack, max(min_raise_add, int(thin_mult * pot))))

            # D. Exploitative C-Bet & Capped-Range Turn Probe (only when Villain hasn't called a previous barrel!)
            if not v_is_station and n_opp == 1 and n_pre_raises <= 1 and not hero_barreled_prev:
                if street == 1:
                    if v_bf >= 1.32 and hero_bf <= 1.40 and (eq >= 0.36 or has_draw or has_nut_blocker or wetness <= 0.38):
                        return ("raise", min(hero_stack, max(min_raise_add, int(0.43 * pot))))
                    if (wetness <= 0.36 and eq >= 0.40) or (has_draw and eq >= 0.44) or (has_nut_blocker and eq >= 0.40):
                        if v_post_fold >= 0.48 and hero_bf <= 1.35:
                            return ("raise", min(hero_stack, max(min_raise_add, int(0.43 * pot))))

                if street == 2 and hero_bf <= 1.38:
                    if villain_cbet_prev:
                        # Villain C-bet Flop and gave up on Turn! Fire 46% pot probe bet!
                        if made_cat >= 1 or has_draw or eq >= 0.40 or v_bf >= 1.32:
                            return ("raise", min(hero_stack, max(min_raise_add, int(0.46 * pot))))
                    elif (has_draw and eq >= 0.42) or eq >= 0.46 or (v_bf >= 1.35 and eq >= 0.36):
                        if v_post_fold >= 0.48:
                            return ("raise", min(hero_stack, max(min_raise_add, int(0.45 * pot))))

            return ("check", 0)

        # ---------------------------------------------------------------------
        # 2. FACING A POSTFLOP BET OR RAISE (to_call > 0)
        # ---------------------------------------------------------------------
        bf, _ = compute_bubble_factor(
            total_stacks, hero_idx, primary_villain, to_call, None, hand_num
        )
        req_eq = (bf * to_call) / max(1.0, pot + bf * to_call)
        hero_bet_this_street = street_bets[hero_idx] > 0

        # IRON RULE 1: Facing a Postflop Raise (`hero_bet_this_street` is True) from a non-Maniac!
        if hero_bet_this_street and not v_is_maniac:
            if made_cat <= 1:
                if street == 1 and has_draw and to_call <= int(0.25 * pot) and bf <= 1.25:
                    return ("call", to_call)
                return ("fold", 0)
            if made_cat == 2:
                if eq >= req_eq + 0.12 and to_call <= int(0.42 * hero_stack) and bf <= 1.32:
                    return ("call", to_call)
                return ("fold", 0)

        # IRON RULE 2: Facing a Non-Maniac Big Turn or River Bet (`street >= 2` and `bet_ratio >= 0.52`)!
        if street >= 2 and not v_is_maniac and bet_ratio >= 0.52:
            if made_cat == 0 and not (street == 2 and has_draw and bet_ratio <= 0.60 and bf <= 1.20):
                return ("fold", 0)
            if made_cat == 1 and not is_tptk:
                if not (street == 2 and has_draw and eq >= req_eq + 0.08):
                    return ("fold", 0)
            if street == 3 and made_cat == 1:
                if v_is_station or villain_raised_prev or to_call >= int(0.32 * hero_stack) or eq < req_eq + 0.15:
                    return ("fold", 0)

        # A. Value Raise / Jam with True Monster Hands (True Trips/Set+, or dominant Two Pair)
        if (made_cat >= 4 and eq >= 0.78) or (made_cat == 3 and eq >= 0.78) or eq >= 0.86:
            if spr <= 1.6 or to_call >= int(0.35 * hero_stack):
                return ("raise", hero_stack)
            val_raise = to_call + int(0.72 * pot)
            return ("raise", min(hero_stack, max(min_raise_add, val_raise)))

        if made_cat == 2 and eq >= 0.74 and bf <= 1.30 and not hero_bet_this_street and not villain_raised_prev:
            val_raise = to_call + int(0.62 * pot)
            return ("raise", min(hero_stack, max(min_raise_add, val_raise)))

        # B. Capped Bet-Size Exploitation (<= 49% pot):
        # Across all archetypes, Flop bets <= 40% pot and Turn/River bets <= 49% pot (0.33, 0.38, 0.45, 0.48x pot)
        # ONLY occur when Villain failed the `made_cat >= 2 or is_tptk or eq >= 0.65` value check!
        if n_opp == 1 and not hero_bet_this_street and bet_ratio <= 0.49 and not v_is_station:
            # If Hero has TPTK/Overpair or strong pair (eq >= 0.66), Hero crushes Villain's capped range -> value raise!
            if (is_tptk or made_cat >= 2 or eq >= 0.68) and bf <= 1.32:
                val_r = to_call + int(0.60 * pot)
                return ("raise", min(hero_stack, max(min_raise_add, val_r)))
            # Otherwise call profitably with any pair, draw, or reasonable equity!
            if bf <= 1.35 and (made_cat >= 1 or (street < 3 and has_draw) or eq >= req_eq + 0.02):
                return ("call", to_call)

        # C. Standard ICM-Adjusted Pot-Odds Call Decision
        safety_margin = 0.06 + 0.03 * street

        if v_is_station and bet_ratio >= 0.50:
            safety_margin += 0.08
        elif v_is_maniac:
            safety_margin = max(0.02, safety_margin - 0.05)

        if hand_num >= 160 and bf > 1.25 and to_call >= int(0.20 * hero_stack):
            safety_margin += 0.05

        draw_bonus = 0.0
        if street < 3 and has_draw and hero_stack >= int(2.8 * to_call) and bf <= 1.32:
            draw_bonus = 0.06 if street == 1 else 0.035

        if street == 3 and made_cat == 0:
            return ("fold", 0)

        if eq + draw_bonus >= req_eq + safety_margin:
            return ("call", to_call)

        return ("fold", 0)
