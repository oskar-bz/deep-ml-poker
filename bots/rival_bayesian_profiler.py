"""
Rival Team #4: Bayesian Hand-Reader & Archetype Classifier (`RivalBayesianProfilerBot`)
Deep-ML 8-Player Sit-and-Go No-Limit Hold'em Tournament Bot.

Core Architecture:
1. 5-Hypothesis Bayesian Opponent Archetype Posterior per seat:
   Tracks normalized log-likelihoods across {AFK, NIT, TAG, LAG, STATION} updated after
   every hand (`on_hand_end`) and during live street actions (`action_history`),
   including Showdown Hand Calibration and decoupled positional steal vs early-open
   and fold-to-steal / fold-to-3bet tracking.
2. Precomputed O(1) Domination-Adjusted Preflop Range Equity Table:
   At module load, integrates all 169x169 canonical Hold'em matchups across 11
   opponent percentile floors (`villain_min_pct` in 0.00..0.97), capturing card
   removal and kicker domination in ~0.15 us per lookup.
3. Dynamic Multi-Street Villain Range Narrowing (`villain_min_pct`):
   Computes Villain's percentile floor from their Bayesian archetype posterior,
   seat position, preflop pot type (limped, single-raised, 3-bet, 4-bet+), and
   earlier postflop barrels/calls/checks.
4. Elastic Value Sizing, Elastic 3-Bet Sizing & Texture-Conditioned Bluff-Catching:
   - Sizes 3-bets to cross opponent continuation thresholds (`to_call > 4.0 BB` and
     `>= 0.22 * v_stack` when medium-stacked), maximizing fold equity against TAG/ICM/GTO/Trap
     openers while cleanly folding to 4-bets (`v_min_pct >= 0.93`) without KK+.
   - Sizes postflop value bets to the maximum elastic threshold of the target archetype
     (85-100% pot vs CallingStation/LAG, 58-64% pot vs Nit/TAG to keep TPTK in).
   - Sizes Flop C-bets to 43.5% pot vs TAG/GTO/Trap opponents (bypassing <=38% floats
     and <=42% check-raises) and 35.5% pot vs Nits (folding out <=33% Nit draws).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence, Tuple

from poker_engine.evaluator import (
    analyze_board_texture,
    conditional_postflop_profile,
    eval_5,
    eval_7,
)
from poker_engine.icm import compute_bubble_factor
from poker_engine.preflop_tables import (
    NASH_CALL_BB,
    NASH_PUSH_BB,
    PREFLOP_HU_EQ,
    PREFLOP_PERCENTILE,
    PREFLOP_STRENGTH,
)

# Archetype Constants
ARCH_AFK = 0
ARCH_NIT = 1
ARCH_TAG = 2
ARCH_LAG = 3
ARCH_STATION = 4
ARCH_NAMES = ("AFK", "NIT", "TAG", "LAG", "STATION")

# Prior probabilities over (AFK, NIT, TAG, LAG, STATION), normalized so max log == 0.0
PRIOR_PROBS: Tuple[float, ...] = (0.06, 0.22, 0.37, 0.18, 0.17)
_MAX_PRIOR_P = max(PRIOR_PROBS)
PRIOR_LOGS: Tuple[float, ...] = tuple(math.log(p / _MAX_PRIOR_P) for p in PRIOR_PROBS)

# Likelihood parameters per archetype: (AFK, NIT, TAG, LAG, STATION)
_L_VPIP_EARLY_YES = tuple(math.log(p) for p in (0.025, 0.14, 0.24, 0.54, 0.44))
_L_VPIP_EARLY_NO = tuple(math.log(1.0 - p) for p in (0.025, 0.14, 0.24, 0.54, 0.44))

# In late position (CO/BTN/SB) unopened, TAGs also steal wide (~0.42), so LP VPIP doesn't confuse TAG with LAG
_L_VPIP_STEAL_YES = tuple(math.log(p) for p in (0.025, 0.18, 0.44, 0.62, 0.44))
_L_VPIP_STEAL_NO = tuple(math.log(1.0 - p) for p in (0.025, 0.18, 0.44, 0.62, 0.44))

_L_PFR_GIVEN_VPIP = tuple(math.log(p) for p in (0.95, 0.45, 0.84, 0.85, 0.09))
_L_CALL_GIVEN_VPIP = tuple(math.log(1.0 - p) for p in (0.95, 0.45, 0.84, 0.85, 0.09))

# Large preflop open (>= 2.9x BB) is a strong signature of ManiacAggroBot (which opens to 4x BB)
_L_BIG_OPEN = tuple(math.log(p) for p in (0.10, 0.15, 0.12, 0.82, 0.20))
_L_SMALL_OPEN = tuple(math.log(1.0 - p) for p in (0.10, 0.15, 0.12, 0.82, 0.20))

_L_3BET_YES = tuple(math.log(p) for p in (0.02, 0.04, 0.10, 0.28, 0.02))
_L_3BET_NO = tuple(math.log(1.0 - p) for p in (0.02, 0.04, 0.10, 0.28, 0.02))

_L_FOLD_STEAL_YES = tuple(math.log(p) for p in (0.975, 0.86, 0.78, 0.44, 0.46))
_L_FOLD_STEAL_NO = tuple(math.log(1.0 - p) for p in (0.975, 0.86, 0.78, 0.44, 0.46))

_L_POST_BET = tuple(math.log(p) for p in (0.002, 0.22, 0.44, 0.74, 0.14))
_L_POST_CHECK = tuple(math.log(1.0 - p) for p in (0.002, 0.22, 0.44, 0.74, 0.14))

_L_POST_FOLD = tuple(math.log(p) for p in (0.994, 0.70, 0.52, 0.32, 0.18))
_L_POST_CALL = tuple(math.log(p) for p in (0.003, 0.25, 0.37, 0.42, 0.74))
_L_POST_RAISE = tuple(math.log(p) for p in (0.003, 0.05, 0.11, 0.26, 0.08))

# Archetype parameter anchors: (AFK, NIT, TAG, LAG, STATION)
_ARCH_OPEN_MIN_PCT = (0.975, 0.84, 0.65, 0.38, 0.80)
_ARCH_CALL_MIN_PCT = (0.975, 0.80, 0.64, 0.36, 0.24)
_ARCH_3BET_MIN_PCT = (0.975, 0.93, 0.86, 0.56, 0.92)
_ARCH_BLUFF_PROB = (0.01, 0.05, 0.17, 0.42, 0.06)
_ARCH_FOLD_TENDENCY = (0.99, 0.70, 0.53, 0.34, 0.18)
_ARCH_FOLD_STEAL = (0.975, 0.85, 0.79, 0.45, 0.45)

# 11 Range Floor Buckets for O(1) Domination-Adjusted Preflop Equity Lookup
RANGE_BUCKETS: Tuple[float, ...] = (
    0.00, 0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97
)
PREFLOP_EQ_VS_RANGE: List[List[List[float]]] = [
    [[0.5] * 52 for _ in range(52)] for _ in range(len(RANGE_BUCKETS))
]


def _pairwise_canonical_equity(
    r1: int, r2: int, s1: bool, r3: int, r4: int, s2: bool
) -> float:
    """
    Fast analytical Texas Hold'em preflop equity of canonical hand (r1 >= r2, s1)
    vs canonical hand (r3 >= r4, s2), accurately modeling pair vs overcards,
    dominated Ace/King kickers, and suitedness.
    """
    pair_a = r1 == r2
    pair_b = r3 == r4
    suit_adj = (0.026 if s1 else 0.0) - (0.026 if s2 else 0.0)

    if pair_a and pair_b:
        if r1 > r3:
            return 0.815
        if r1 < r3:
            return 0.185
        return 0.500

    if pair_a and not pair_b:
        if r1 > r3:
            conn_pen = -0.025 if (r3 - r4 <= 2 and r4 >= 3) else 0.0
            return min(0.88, max(0.75, 0.825 + suit_adj + conn_pen))
        if r1 == r3:
            return min(0.95, 0.922 + suit_adj)
        if r1 == r4:
            return min(0.76, max(0.64, 0.695 + suit_adj))
        if r3 > r1 > r4:
            return min(0.76, max(0.65, 0.715 + suit_adj))
        conn_pen = -0.018 if (r3 - r4 <= 2) else 0.0
        return min(0.58, max(0.49, 0.542 + suit_adj + conn_pen))

    if not pair_a and pair_b:
        return 1.0 - _pairwise_canonical_equity(r3, r4, s2, r1, r2, s1)

    if r1 == r3 and r2 == r4:
        return 0.500 + suit_adj
    if r1 == r3:
        base = 0.738 if r2 > r4 else 0.262
        return min(0.82, max(0.18, base + suit_adj))
    if r2 == r4:
        base = 0.722 if r1 > r3 else 0.278
        return min(0.80, max(0.20, base + suit_adj))
    if r2 == r3:
        return min(0.78, max(0.62, 0.705 + suit_adj))
    if r1 == r4:
        return min(0.38, max(0.22, 0.295 + suit_adj))

    if r1 > r3 and r2 > r3:
        base = 0.645
    elif r1 > r3 > r2 > r4:
        base = 0.618
    elif r1 > r3 > r4 > r2:
        base = 0.582
    elif r3 > r1 and r4 > r1:
        base = 0.355
    elif r3 > r1 > r4 > r2:
        base = 0.382
    else:
        base = 0.418

    return min(0.85, max(0.15, base + suit_adj))


def _init_range_equity_tables() -> None:
    """Precomputes O(1) domination-adjusted preflop equity tables across 11 range floors."""
    canon_list: List[Tuple[int, int, bool, float, int]] = []
    for r1 in range(13):
        for r2 in range(r1 + 1):
            if r1 == r2:
                c0 = r1 << 2
                c1 = (r1 << 2) | 1
                pct = PREFLOP_PERCENTILE[c0][c1]
                canon_list.append((r1, r2, False, pct, 6))
            else:
                c0 = r1 << 2
                c1_s = r2 << 2
                c1_o = (r2 << 2) | 1
                canon_list.append((r1, r2, True, PREFLOP_PERCENTILE[c0][c1_s], 4))
                canon_list.append((r1, r2, False, PREFLOP_PERCENTILE[c0][c1_o], 12))

    n_canon = len(canon_list)
    pair_eq = [[0.5] * n_canon for _ in range(n_canon)]
    pair_w = [[0.0] * n_canon for _ in range(n_canon)]

    for i in range(n_canon):
        r1, r2, s1, _, _ = canon_list[i]
        for j in range(n_canon):
            r3, r4, s2, _, combos_b = canon_list[j]
            eq = _pairwise_canonical_equity(r1, r2, s1, r3, r4, s2)
            pair_eq[i][j] = eq
            shared = (1 if (r3 == r1 or r3 == r2) else 0) + (
                1 if (r4 != r3 and (r4 == r1 or r4 == r2)) else 0
            )
            rem_mult = 1.0 if shared == 0 else (0.68 if shared == 1 else 0.33)
            pair_w[i][j] = combos_b * rem_mult

    canon_vs_bucket = [[0.5] * n_canon for _ in range(len(RANGE_BUCKETS))]
    for b_idx, min_pct in enumerate(RANGE_BUCKETS):
        for i in range(n_canon):
            tot_w = 0.0
            tot_eq = 0.0
            for j in range(n_canon):
                pct_b = canon_list[j][3]
                w = pair_w[i][j] * (1.0 if pct_b >= min_pct else 0.025)
                tot_w += w
                tot_eq += w * pair_eq[i][j]
            canon_vs_bucket[b_idx][i] = tot_eq / max(1e-9, tot_w)

    canon_idx_map: Dict[Tuple[int, int, bool], int] = {}
    for idx, (r1, r2, s1, _, _) in enumerate(canon_list):
        canon_idx_map[(r1, r2, s1)] = idx

    for c0 in range(52):
        r0 = c0 >> 2
        s0 = c0 & 3
        for c1 in range(52):
            if c0 == c1:
                continue
            r1 = c1 >> 2
            s1 = c1 & 3
            hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
            suited = (s0 == s1) and (hi != lo)
            c_idx = canon_idx_map[(hi, lo, suited)]
            for b_idx in range(len(RANGE_BUCKETS)):
                PREFLOP_EQ_VS_RANGE[b_idx][c0][c1] = canon_vs_bucket[b_idx][c_idx]


_init_range_equity_tables()


def lookup_preflop_eq_vs_range(c0: int, c1: int, v_min_pct: float) -> float:
    """O(1) interpolated lookup of Hero's preflop equity against [v_min_pct, 1.0]."""
    if v_min_pct <= 0.02:
        return PREFLOP_HU_EQ[c0][c1]
    if v_min_pct >= 0.97:
        return PREFLOP_EQ_VS_RANGE[-1][c0][c1]
    for b in range(len(RANGE_BUCKETS) - 1):
        lo_p = RANGE_BUCKETS[b]
        hi_p = RANGE_BUCKETS[b + 1]
        if v_min_pct <= hi_p:
            t = (v_min_pct - lo_p) / (hi_p - lo_p)
            e0 = PREFLOP_EQ_VS_RANGE[b][c0][c1]
            e1 = PREFLOP_EQ_VS_RANGE[b + 1][c0][c1]
            return e0 + t * (e1 - e0)
    return PREFLOP_EQ_VS_RANGE[-1][c0][c1]


class RivalBayesianProfilerBot:
    """
    Rival Team #4: Bayesian Hand-Reader & Archetype Classifier Bot.
    Maintains a 5-archetype Bayesian posterior per opponent seat and dynamically
    narrows opponent ranges across streets for precision value sizing and bluff-catching.
    """

    name: str = "RivalBayesianProfiler"

    def __init__(self) -> None:
        self._reset_profiles()

    def _reset_profiles(self) -> None:
        self.log_post: List[List[float]] = [list(PRIOR_LOGS) for _ in range(8)]
        self.posterior: List[List[float]] = [list(PRIOR_PROBS) for _ in range(8)]
        self.never_afk: List[bool] = [False] * 8

        # Empirical counters per seat
        self.hands_dealt: List[int] = [0] * 8
        self.vpip_count: List[int] = [0] * 8
        self.pfr_count: List[int] = [0] * 8
        self.ep_hands: List[int] = [0] * 8
        self.ep_pfr: List[int] = [0] * 8
        self.lp_hands: List[int] = [0] * 8
        self.lp_pfr: List[int] = [0] * 8
        self.threebet_count: List[int] = [0] * 8
        self.faced_steal: List[int] = [0] * 8
        self.folded_steal: List[int] = [0] * 8
        self.faced_3bet: List[int] = [0] * 8
        self.folded_3bet: List[int] = [0] * 8

        self.post_faced_bet: List[int] = [0] * 8
        self.post_folded_bet: List[int] = [0] * 8
        self.post_called_bet: List[int] = [0] * 8
        self.post_raised_bet: List[int] = [0] * 8
        self.post_checked_op: List[int] = [0] * 8
        self.post_bet_op: List[int] = [0] * 8

        # Showdown-verified counters
        self.sd_count: List[int] = [0] * 8
        self.sd_bluffs: List[int] = [0] * 8
        self.sd_light_calls: List[int] = [0] * 8
        self.sd_open_pct_sum: List[float] = [0.0] * 8
        self.sd_open_pct_n: List[int] = [0] * 8

        # Derived O(1) Bayesian seat parameters used in `act(state)`
        self.open_min_pct: List[float] = [0.68] * 8
        self.steal_min_pct: List[float] = [0.54] * 8
        self.call_min_pct: List[float] = [0.58] * 8
        self.threebet_min_pct: List[float] = [0.85] * 8
        self.bluff_prob: List[float] = [0.18] * 8
        self.fold_tendency: List[float] = [0.54] * 8
        self.fold_to_steal: List[float] = [0.76] * 8
        self.fold_to_3bet: List[float] = [0.80] * 8
        for s in range(8):
            self._recompute_seat_params(s)

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self._reset_profiles()

    def _recompute_seat_params(self, s: int) -> None:
        """Computes softmax posterior and blends with empirical Beta-Binomial statistics."""
        logs = self.log_post[s]
        if self.never_afk[s]:
            max_l = max(logs[1], logs[2], logs[3], logs[4])
        else:
            max_l = max(logs)

        for k in range(5):
            if k == ARCH_AFK and self.never_afk[s]:
                logs[k] = -30.0
            else:
                diff = logs[k] - max_l
                logs[k] = -9.0 if diff < -9.0 else diff

        exps = [math.exp(l) for l in logs]
        tot = sum(exps)
        post = [e / tot for e in exps]
        self.posterior[s] = post

        p_open = sum(post[k] * _ARCH_OPEN_MIN_PCT[k] for k in range(5))
        p_call = sum(post[k] * _ARCH_CALL_MIN_PCT[k] for k in range(5))
        p_3bet = sum(post[k] * _ARCH_3BET_MIN_PCT[k] for k in range(5))
        p_bluff = sum(post[k] * _ARCH_BLUFF_PROB[k] for k in range(5))
        p_fold = sum(post[k] * _ARCH_FOLD_TENDENCY[k] for k in range(5))
        p_fsteal = sum(post[k] * _ARCH_FOLD_STEAL[k] for k in range(5))

        n_hands = self.hands_dealt[s]
        if self.ep_hands[s] >= 3:
            emp_ep_pfr = (self.ep_pfr[s] + 0.8) / (self.ep_hands[s] + 4.0)
            emp_open_pct = max(0.20, min(0.95, 1.0 - 1.15 * emp_ep_pfr))
            w_emp = min(0.45, self.ep_hands[s] / 25.0)
            p_open = (1.0 - w_emp) * p_open + w_emp * emp_open_pct

        p_steal = max(0.18, p_open - 0.14 * post[ARCH_TAG] - 0.10 * post[ARCH_LAG])
        if self.lp_hands[s] >= 2:
            emp_lp_pfr = (self.lp_pfr[s] + 1.0) / (self.lp_hands[s] + 3.0)
            emp_steal_pct = max(0.18, min(0.92, 1.0 - 1.05 * emp_lp_pfr))
            w_lp = min(0.55, self.lp_hands[s] / 15.0)
            p_steal = (1.0 - w_lp) * p_steal + w_lp * emp_steal_pct

        if n_hands >= 4:
            emp_vpip = (self.vpip_count[s] + 1.4) / (n_hands + 6.0)
            emp_call_pct = max(0.15, min(0.90, 1.0 - 1.05 * emp_vpip))
            w_emp = min(0.45, n_hands / 40.0)
            p_call = (1.0 - w_emp) * p_call + w_emp * emp_call_pct

        if self.sd_open_pct_n[s] >= 1:
            avg_sd_open = self.sd_open_pct_sum[s] / self.sd_open_pct_n[s]
            est_min_from_sd = max(0.15, min(0.92, 1.75 * avg_sd_open - 0.72))
            p_open = 0.75 * p_open + 0.25 * est_min_from_sd

        if self.faced_steal[s] >= 1:
            emp_fsteal = (self.folded_steal[s] + 3.2) / (self.faced_steal[s] + 4.2)
            p_fsteal = 0.35 * p_fsteal + 0.65 * emp_fsteal

        p_f3bet = 0.84 * (post[ARCH_NIT] + post[ARCH_TAG]) + 0.98 * post[ARCH_AFK] + 0.45 * post[ARCH_LAG] + 0.35 * post[ARCH_STATION]
        if self.faced_3bet[s] >= 1:
            emp_f3b = (self.folded_3bet[s] + 2.5 * p_f3bet) / (self.faced_3bet[s] + 2.5)
            p_f3bet = 0.40 * p_f3bet + 0.60 * emp_f3b

        if self.post_faced_bet[s] >= 2:
            emp_pfold = (self.post_folded_bet[s] + 2.5 * p_fold) / (self.post_faced_bet[s] + 2.5)
            p_fold = 0.45 * p_fold + 0.55 * emp_pfold

        if self.sd_count[s] >= 1 and self.sd_bluffs[s] >= 1:
            bluff_rate_sd = self.sd_bluffs[s] / max(1, self.sd_count[s])
            p_bluff = max(p_bluff, min(0.52, 0.55 * p_bluff + 0.45 * (bluff_rate_sd + 0.15)))

        self.open_min_pct[s] = max(0.15, min(0.96, p_open))
        self.steal_min_pct[s] = max(0.15, min(0.94, p_steal))
        self.call_min_pct[s] = max(0.12, min(0.92, p_call))
        self.threebet_min_pct[s] = max(0.45, min(0.97, p_3bet))
        self.bluff_prob[s] = max(0.02, min(0.52, p_bluff))
        self.fold_tendency[s] = max(0.08, min(0.98, p_fold))
        self.fold_to_steal[s] = max(0.20, min(0.98, p_fsteal))
        self.fold_to_3bet[s] = max(0.20, min(0.98, p_f3bet))

    def on_hand_end(self, hero_idx: int, hand_summary: Dict[str, Any]) -> None:
        """Updates the 5-archetype Bayesian posterior for all opponents after each hand."""
        starting_stacks = hand_summary.get("starting_stacks", [])
        action_history = hand_summary.get("action_history", [])
        showdown_hands = hand_summary.get("showdown_hands", {})
        board = hand_summary.get("board", [])

        if not starting_stacks or not action_history:
            return

        bb_amt = 20
        sb_seat = -1
        bb_seat = -1
        for st, s, act_t, amt in action_history:
            if st == 0 and act_t == "sb":
                sb_seat = s
            elif st == 0 and act_t == "bb" and amt > 0:
                bb_seat = s
                bb_amt = amt
                break

        vpip = [False] * 8
        pfr = [False] * 8
        opened_unopened = [False] * 8
        open_size_bb = [0.0] * 8
        was_unopened_op = [False] * 8
        three_bet = [False] * 8
        faced_steal = [False] * 8
        folded_steal = [False] * 8
        faced_3bet = [False] * 8
        folded_3bet = [False] * 8
        had_pre_decision = [False] * 8

        post_bets = [0] * 8
        post_checks = [0] * 8
        post_folds = [0] * 8
        post_calls = [0] * 8
        post_raises = [0] * 8
        late_street_aggr = [False] * 8
        late_street_call = [False] * 8

        cur_street = -1
        cur_high_bet = 0
        seat_street_bet = [0] * 8
        pre_raise_count = 0
        first_raiser = -1

        for st, s, act_t, amt in action_history:
            if st != cur_street:
                cur_street = st
                cur_high_bet = 0
                for i in range(8):
                    seat_street_bet[i] = 0

            if act_t in ("sb", "bb"):
                seat_street_bet[s] += amt
                if seat_street_bet[s] > cur_high_bet:
                    cur_high_bet = seat_street_bet[s]
                continue

            facing_bet = cur_high_bet > seat_street_bet[s]
            if st == 0:
                had_pre_decision[s] = True
                if pre_raise_count == 0:
                    was_unopened_op[s] = True
                elif pre_raise_count == 1 and facing_bet:
                    faced_steal[s] = True
                    if act_t == "fold":
                        folded_steal[s] = True
                elif pre_raise_count >= 2 and s == first_raiser and facing_bet:
                    faced_3bet[s] = True
                    if act_t == "fold":
                        folded_3bet[s] = True

                if act_t == "call":
                    vpip[s] = True
                    self.never_afk[s] = True
                    seat_street_bet[s] += amt
                elif act_t == "raise":
                    vpip[s] = True
                    pfr[s] = True
                    seat_street_bet[s] += amt
                    if pre_raise_count == 0:
                        opened_unopened[s] = True
                        first_raiser = s
                        open_size_bb[s] = seat_street_bet[s] / max(1, bb_amt)
                    else:
                        three_bet[s] = True
                    pre_raise_count += 1
                    if seat_street_bet[s] > cur_high_bet:
                        cur_high_bet = seat_street_bet[s]
                    if seat_street_bet[s] < starting_stacks[s]:
                        self.never_afk[s] = True
            else:
                if not facing_bet:
                    if act_t == "check":
                        post_checks[s] += 1
                    elif act_t == "raise":
                        post_bets[s] += 1
                        self.never_afk[s] = True
                        seat_street_bet[s] += amt
                        if seat_street_bet[s] > cur_high_bet:
                            cur_high_bet = seat_street_bet[s]
                        if st >= 2:
                            late_street_aggr[s] = True
                else:
                    if act_t == "fold":
                        post_folds[s] += 1
                    elif act_t == "call":
                        post_calls[s] += 1
                        self.never_afk[s] = True
                        seat_street_bet[s] += amt
                        if st >= 2:
                            late_street_call[s] = True
                    elif act_t == "raise":
                        post_raises[s] += 1
                        self.never_afk[s] = True
                        seat_street_bet[s] += amt
                        if seat_street_bet[s] > cur_high_bet:
                            cur_high_bet = seat_street_bet[s]
                        if st >= 2:
                            late_street_aggr[s] = True

        board_only_score = 0
        if len(board) == 5:
            board_only_score = eval_5(board[0], board[1], board[2], board[3], board[4])

        # Count alive players in this hand to distinguish late-position steal vs early-position open
        alive_in_hand = [i for i in range(8) if starting_stacks[i] > 0]
        n_alive = len(alive_in_hand)

        for s in range(8):
            if s == hero_idx or starting_stacks[s] <= 0:
                continue

            self.hands_dealt[s] += 1
            logs = self.log_post[s]

            for k in range(5):
                logs[k] = PRIOR_LOGS[k] + 0.985 * (logs[k] - PRIOR_LOGS[k])

            # Approximate players behind `s` before BB
            dist_to_bb = (bb_seat - s) & 7 if bb_seat >= 0 else 3
            is_lp_steal_spot = (dist_to_bb <= 3) or (n_alive <= 4)

            if was_unopened_op[s]:
                if is_lp_steal_spot:
                    self.lp_hands[s] += 1
                    if opened_unopened[s]:
                        self.lp_pfr[s] += 1
                else:
                    self.ep_hands[s] += 1
                    if opened_unopened[s]:
                        self.ep_pfr[s] += 1

            if vpip[s]:
                self.vpip_count[s] += 1
                l_vpip = _L_VPIP_STEAL_YES if (was_unopened_op[s] and is_lp_steal_spot) else _L_VPIP_EARLY_YES
                for k in range(5):
                    logs[k] += l_vpip[k]
                if pfr[s]:
                    self.pfr_count[s] += 1
                    for k in range(5):
                        logs[k] += _L_PFR_GIVEN_VPIP[k]
                    # Check open sizing signature: ManiacAggroBot opens to 4.0x BB (or shoves), while TAGs open 2.0-2.5x BB
                    if opened_unopened[s] and seat_street_bet[s] < starting_stacks[s]:
                        if open_size_bb[s] >= 2.9:
                            for k in range(5):
                                logs[k] += 0.65 * _L_BIG_OPEN[k]
                        else:
                            for k in range(5):
                                logs[k] += 0.35 * _L_SMALL_OPEN[k]
                    if three_bet[s]:
                        self.threebet_count[s] += 1
                        for k in range(5):
                            logs[k] += _L_3BET_YES[k]
                    elif faced_steal[s]:
                        for k in range(5):
                            logs[k] += _L_3BET_NO[k]
                else:
                    for k in range(5):
                        logs[k] += _L_CALL_GIVEN_VPIP[k]
            elif had_pre_decision[s]:
                l_no = _L_VPIP_STEAL_NO if (was_unopened_op[s] and is_lp_steal_spot) else _L_VPIP_EARLY_NO
                for k in range(5):
                    logs[k] += l_no[k]

            if faced_steal[s]:
                self.faced_steal[s] += 1
                if folded_steal[s]:
                    self.folded_steal[s] += 1
                    for k in range(5):
                        logs[k] += 0.45 * _L_FOLD_STEAL_YES[k]
                else:
                    for k in range(5):
                        logs[k] += 0.45 * _L_FOLD_STEAL_NO[k]

            if faced_3bet[s]:
                self.faced_3bet[s] += 1
                if folded_3bet[s]:
                    self.folded_3bet[s] += 1

            nb = post_bets[s]
            nc = post_checks[s]
            nf = post_folds[s]
            nca = post_calls[s]
            nr = post_raises[s]

            self.post_bet_op[s] += nb
            self.post_checked_op[s] += nc + nb
            self.post_faced_bet[s] += nf + nca + nr
            self.post_folded_bet[s] += nf
            self.post_called_bet[s] += nca
            self.post_raised_bet[s] += nr

            if nb + nc + nf + nca + nr > 0:
                for k in range(5):
                    logs[k] += (
                        nb * _L_POST_BET[k]
                        + nc * _L_POST_CHECK[k]
                        + nf * _L_POST_FOLD[k]
                        + nca * _L_POST_CALL[k]
                        + nr * _L_POST_RAISE[k]
                    )

            if s in showdown_hands and len(board) == 5:
                self.sd_count[s] += 1
                h0, h1 = showdown_hands[s]
                sd_pct = PREFLOP_PERCENTILE[h0][h1]
                sd_score = eval_7(h0, h1, board[0], board[1], board[2], board[3], board[4])
                sd_cat = sd_score >> 20

                if pfr[s]:
                    if not is_lp_steal_spot:
                        self.sd_open_pct_sum[s] += sd_pct
                        self.sd_open_pct_n[s] += 1
                    if sd_pct < 0.52 and not is_lp_steal_spot:
                        logs[ARCH_LAG] += 1.75
                        logs[ARCH_NIT] -= 2.10
                        logs[ARCH_STATION] -= 1.20
                    elif sd_pct >= 0.88:
                        logs[ARCH_NIT] += 0.65
                        logs[ARCH_TAG] += 0.45
                elif vpip[s]:
                    if sd_pct < 0.55:
                        logs[ARCH_STATION] += 1.45
                        logs[ARCH_NIT] -= 1.60
                    elif sd_pct >= 0.82:
                        logs[ARCH_NIT] += 0.80

                if late_street_aggr[s]:
                    if sd_cat == 0 or sd_score <= board_only_score:
                        self.sd_bluffs[s] += 1
                        logs[ARCH_LAG] += 1.90
                        logs[ARCH_TAG] += 0.30
                        logs[ARCH_NIT] -= 2.20
                        logs[ARCH_STATION] -= 2.00
                    elif sd_cat >= 2:
                        logs[ARCH_TAG] += 0.35
                        logs[ARCH_NIT] += 0.35

                if late_street_call[s] and not late_street_aggr[s]:
                    if sd_cat <= 1:
                        self.sd_light_calls[s] += 1
                        logs[ARCH_STATION] += 1.25
                        logs[ARCH_NIT] -= 1.10

            self._recompute_seat_params(s)

    def _infer_villain_range_and_postflop_params(
        self,
        state: Dict[str, Any],
        villain: int,
        bet_ratio: float,
        wetness: float,
    ) -> Tuple[float, float, float, bool]:
        """
        Dynamically computes for `villain`:
          (villain_min_pct, eff_bluff_prob, eff_fold_tendency, villain_raised_this_street)
        """
        post = self.posterior[villain]
        p_nit = post[ARCH_NIT]
        p_tag = post[ARCH_TAG]
        p_lag = post[ARCH_LAG]
        p_station = post[ARCH_STATION]

        button = state["button_idx"]
        bb_seat = state.get("bb_seat", (button + 2) & 7)
        sb_seat = state.get("sb_seat", (button + 1) & 7)
        v_dist = (button - villain) & 7
        is_v_late = (v_dist <= 2) or (villain == sb_seat)
        pos_adj = -0.05 if is_v_late else (0.025 * (v_dist - 2.5) if villain != bb_seat else 0.0)

        action_history = state.get("action_history", [])
        cur_street = state["street"]

        pre_raises = 0
        v_pre_raises = 0
        v_pre_calls = 0
        v_post_bets = 0
        v_post_raises = 0
        v_post_calls = 0
        v_post_checks = 0
        v_raised_this_street = False
        street_bet_occurred = False
        last_st = -1

        for st, s, act_t, _amt in action_history:
            if st != last_st:
                last_st = st
                street_bet_occurred = False
            if st == 0:
                if act_t == "raise":
                    pre_raises += 1
                    if s == villain:
                        v_pre_raises += 1
                elif act_t == "call" and s == villain:
                    v_pre_calls += 1
            else:
                if act_t == "raise":
                    if s == villain:
                        if street_bet_occurred:
                            v_post_raises += 1
                            if st == cur_street:
                                v_raised_this_street = True
                        else:
                            v_post_bets += 1
                    street_bet_occurred = True
                elif act_t == "call" and s == villain:
                    v_post_calls += 1
                elif act_t == "check" and s == villain:
                    v_post_checks += 1

        if pre_raises == 0:
            v_min_pct = 0.02 if villain == bb_seat else max(0.08, self.call_min_pct[villain] - 0.18)
        elif pre_raises == 1:
            if v_pre_raises >= 1:
                base_o = self.steal_min_pct[villain] if is_v_late else (self.open_min_pct[villain] + pos_adj)
                v_min_pct = max(0.16, min(0.92, base_o))
            else:
                bb_disc = -0.14 if villain == bb_seat else 0.0
                v_min_pct = max(0.12, min(0.86, self.call_min_pct[villain] + bb_disc))
        elif pre_raises == 2:
            if v_pre_raises >= 1 and v_pre_calls == 0:
                v_min_pct = max(0.42, min(0.95, self.threebet_min_pct[villain] + 0.4 * pos_adj))
            else:
                v_min_pct = max(0.38, min(0.88, self.threebet_min_pct[villain] - 0.10))
        else:
            v_min_pct = max(0.65, min(0.97, self.threebet_min_pct[villain] + 0.06))

        if v_post_bets >= 1:
            barrel_step = 0.065 + 0.055 * (p_nit + p_station) - 0.035 * p_lag
            v_min_pct = min(0.93, v_min_pct + v_post_bets * barrel_step)
        if v_post_raises >= 1:
            raise_step = 0.11 + 0.06 * (p_nit + p_station) - 0.04 * p_lag
            v_min_pct = min(0.95, v_min_pct + v_post_raises * raise_step)
        if v_post_calls >= 1 and v_post_bets == 0 and v_post_raises == 0:
            v_min_pct = min(0.84, v_min_pct + 0.04 * v_post_calls)
        if v_post_checks >= 1 and v_post_bets == 0 and v_post_raises == 0:
            v_min_pct = max(0.05, v_min_pct - 0.04 * v_post_checks)

        eff_bluff = self.bluff_prob[villain]
        eff_fold = self.fold_tendency[villain]

        board = state["board"]
        if cur_street == 1:
            if wetness <= 0.32:
                eff_bluff += 0.05 * (p_tag + p_lag)
        elif cur_street == 3 and len(board) == 5:
            f_suits = [0, 0, 0, 0]
            r_suits = [0, 0, 0, 0]
            for idx_b, c in enumerate(board):
                s_c = c & 3
                if idx_b < 4:
                    f_suits[s_c] += 1
                r_suits[s_c] += 1
            missed_fd = max(f_suits) == 2 and max(r_suits) == 2
            if missed_fd or (wetness >= 0.30 and max(r_suits) < 3):
                eff_bluff += 0.07 * (p_lag + 0.6 * p_tag)
            elif wetness <= 0.20:
                eff_bluff -= 0.04 * (p_nit + p_tag)

        # Bayesian Bet-Size Conditioning: P(Range | BetSize, Archetype)
        if bet_ratio > 0.05 and not v_raised_this_street and p_lag < 0.45:
            if 0.30 <= bet_ratio <= 0.49:
                # Capped c-bet / blocker-bluff / turn-probe sizing branch (33%, 38%, 45%, 48% pot)
                eff_bluff = max(eff_bluff, 0.34 * p_tag + 0.15)
            elif 0.52 <= bet_ratio <= 0.72:
                # Uncapped value-bet sizing branch (55%, 58%, 60%, 62%, 65% pot) -> near-zero bluffs!
                eff_bluff *= 0.28

        if v_raised_this_street:
            eff_bluff *= 0.30 + 0.55 * p_lag
            eff_fold *= 0.55
        elif v_post_bets >= 2 and cur_street >= 2:
            eff_bluff *= 0.55 + 0.45 * p_lag

        if bet_ratio >= 0.65 and (p_nit + p_station) >= 0.50:
            eff_bluff *= 0.65

        eff_bluff = max(0.015, min(0.55, eff_bluff))
        eff_fold = max(0.06, min(0.96, eff_fold))
        return v_min_pct, eff_bluff, eff_fold, v_raised_this_street

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        to_call = state["to_call"]
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        active = state["active_seats"]
        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        if state["street"] == 0:
            return self._act_preflop(state, active_opps)
        return self._act_postflop(state, active_opps)

    def _act_preflop(
        self, state: Dict[str, Any], active_opps: List[int]
    ) -> Tuple[str, int]:
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        street_bets = state["street_bets"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = max(1, state["big_blind"])
        min_raise_add = state["min_raise_add"]
        hand_num = state.get("hand_number", 1)
        h0, h1 = state["hole_cards"]
        pct = PREFLOP_PERCENTILE[h0][h1]
        hu_eq = PREFLOP_HU_EQ[h0][h1]

        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        button = state["button_idx"]
        bb_seat = state.get("bb_seat", (button + 2) & 7)
        sb_seat = state.get("sb_seat", (button + 1) & 7)

        action_history = state.get("action_history", [])
        pre_raises = 0
        last_raiser = -1
        limpers = 0
        for st, s, act_t, _amt in action_history:
            if st == 0:
                if act_t == "raise":
                    pre_raises += 1
                    last_raiser = s
                elif act_t == "call":
                    limpers += 1

        opps_behind: List[int] = []
        if hero_idx != bb_seat:
            cur = (hero_idx + 1) & 7
            for _ in range(7):
                if cur != hero_idx and state["active_seats"][cur] and stacks[cur] > 0:
                    opps_behind.append(cur)
                if cur == bb_seat:
                    break
                cur = (cur + 1) & 7

        primary_villain = (
            last_raiser
            if last_raiser >= 0 and state["active_seats"][last_raiser]
            else max(active_opps, key=lambda o: (street_bets[o], total_stacks[o]))
        )

        eff_chips = min(
            hero_stack + street_bets[hero_idx],
            max(stacks[o] + street_bets[o] for o in active_opps),
        )
        eff_bb = eff_chips / bb

        # CASE 1: Unopened Pot (or limped pot with no raises)
        if pre_raises == 0 and max(street_bets) <= bb:
            if not opps_behind:
                if limpers >= 1 and pct >= 0.74:
                    iso_size = int((2.6 + 0.8 * limpers) * bb) - street_bets[hero_idx]
                    return ("raise", min(hero_stack, max(min_raise_add, iso_size)))
                return ("check", 0)

            avg_fold_steal = (
                sum(self.fold_to_steal[o] for o in opps_behind) / len(opps_behind)
            )
            p_all_fold = 1.0
            all_afk = True
            any_station_or_lag = False
            for o in opps_behind:
                p_all_fold *= self.fold_to_steal[o]
                if self.posterior[o][ARCH_AFK] < 0.72:
                    all_afk = False
                if (self.posterior[o][ARCH_STATION] + self.posterior[o][ARCH_LAG]) > 0.48:
                    any_station_or_lag = True

            if all_afk and len(opps_behind) <= 3 and limpers == 0:
                steal_add = max(min_raise_add, int(2.0 * bb) - street_bets[hero_idx])
                if steal_add <= hero_stack:
                    return ("raise", steal_add)

            bf, _ = compute_bubble_factor(
                total_stacks, hero_idx, primary_villain, min(hero_stack, int(2.2 * bb)), pot, hand_num
            )

            # High Fold-Equity Blind Steal from CO / BTN / SB (len(opps_behind) <= 3)
            if (
                limpers == 0
                and len(opps_behind) <= 3
                and eff_bb >= 6.0
                and not any_station_or_lag
                and bf <= 1.38
            ):
                if avg_fold_steal >= 0.74 and pct >= 0.20:
                    steal_add = max(min_raise_add, int(2.0 * bb) - street_bets[hero_idx])
                    return ("raise", min(hero_stack, steal_add))
                if avg_fold_steal >= 0.68 and len(opps_behind) <= 2 and pct >= 0.32:
                    steal_add = max(min_raise_add, int(2.0 * bb) - street_bets[hero_idx])
                    return ("raise", min(hero_stack, steal_add))

            # Short-Stack Nash Push/Fold (eff_bb <= 12.0)
            if eff_bb <= 12.0:
                shove_bf, _ = compute_bubble_factor(
                    total_stacks, hero_idx, primary_villain, eff_chips, pot, hand_num
                )
                fold_scale = 0.85 + 0.45 * p_all_fold
                pos_div = (1.0 + 0.12 * max(0, len(opps_behind) - 1)) * (shove_bf ** 1.15)
                req_push_bb = (NASH_PUSH_BB[h0][h1] * fold_scale) / max(0.5, pos_div)
                if eff_bb <= req_push_bb and (pct >= 0.46 or hu_eq >= 0.54):
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            base_open = 0.44 + 0.065 * len(opps_behind) + (bf - 1.08) * 0.16
            if any_station_or_lag:
                base_open += 0.05

            if limpers >= 1:
                if pct >= max(0.72, base_open + 0.08):
                    iso_add = max(min_raise_add, int((2.65 + 0.8 * limpers) * bb) - street_bets[hero_idx])
                    return ("raise", min(hero_stack, iso_add))
                if to_call <= bb and (PREFLOP_STRENGTH[h0][h1] >= 0.60 or (h0 >> 2) == (h1 >> 2)):
                    return ("call", to_call)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            if pct >= base_open:
                if pct >= 0.66 and eff_bb >= 15.0:
                    open_target = int(2.62 * bb) + 1
                else:
                    open_target = int(2.05 * bb)
                open_add = max(min_raise_add, open_target - street_bets[hero_idx])
                return ("raise", min(hero_stack, open_add))

            return ("check", 0) if to_call == 0 else ("fold", 0)

        # CASE 2: Facing a Raise, 3-Bet, or All-In Shove (`pre_raises >= 1`)
        v_post = self.posterior[primary_villain]
        p_v_afk = v_post[ARCH_AFK]
        p_v_nit = v_post[ARCH_NIT]
        p_v_lag = v_post[ARCH_LAG]
        p_v_station = v_post[ARCH_STATION]

        v_dist = (button - primary_villain) & 7
        is_v_late = (v_dist <= 2) or (primary_villain == sb_seat)
        pos_adj = -0.05 if is_v_late else (0.025 * (v_dist - 2.5) if primary_villain != bb_seat else 0.0)
        raise_total_bb = max(street_bets) / bb

        if p_v_afk > 0.65 and raise_total_bb >= 8.0:
            v_min_pct = 0.975
        elif pre_raises == 1:
            base_o = self.steal_min_pct[primary_villain] if (is_v_late and raise_total_bb <= 2.4) else (self.open_min_pct[primary_villain] + pos_adj)
            v_min_pct = max(0.18, min(0.93, base_o))
            if raise_total_bb >= 6.0:
                v_min_pct = max(0.42, min(0.95, v_min_pct + 0.06 * (1.0 - p_v_lag)))
        elif pre_raises == 2:
            # Note: If Villain 3-bet or 4-bet over a raise, check archetype!
            v_min_pct = max(0.48, min(0.95, self.threebet_min_pct[primary_villain] + 0.4 * pos_adj))
        else:
            # Facing a 4-bet+: Nit/TAG/Trap/ICM_Exploiter ONLY 4-bet with pct >= 0.92..0.945!
            v_min_pct = 0.93 if p_v_lag < 0.45 else 0.78

        hero_eq_vs_v = lookup_preflop_eq_vs_range(h0, h1, v_min_pct)
        bf, req_eq = compute_bubble_factor(
            total_stacks, hero_idx, primary_villain, to_call, pot, hand_num
        )

        r0, r1 = h0 >> 2, h1 >> 2
        is_pair = r0 == r1
        is_suited = (h0 & 3) == (h1 & 3)
        has_ace_blocker = (r0 == 12 or r1 == 12) and (is_suited or min(r0, r1) >= 8)

        # 2a. Facing an All-In or committing raise (`to_call >= 0.38 * hero_stack` or `eff_bb <= 12.5`)
        if to_call >= int(0.38 * hero_stack) or eff_bb <= 12.5:
            if pre_raises >= 2 and p_v_lag < 0.45:
                # Facing a 3-bet/4-bet shove from non-LAG -> require premium
                if pct >= 0.95 and hero_eq_vs_v >= req_eq + 0.03:
                    return ("call", to_call) if to_call >= hero_stack else ("raise", hero_stack)
                return ("fold", 0)
            if to_call < hero_stack and hero_eq_vs_v >= req_eq + 0.11 and pct >= 0.88:
                return ("raise", hero_stack)
            safety = 0.035 if len(active_opps) == 1 else 0.065
            if hero_eq_vs_v >= req_eq + safety and pct >= max(0.66, v_min_pct - 0.08):
                return ("call", to_call)
            return ("fold", 0)

        # 2b. Elastic 3-Bet Value & Re-Steal Engine (`pre_raises == 1`)
        if pre_raises == 1:
            # Compute optimal elastic 3-bet target that makes Villain face `to_call > 4.0 * bb`
            # (and `>= 0.22 * v_stack` when Villain is medium-stacked <= 25 BB)
            v_rem_stack = stacks[primary_villain]
            v_cur_bet = street_bets[primary_villain]
            elastic_v_call = int(4.05 * bb) + 1
            if v_rem_stack <= 25 * bb:
                elastic_v_call = max(elastic_v_call, int(0.22 * v_rem_stack) + 2)
            elastic_3b_total = v_cur_bet + elastic_v_call
            elastic_3b_add = max(min_raise_add, elastic_3b_total - street_bets[hero_idx])

            val_3bet_pct = max(0.85, min(0.96, v_min_pct + 0.14))
            if pct >= val_3bet_pct and hero_eq_vs_v >= 0.58 and bf <= 1.42:
                if eff_bb <= 18.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, elastic_3b_add))

            # High-EV Elastic 3-Bet Re-Steal against CO/BTN/SB small opens (`<= 2.55x BB`)
            if (
                is_v_late
                and raise_total_bb <= 2.55
                and len(active_opps) <= 2
                and bf <= 1.30
                and self.fold_to_3bet[primary_villain] >= 0.72
                and p_v_station < 0.25
                and p_v_lag < 0.35
                and elastic_3b_add <= int(0.36 * hero_stack)
                and (
                    pct >= 0.74
                    or (has_ace_blocker and pct >= 0.58)
                    or (is_suited and pct >= 0.64)
                )
            ):
                return ("raise", min(hero_stack, elastic_3b_add))
        else:
            # Facing a 3-bet or 4-bet: only 4-bet/5-bet with super-premiums (`QQ+, AKs`)
            if pct >= 0.975 and hero_eq_vs_v >= req_eq + 0.05:
                if eff_bb <= 28.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, max(min_raise_add, int(2.4 * to_call))))
            # Fold 3-bet bluffs when 4-bet!
            if pre_raises >= 3 and pct < 0.96:
                return ("fold", 0)

        # 2c. Strict Flat-Call Defense
        in_bb_hu = (hero_idx == bb_seat) and (len(active_opps) == 1) and (pre_raises == 1)
        if in_bb_hu:
            if (
                to_call <= int(1.65 * bb)
                and bf <= 1.35
                and hero_eq_vs_v >= req_eq + 0.025
                and (pct >= max(0.54, v_min_pct - 0.14) or is_pair or (is_suited and pct >= 0.46))
            ):
                return ("call", to_call)
            if (
                to_call <= int(2.5 * bb)
                and bf <= 1.30
                and hero_eq_vs_v >= req_eq + 0.045
                and pct >= max(0.72, v_min_pct - 0.04)
            ):
                return ("call", to_call)
            return ("fold", 0)

        if pre_raises == 1 and bf <= 1.32 and to_call <= int(0.20 * hero_stack):
            if is_pair and hero_stack >= 12 * to_call and pct >= 0.58:
                return ("call", to_call)
            if pct >= max(0.78, v_min_pct + 0.04) and hero_eq_vs_v >= req_eq + 0.06:
                return ("call", to_call)

        return ("fold", 0)

    def _act_postflop(
        self, state: Dict[str, Any], active_opps: List[int]
    ) -> Tuple[str, int]:
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        street_bets = state["street_bets"]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = max(1, state["big_blind"])
        min_raise_add = state["min_raise_add"]
        street = state["street"]
        hand_num = state.get("hand_number", 1)
        h0, h1 = state["hole_cards"]
        board = state["board"]
        n_opp = len(active_opps)

        hand_contrib = state.get("hand_contributions", street_bets)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]

        primary_villain = max(
            active_opps,
            key=lambda o: (street_bets[o], hand_contrib[o], total_stacks[o]),
        )
        v_post = self.posterior[primary_villain]
        p_v_nit = v_post[ARCH_NIT]
        p_v_tag = v_post[ARCH_TAG]
        p_v_lag = v_post[ARCH_LAG]
        p_v_station = v_post[ARCH_STATION]

        made_cat, wetness, has_draw, is_tptk, has_nut_blocker = analyze_board_texture(
            (h0, h1), board
        )

        pot_before_bet = max(bb, pot - to_call)
        bet_ratio = to_call / pot_before_bet if to_call > 0 else 0.0

        v_min_pct, eff_bluff, eff_fold, v_raised_street = (
            self._infer_villain_range_and_postflop_params(
                state, primary_villain, bet_ratio, wetness
            )
        )

        if n_opp >= 2:
            for o in active_opps:
                if o != primary_villain:
                    o_min, _, _, _ = self._infer_villain_range_and_postflop_params(
                        state, o, bet_ratio, wetness
                    )
                    if o_min > v_min_pct:
                        v_min_pct = o_min

        n_samples = 60 if n_opp == 1 else 48
        eq_cur, eq_sm, fold_sm, eq_bg, fold_bg, eq_jm, fold_jm = (
            conditional_postflop_profile(
                hole=(h0, h1),
                board=board,
                num_opponents=n_opp,
                num_samples=n_samples,
                villain_min_pct=v_min_pct,
                villain_bet_ratio=bet_ratio,
                villain_bluff_prob=eff_bluff,
                villain_fold_tendency=eff_fold,
            )
        )

        playing_board_river = False
        if street == 3 and len(board) == 5:
            b_sc = eval_5(board[0], board[1], board[2], board[3], board[4])
            h_sc = eval_7(h0, h1, board[0], board[1], board[2], board[3], board[4])
            if h_sc <= b_sc:
                playing_board_river = True

        # =====================================================================
        # BRANCH A: CHECKED TO HERO (`to_call == 0`)
        # =====================================================================
        if to_call == 0:
            if all(self.posterior[o][ARCH_AFK] > 0.78 for o in active_opps):
                return ("raise", min(hero_stack, max(min_raise_add, bb)))

            if (
                street == 1
                and n_opp == 1
                and made_cat >= 3
                and wetness <= 0.26
                and p_v_lag >= 0.45
            ):
                return ("check", 0)

            if made_cat >= 2 or (is_tptk and eq_cur >= 0.62) or eq_cur >= 0.68:
                inelastic_weight = p_v_station + 0.75 * p_v_lag
                elastic_weight = p_v_nit + 0.75 * p_v_tag

                if made_cat >= 2 or eq_cur >= 0.78:
                    if inelastic_weight >= 0.42:
                        val_frac = 0.88 if street < 3 else 0.96
                    elif elastic_weight >= 0.52:
                        val_frac = 0.60 if wetness <= 0.40 else 0.64
                    else:
                        val_frac = 0.68 + 0.12 * wetness
                else:
                    if p_v_station >= 0.40:
                        val_frac = 0.72
                    elif p_v_nit >= 0.40:
                        val_frac = 0.44
                    else:
                        val_frac = 0.54

                target_bet = max(min_raise_add, int(val_frac * pot) + 1)
                if target_bet >= int(0.78 * hero_stack) and (made_cat >= 2 or eq_cur >= 0.76):
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, target_bet))

            if (
                street == 3
                and n_opp == 1
                and made_cat == 1
                and eq_cur >= 0.57
                and p_v_station >= 0.35
                and not playing_board_river
            ):
                thin_bet = max(min_raise_add, int(0.55 * pot))
                return ("raise", min(hero_stack, thin_bet))

            if n_opp == 1 and p_v_station < 0.35:
                bf_bluff, _ = compute_bubble_factor(
                    total_stacks, hero_idx, primary_villain, int(0.42 * pot), pot, hand_num
                )
                if street == 1 and bf_bluff <= 1.32:
                    if (p_v_nit >= 0.35 or eff_fold >= 0.58) and (
                        eq_cur >= 0.36 or has_draw or wetness <= 0.42
                    ):
                        cbet = max(min_raise_add, int(0.355 * pot) + 1)
                        return ("raise", min(hero_stack, cbet))
                    if wetness <= 0.38 and (eq_cur >= 0.40 or has_draw or has_nut_blocker):
                        cbet = max(min_raise_add, int(0.43 * pot) + 1)
                        return ("raise", min(hero_stack, cbet))

                if (
                    street == 2
                    and bf_bluff <= 1.25
                    and eff_fold >= 0.48
                    and (has_draw or has_nut_blocker or eq_cur >= 0.48)
                    and eq_cur >= 0.36
                ):
                    probe = max(min_raise_add, int(0.44 * pot) + 1)
                    return ("raise", min(hero_stack, probe))

                if (
                    street == 3
                    and bf_bluff <= 1.20
                    and eff_fold >= 0.56
                    and (p_v_nit + p_v_tag) >= 0.60
                    and has_nut_blocker
                    and (made_cat == 0 or playing_board_river)
                    and fold_sm >= 0.52
                ):
                    r_bluff = max(min_raise_add, int(0.42 * pot))
                    if r_bluff <= int(0.35 * hero_stack):
                        return ("raise", min(hero_stack, r_bluff))

            return ("check", 0)

        # =====================================================================
        # BRANCH B: FACING A BET OR RAISE (`to_call > 0`)
        # =====================================================================
        if street == 3 and (made_cat == 0 or playing_board_river):
            return ("fold", 0)

        adj_eq = eq_cur
        if v_raised_street and made_cat <= 1:
            if street >= 2:
                adj_eq *= 0.32 + 0.52 * p_v_lag
            else:
                adj_eq *= 0.60 + 0.32 * p_v_lag
        elif p_v_lag < 0.45 and 0.30 <= bet_ratio <= 0.49 and made_cat >= 1:
            # Villain took their capped C-bet / Blocker-bluff / Turn-probe sizing branch (33-48% pot)!
            # They rarely hold Two Pair+ or TPTK in this sizing branch -> boost Hero's pair equity!
            adj_eq = min(0.92, adj_eq + 0.065 * p_v_tag)
        elif p_v_lag < 0.45 and 0.52 <= bet_ratio <= 0.72 and made_cat <= 1:
            # Villain took their uncapped value-bet branch (55-65% pot -> TPTK or Two Pair+)!
            dom_pen = (0.06 if is_tptk else 0.14) * (p_v_nit + p_v_tag + p_v_station)
            adj_eq = max(0.05, adj_eq - dom_pen)
        elif street >= 2 and made_cat <= 1:
            if (p_v_nit + p_v_station) >= 0.48 and bet_ratio >= 0.45:
                dom_pen = (0.10 if is_tptk else 0.18) * (p_v_nit + p_v_station)
                adj_eq = max(0.05, adj_eq - dom_pen)
            elif p_v_lag >= 0.40 and made_cat == 1:
                adj_eq = min(0.92, adj_eq + 0.065 * p_v_lag)

        bf, req_eq = compute_bubble_factor(
            total_stacks, hero_idx, primary_villain, to_call, pot, hand_num
        )

        if (
            (made_cat >= 3 and adj_eq >= 0.68)
            or (made_cat == 2 and adj_eq >= 0.74 and not v_raised_street)
            or adj_eq >= 0.82
        ):
            if bf <= 1.50 or adj_eq >= 0.84:
                if (p_v_station + p_v_lag) >= 0.42:
                    raise_target = to_call + int(0.88 * pot)
                else:
                    raise_target = to_call + int(0.62 * pot)
                if raise_target >= int(0.72 * hero_stack) and adj_eq >= 0.76:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, max(min_raise_add, raise_target)))

        if (
            street == 1
            and n_opp == 1
            and not v_raised_street
            and bet_ratio <= 0.40
            and has_draw
            and has_nut_blocker
            and adj_eq >= 0.48
            and eff_fold >= 0.48
            and p_v_station < 0.30
            and bf <= 1.22
            and hero_stack >= 4 * to_call
        ):
            cr_target = max(min_raise_add, to_call + int(0.62 * pot))
            return ("raise", min(hero_stack, cr_target))

        draw_bonus = 0.0
        if street < 3 and has_draw and hero_stack >= 2.5 * to_call:
            draw_bonus = 0.055 if street == 1 else 0.038
            if (p_v_station + p_v_nit) >= 0.45:
                draw_bonus += 0.02

        float_bonus = 0.0
        if (
            street == 1
            and n_opp == 1
            and bet_ratio <= 0.36
            and (made_cat >= 1 or has_draw or has_nut_blocker)
            and p_v_nit < 0.50
        ):
            float_bonus = 0.035

        if p_v_lag >= 0.42:
            margin = 0.015 + 0.01 * street
        elif (p_v_nit + p_v_station) >= 0.52:
            margin = 0.055 + 0.028 * street
        else:
            margin = 0.042 + 0.022 * street

        if adj_eq + draw_bonus + float_bonus >= req_eq + margin:
            return ("call", to_call)

        return ("fold", 0)
