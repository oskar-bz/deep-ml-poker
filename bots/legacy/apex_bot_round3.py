"""
ApexBot — Unified Championship AI for the Deep-ML No-Limit Pokerbot Tournament.

Synthesizes the winning breakthroughs from 7 independent rival architectures:
1. 5-Hypothesis Bayesian Opponent Archetype Posterior (`{AFK, NIT, TAG, LAG, STATION}`)
   with decoupled positional VPIP/PFR likelihoods, open-sizing signatures, check-fold
   tracking, and Showdown Hand Calibration (`showdown_hands`) in O(1) decision time.
2. Precomputed O(1) Domination-Adjusted Preflop Range Equity Table (`PREFLOP_EQ_VS_RANGE`)
   across 11 opponent range floors (`0.00..0.97`) + Unbiased 3-Bet/4-Bet/All-In Tiering
   (fixing the multiway set-mining percentile inflation of 66-88 vs AKo/AQs).
3. O(1) Board-Relative True Hand Classifier (`_classify_true_hand`) across all streets
   (eliminating paired-board One-Pair / Two-Pair / Trips illusions).
4. Dynamic Multi-Street Villain Range Narrowing & Bayesian Bet-Size Conditioning
   (distinguishing capped 33-48% C-bet/probe branches from uncapped 52-72% value branches
   and postflop raises).
5. Capped-Range Turn Takeover & Anti-Spew Barrel Gating + `2*bb + 1` Steal Sizing +
   Exact Contestable Side-Pot Stripping + Asymmetric ICM Bubble Warfare.
"""

from __future__ import annotations
import math
from time import perf_counter_ns as _perf_ns
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

# Prior probabilities over (AFK, NIT, TAG, LAG, STATION), normalized so max log == 0.0
PRIOR_PROBS: Tuple[float, ...] = (0.06, 0.22, 0.37, 0.18, 0.17)
_MAX_PRIOR_P = max(PRIOR_PROBS)
PRIOR_LOGS: Tuple[float, ...] = tuple(math.log(p / _MAX_PRIOR_P) for p in PRIOR_PROBS)

# Log-likelihood tables per archetype: (AFK, NIT, TAG, LAG, STATION)
_L_VPIP_EARLY_YES = tuple(math.log(p) for p in (0.025, 0.14, 0.24, 0.54, 0.44))
_L_VPIP_EARLY_NO = tuple(math.log(1.0 - p) for p in (0.025, 0.14, 0.24, 0.54, 0.44))

_L_VPIP_STEAL_YES = tuple(math.log(p) for p in (0.025, 0.18, 0.44, 0.62, 0.44))
_L_VPIP_STEAL_NO = tuple(math.log(1.0 - p) for p in (0.025, 0.18, 0.44, 0.62, 0.44))

_L_PFR_GIVEN_VPIP = tuple(math.log(p) for p in (0.95, 0.45, 0.84, 0.85, 0.09))
_L_CALL_GIVEN_VPIP = tuple(math.log(1.0 - p) for p in (0.95, 0.45, 0.84, 0.85, 0.09))

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
    """Analytical Hold'em preflop equity of canonical (r1 >= r2, s1) vs (r3 >= r4, s2)."""
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


def _classify_true_hand(
    h0: int,
    h1: int,
    board: Sequence[int],
    raw_made_cat: int,
    raw_is_tptk: bool,
) -> Tuple[int, bool]:
    """
    O(1) Board-Relative True Hand Classifier (~1.0 us).
    Eliminates the paired-board illusion where an underpair (e.g. 44 on Q-Q-8) or
    unpaired high-cards on a paired/double-paired/trips board are misclassified
    as Two Pair or Trips by raw 7-card evaluation.
    """
    if raw_made_cat == 0:
        return 0, False

    hr0, hr1 = h0 >> 2, h1 >> 2
    hi, lo = (hr0, hr1) if hr0 >= hr1 else (hr1, hr0)
    is_pocket_pair = hr0 == hr1

    b_counts = [0] * 13
    max_b = 0
    for c in board:
        r = c >> 2
        b_counts[r] += 1
        if r > max_b:
            max_b = r

    max_b_cnt = max(b_counts)
    if max_b_cnt == 1:
        return raw_made_cat, raw_is_tptk

    n_b_pairs = sum(1 for cnt in b_counts if cnt >= 2)

    if max_b_cnt == 2 and n_b_pairs == 1:
        paired_r = next(r for r in range(13) if b_counts[r] == 2)
        if hr0 == paired_r or hr1 == paired_r:
            return raw_made_cat, True
        if raw_made_cat >= 4:
            if raw_made_cat == 6 and is_pocket_pair and hi < max_b:
                return 3, False
            return raw_made_cat, True
        if is_pocket_pair:
            return (2, True) if hi > max_b else (1, False)
        unpaired_max = max((r for r in range(13) if b_counts[r] == 1), default=0)
        if hr0 == unpaired_max or hr1 == unpaired_max:
            other_kicker = lo if hi == unpaired_max else hi
            return 1, (unpaired_max >= max_b and other_kicker >= 9)
        if b_counts[hr0] > 0 or b_counts[hr1] > 0:
            return 1, False
        return 0, False

    if max_b_cnt >= 3:
        trips_r = next(r for r in range(13) if b_counts[r] >= 3)
        if hr0 == trips_r or hr1 == trips_r:
            return 7, True
        if is_pocket_pair and hi >= max_b:
            return 6, True
        if b_counts[hr0] > 0 or b_counts[hr1] > 0:
            matched_r = hr0 if (b_counts[hr0] > 0 and hr0 != trips_r) else hr1
            return (6, True) if matched_r >= max_b else (2, False)
        if is_pocket_pair:
            return 1, (hi > max_b - 2)
        return 0, False

    # Double-paired board (2 pairs on board)
    paired_ranks = [r for r in range(13) if b_counts[r] == 2]
    top_pair_r = max(paired_ranks)
    if hr0 == top_pair_r or hr1 == top_pair_r:
        return 6, True
    if (hr0 in paired_ranks) or (hr1 in paired_ranks):
        return 3, True
    if is_pocket_pair and hi > max_b:
        return 2, True
    if is_pocket_pair and hi > min(paired_ranks):
        return 1, False
    if b_counts[hr0] > 0 or b_counts[hr1] > 0:
        return 1, False
    return 0, False


def _unbiased_allin_pct(h0: int, h1: int) -> float:
    """
    Corrects `PREFLOP_PERCENTILE` for 3-bet / 4-bet / all-in decisions so medium
    pocket pairs (66-88) do not outrank AKo/AQs/AQo due to multiway set-mining bonuses.
    """
    pct = PREFLOP_PERCENTILE[h0][h1]
    r0, r1 = h0 >> 2, h1 >> 2
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    is_pair = r0 == r1
    is_suited = (h0 & 3) == (h1 & 3)
    if is_pair:
        if hi >= 11:  # AA, KK
            return 0.995
        if hi == 10:  # QQ
            return 0.985
        if hi == 9:   # JJ
            return 0.968
        if hi == 8:   # TT
            return 0.948
        if hi == 7:   # 99
            return 0.915
        if hi == 6:   # 88
            return 0.885
        if hi == 5:   # 77
            return 0.855
        if hi == 4:   # 66
            return 0.825
        return max(0.62, 0.62 + 0.045 * hi)
    if hi == 12 and lo == 11:  # AKs / AKo
        return 0.980 if is_suited else 0.962
    if hi == 12 and lo == 10:  # AQs / AQo
        return 0.952 if is_suited else 0.925
    if hi == 12 and lo == 9:   # AJs / AJo
        return 0.918 if is_suited else 0.865
    if hi == 11 and lo == 10 and is_suited:  # KQs
        return 0.905
    return pct


class ApexBot:
    """
    Unified Championship Bot: 5-Hypothesis Bayesian Hand-Reader + Board-Relative
    True Category Engine + Capped-Range Turn Takeover + Asymmetric ICM Sniper.
    """

    name: str = "ApexBot"

    def __init__(self) -> None:
        self._reset_profiles()

    def _reset_profiles(self) -> None:
        self.log_post: List[List[float]] = [list(PRIOR_LOGS) for _ in range(8)]
        self.posterior: List[List[float]] = [list(PRIOR_PROBS) for _ in range(8)]
        self.never_afk: List[bool] = [False] * 8

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
        self.faced_jam: List[int] = [0] * 8     # faced an unopened-pot open-jam
        self.called_jam: List[int] = [0] * 8
        self._pending_jam = None                # (hand_number, seats behind) after our open-jam
        self._pending_rejam = None              # (hand_number, opener, opener's chips in) after our re-jam
        self.tb_deep: List[List[float]] = [[] for _ in range(8)]   # top% of observed deep 3-bet hands
        self.f4_n: List[int] = [0] * 8          # deep 3-bettor faced a 4-bet
        self.fourbet_count: List[int] = [0] * 8  # opener re-raised a 3-bet
        # hole-card bluff learning: [raise, bet<0.4, bet0.4-0.7, bet>=0.7] -> (n, air)
        self.hb_n: List[List[int]] = [[0, 0, 0, 0] for _ in range(8)]
        self.hb_air: List[List[int]] = [[0, 0, 0, 0] for _ in range(8)]
        self.tb4: List[List[float]] = [[] for _ in range(8)]       # top% of observed 4-bet hands
        self.f4_fold: List[int] = [0] * 8

        self.post_faced_bet: List[int] = [0] * 8
        self.post_folded_bet: List[int] = [0] * 8
        self.post_called_bet: List[int] = [0] * 8
        self.post_raised_bet: List[int] = [0] * 8
        self.post_checked_op: List[int] = [0] * 8
        self.post_bet_op: List[int] = [0] * 8
        self.check_faced_bet: List[int] = [0] * 8
        self.check_folded_bet: List[int] = [0] * 8

        self.sd_count: List[int] = [0] * 8
        self.sd_bluffs: List[int] = [0] * 8
        self.sd_light_calls: List[int] = [0] * 8
        self.sd_open_pct_sum: List[float] = [0.0] * 8
        self.sd_open_pct_n: List[int] = [0] * 8

        self.open_min_pct: List[float] = [0.68] * 8
        self.steal_min_pct: List[float] = [0.54] * 8
        self.call_min_pct: List[float] = [0.58] * 8
        self.threebet_min_pct: List[float] = [0.85] * 8
        self.bluff_prob: List[float] = [0.18] * 8
        self.fold_tendency: List[float] = [0.54] * 8
        self.fold_to_steal: List[float] = [0.76] * 8
        self.fold_to_3bet: List[float] = [0.80] * 8
        self.check_fold_rate: List[float] = [0.62] * 8

        self._last_recorded_hand: int = 0
        self._last_seen_hand: int = 0
        self._cached_summary: Dict[str, Any] = {}

        for s in range(8):
            self._recompute_seat_params(s)

    def on_table_start(self, seat_idx: int, starting_stack: int) -> None:
        self._reset_profiles()

    def _recompute_seat_params(self, s: int) -> None:
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

        p_f3bet = (
            0.84 * (post[ARCH_NIT] + post[ARCH_TAG])
            + 0.98 * post[ARCH_AFK]
            + 0.45 * post[ARCH_LAG]
            + 0.35 * post[ARCH_STATION]
        )
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
        self.check_fold_rate[s] = (self.check_folded_bet[s] + 2.6 * p_fold) / (
            self.check_faced_bet[s] + 2.6
        )

    def on_hand_end(self, hero_idx: int, hand_summary: Dict[str, Any]) -> None:
        hand_num = hand_summary.get("hand_number", 0)
        if hand_num <= self._last_recorded_hand:
            return
        self._last_recorded_hand = hand_num

        starting_stacks = hand_summary.get("starting_stacks", [])
        action_history = hand_summary.get("action_history", [])
        showdown_hands = hand_summary.get("showdown_hands", {})
        board = hand_summary.get("board", [])

        ah = hand_summary.get("all_hole") or {}
        hist0 = [x for x in action_history if x[0] == 0]
        if hist0 and starting_stacks:
            bbx = next((x[3] for x in hist0 if x[2] == "bb"), 0) or 20
            rs = [x for x in hist0 if x[2] == "raise"]
            if len(rs) >= 3 and rs[2][1] != hero_idx and rs[2][1] in ah:
                c0, c1 = ah[rs[2][1]]
                self.tb4[rs[2][1]].append(1.0 - PREFLOP_PERCENTILE[c0][c1])
            if len(rs) >= 2:
                op, t = rs[0][1], rs[1][1]
                if t != op and min(starting_stacks[t], starting_stacks[op]) >= self.P_X4_MIN_BB * bbx:
                    if t in ah and t != hero_idx:
                        c0, c1 = ah[t]
                        self.tb_deep[t].append(1.0 - PREFLOP_PERCENTILE[c0][c1])
                    if len(rs) >= 3 and rs[2][1] != t:
                        i3 = hist0.index(rs[2])
                        resp = next((x for x in hist0[i3 + 1:] if x[1] == t), None)
                        if resp is not None:
                            self.f4_n[t] += 1
                            self.f4_fold[t] += resp[2] == "fold"

        if self.P_HB_ON > 0 and ah and action_history:
            fb = hand_summary.get("board") or board
            pot_run = 0
            cur_st = -1
            street_bet = False
            for st, s_, act_t, amt in action_history:
                if st != cur_st:
                    cur_st = st; street_bet = False
                if st >= 1 and act_t == "raise" and s_ != hero_idx and s_ in ah and len(fb) >= 2 + st:
                    bd = list(fb[: 2 + st])
                    c0, c1 = ah[s_]
                    try:
                        rmc, _w, _d, rtp, _nb = analyze_board_texture((c0, c1), bd)
                        mc, _t = _classify_true_hand(c0, c1, bd, rmc, rtp)
                        if not street_bet:
                            frac = amt / max(1, pot_run)
                            k = 1 if frac < 0.4 else (2 if frac < 0.7 else 3)
                        else:
                            k = 0
                        self.hb_n[s_][k] += 1
                        self.hb_air[s_][k] += mc == 0
                    except Exception:
                        pass
                if act_t == "raise":
                    street_bet = True
                pot_run += max(0, amt)

        pr = self._pending_rejam
        if pr is not None and pr[0] == hand_num:
            self._pending_rejam = None
            dl = hand_summary.get("deltas")
            o = pr[1]
            hist_pf = [x for x in hand_summary.get("action_history", ()) if x[0] == 0]
            last_hero_raise = max((i for i, x in enumerate(hist_pf) if x[1] == hero_idx and x[2] == "raise"), default=-1)
            seen_resp = last_hero_raise >= 0 and any(x[1] == o for x in hist_pf[last_hero_raise + 1:])
            if dl and not seen_resp:                      # full history already counts it otherwise
                o_in = pr[2]
                self.faced_3bet[o] += 1
                if abs(dl[o]) <= o_in + 1:
                    self.folded_3bet[o] += 1
                self._recompute_seat_params(o)

        pj = self._pending_jam
        if pj is not None and pj[0] == hand_num:
            self._pending_jam = None
            deltas = hand_summary.get("deltas")
            bbj = hand_summary.get("bb") or 0
            if deltas and bbj > 0:
                # caller = big chip swing; folders lose at most their blind
                for o in pj[1]:
                    self.faced_jam[o] += 1
                    if abs(deltas[o]) > 1.5 * bbj:
                        self.called_jam[o] += 1
            elif showdown_hands:                          # fallback: shown cards (needs a showdown)
                for o in pj[1]:
                    self.faced_jam[o] += 1
                    if o in showdown_hands:
                        self.called_jam[o] += 1

        if not starting_stacks or not action_history:
            return

        bb_amt = 20
        bb_seat = -1
        for st, s, act_t, amt in action_history:
            if st == 0 and act_t == "bb" and amt > 0:
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
        fourbet = [False] * 8
        had_pre_decision = [False] * 8

        post_bets = [0] * 8
        post_checks = [0] * 8
        post_folds = [0] * 8
        post_calls = [0] * 8
        post_raises = [0] * 8
        late_street_aggr = [False] * 8
        late_street_call = [False] * 8
        checked_this_street = [False] * 8
        check_faced_this_street = [False] * 8

        cur_street = -1
        cur_high_bet = 0
        seat_street_bet = [0] * 8
        pre_raise_count = 0
        first_raiser = -1
        open_jammer = -1
        jam_seen = [False] * 8

        for st, s, act_t, amt in action_history:
            if st != cur_street:
                cur_street = st
                cur_high_bet = 0
                for i in range(8):
                    seat_street_bet[i] = 0
                    checked_this_street[i] = False
                    check_faced_this_street[i] = False

            if act_t in ("sb", "bb"):
                seat_street_bet[s] += amt
                if seat_street_bet[s] > cur_high_bet:
                    cur_high_bet = seat_street_bet[s]
                continue

            facing_bet = cur_high_bet > seat_street_bet[s]
            if st == 0 and open_jammer >= 0 and open_jammer != hero_idx and s != open_jammer and not jam_seen[s]:
                jam_seen[s] = True
                self.faced_jam[s] += 1
                if act_t in ("call", "raise"):
                    self.called_jam[s] += 1
            if st == 0:
                had_pre_decision[s] = True
                if pre_raise_count == 0:
                    was_unopened_op[s] = True
                elif pre_raise_count == 1 and facing_bet:
                    faced_steal[s] = True
                    if act_t == "fold":
                        folded_steal[s] = True
                elif pre_raise_count >= 2 and s == first_raiser and facing_bet:
                    if not faced_3bet[s] and act_t == "raise":
                        fourbet[s] = True
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
                        if seat_street_bet[s] >= starting_stacks[s]:
                            open_jammer = s
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
                        checked_this_street[s] = True
                    elif act_t == "raise":
                        post_bets[s] += 1
                        self.never_afk[s] = True
                        seat_street_bet[s] += amt
                        if seat_street_bet[s] > cur_high_bet:
                            cur_high_bet = seat_street_bet[s]
                        if st >= 2:
                            late_street_aggr[s] = True
                else:
                    if checked_this_street[s] and not check_faced_this_street[s]:
                        check_faced_this_street[s] = True
                        self.check_faced_bet[s] += 1
                        if act_t == "fold":
                            self.check_folded_bet[s] += 1

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

        alive_in_hand = [i for i in range(8) if starting_stacks[i] > 0]
        n_alive = len(alive_in_hand)

        for s in range(8):
            if s == hero_idx or starting_stacks[s] <= 0:
                continue

            self.hands_dealt[s] += 1
            logs = self.log_post[s]

            for k in range(5):
                logs[k] = PRIOR_LOGS[k] + 0.985 * (logs[k] - PRIOR_LOGS[k])

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
                l_vpip = (
                    _L_VPIP_STEAL_YES
                    if (was_unopened_op[s] and is_lp_steal_spot)
                    else _L_VPIP_EARLY_YES
                )
                for k in range(5):
                    logs[k] += l_vpip[k]
                if pfr[s]:
                    self.pfr_count[s] += 1
                    for k in range(5):
                        logs[k] += _L_PFR_GIVEN_VPIP[k]
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
                l_no = (
                    _L_VPIP_STEAL_NO
                    if (was_unopened_op[s] and is_lp_steal_spot)
                    else _L_VPIP_EARLY_NO
                )
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
                if fourbet[s]:
                    self.fourbet_count[s] += 1

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
                base_o = (
                    self.steal_min_pct[villain]
                    if is_v_late
                    else (self.open_min_pct[villain] + pos_adj)
                )
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

        if bet_ratio > 0.05 and not v_raised_this_street and p_lag < 0.45:
            if 0.30 <= bet_ratio <= 0.49:
                eff_bluff = max(eff_bluff, 0.34 * p_tag + 0.15)
            elif 0.52 <= bet_ratio <= 0.72:
                eff_bluff *= 0.28

        if v_raised_this_street:
            eff_bluff *= 0.30 + 0.55 * p_lag
            eff_fold *= 0.55
        elif v_post_bets >= 2 and cur_street >= 2:
            eff_bluff *= 0.55 + 0.45 * p_lag

        if bet_ratio >= 0.65 and (p_nit + p_station) >= 0.50:
            eff_bluff *= 0.65

        if self.P_HB_ON > 0:
            if v_raised_this_street:
                k = 0
            elif bet_ratio > 0.05:
                k = 1 if bet_ratio < 0.4 else (2 if bet_ratio < 0.7 else 3)
            else:
                k = -1
            if k >= 0 and self.hb_n[villain][k] >= self.P_HB_MIN:
                n_ = self.hb_n[villain][k]
                eff_bluff = (self.hb_air[villain][k] + self.P_HB_K * eff_bluff) / (n_ + self.P_HB_K)

        eff_bluff = max(0.015, min(0.55, eff_bluff))
        eff_fold = max(0.06, min(0.96, eff_fold))
        return v_min_pct, eff_bluff, eff_fold, v_raised_this_street

    # Monte Carlo stops sampling this long after the decision started (1 ms hard limit)
    MC_BUDGET_NS: int = 300_000

    # Tunable strategy parameters (defaults = original hand-tuned values)
    P_OPEN_BASE: float = 0.44          # unopened-pot open threshold (percentile) ...
    P_OPEN_PER_BEHIND: float = 0.065   # ... plus this per player left to act
    P_OPEN_SIZE_BB: float = 2.62       # open size with strong hands (bb)
    P_PUSH_ZONE_BB: float = 12.0       # effective stack (bb) at/below which we push/fold
    P_PUSH_MULT: float = 1.0           # scales Nash push thresholds
    P_PUSH_MIN_PCT: float = 0.46       # minimum percentile to shove
    P_CALL_SAFETY_HU: float = 0.035    # extra equity required to call an all-in heads-up
    P_CALL_SAFETY_MW: float = 0.065    # ... multiway
    P_CALL_MIN_ALLIN_PCT: float = 0.66 # minimum unbiased all-in percentile to call off
    P_3BET_MIN: float = 0.86           # minimum value 3-bet percentile
    P_CALL_MARGIN_ADD: float = 0.0     # added to postflop calling margin
    P_VALUE_MULT: float = 1.0          # scales postflop value-bet sizes
    P_VALUE_TPTK_EQ: float = 0.62      # equity to value-bet top pair top kicker
    P_VALUE_EQ: float = 0.68           # equity to value-bet anything
    P_CBET_FOLD_GATE: float = 0.58     # villain fold tendency needed for small c-bet bluffs
    P_ICM_SCALE: float = 1.0           # 0 = chip-EV, 1 = full ICM bubble factors
    P_RESTEAL_ON: float = 0.0          # 1 = re-steal vs wide openers that over-fold to 3-bets
    P_RS_PRIOR: float = 0.80           # prior fold-to-3bet (real entrants: 0.77-0.92)
    P_RS_PRIOR_N: float = 4.0          # prior weight in observations
    P_RS_MIN_PFR: float = 0.25         # villain open frequency to target
    P_RS_MIN_PCT: float = 0.25         # weakest hand percentile to re-steal with
    P_RS_SIZE: float = 3.2             # re-raise to this multiple of the open
    P_W3_ON: float = 0.0               # 1 = defend vs empirically wide 3-bettors / re-jammers
    P_W3_MIN_N: float = 8.0            # faced-open observations before trusting the frequency
    P_W3_MIN_F: float = 0.10           # 3-bet frequency that counts as "wide"
    P_W3_SPREAD: float = 1.25          # range = top (spread * f) (>1: real ranges are not pure top-f)
    P_W3_JAM_BB: float = 40.0          # 4-bet jam vs wide non-all-in 3-bets up to this depth
    P_W3_JAM_MARGIN: float = 0.02      # equity margin over required for the 4-bet jam
    P_STAB_ON: float = 0.0             # 1 = EV small stab when checked to vs empirical over-folders
    P_STAB_MIN_N: float = 4.0          # villain postflop faced-bet observations needed
    P_STAB_FRAC: float = 0.33          # stab size (fraction of pot)
    P_STAB_MARGIN: float = 0.05        # required EV gain over checking (fraction of pot)
    P_XR_ON: float = 0.0               # 1 = EV re-jam over a single open vs openers who over-fold
    P_XR_MAX_BB: float = 25.0          # max effective stack (bb)
    P_XR_MIN_BB: float = 8.0           # below this push/fold logic is fine as is
    P_XR_PRIOR: float = 0.80           # prior fold-to-3bet of the opener (round-1 vs jams: 0.87)
    P_XR_N: float = 6.0                # prior weight
    P_XR_MARGIN: float = 0.5           # required EV gain over the baseline action, in bb
    P_X4_ON: float = 0.0               # 1 = EV 4-bet vs deep 3-bettors whose shown 3-bet hands are light
    P_X4_MIN_BB: float = 25.0          # min effective stack (bb)
    P_X4_MIN_OBS: float = 4.0          # observed 3-bet hands (hole cards at hand end) needed
    P_X4_CONT: float = 0.09            # assumed continue-vs-4bet range (top fraction of all hands)
    P_X4_K: float = 4.0                # prior weight vs observed fold-to-4bet
    P_X4_JAM_BB: float = 45.0          # jam the 4-bet up to this depth, else raise 2.3x
    P_X4_MARGIN: float = 1.0           # required EV, in bb
    P_X4_MIN_PRIOR: float = 0.55       # only vs clearly light 3-bettors (median shown 3-bet hand weaker than top ~10%)
    P_D3_ON: float = 0.0               # 1 = equity-based defence vs non-all-in 3-bets when deep
    P_D3_MIN_BB: float = 25.0
    P_D3_EQ_CALL: float = 0.50         # equity vs estimated 3-bet range to call
    P_D3_EQ_4B: float = 0.60           # ... to 4-bet for value
    P_D3_MAX_CALL: float = 0.25        # max call as a fraction of our stack
    P_D3_GATE: float = 0.0             # 1 = only vs clearly light 3-bettors (shown median or frequency)
    P_D3_GATE_MED: float = 0.12        # shown deep 3-bet hands median weaker than top 12%
    P_D3_GATE_F3: float = 0.20         # ... or 3-bet frequency >= 20%
    P_W4_ON: float = 0.0               # 1 = adaptive defence vs 4-bets (call / 5-bet jam by equity vs observed range)
    P_W4_PRIOR: float = 0.06           # prior 4-bet frequency when facing a 3-bet
    P_W4_K: float = 4.0
    P_W4_SPREAD: float = 1.3
    P_W4_SAFETY: float = 0.03          # equity margin over the ICM-required equity
    P_HB_ON: float = 0.0               # 1 = learn each opponent's postflop bluff rate from hole cards at hand end
    P_HB_MIN: float = 2.0              # observations of that action type before it is used
    P_HB_K: float = 3.0                # prior weight (prior = current model estimate)
    P_RIVX_ON: float = 0.0             # 1 = exact (enumerated) heads-up river profile instead of Monte Carlo
    P_XJ_ON: float = 0.0               # 1 = EV open-jam vs players who rarely call jams
    P_XJ_MAX_BB: float = 20.0          # max effective stack (bb) for the exploit jam
    P_XJ_PRIOR: float = 0.12           # prior call-vs-open-jam rate (round-1 median ~0.08)
    P_XJ_N: float = 6.0                # prior weight (observations)
    P_XJ_MARGIN: float = 0.35          # required EV gain over folding, in bb

    def _bubble(self, stacks, hero_idx, villain_idx, risk_chips, reward_chips=None, hand_number=1):
        """compute_bubble_factor with the ICM pressure scaled by P_ICM_SCALE."""
        bf, req = compute_bubble_factor(stacks, hero_idx, villain_idx, risk_chips, reward_chips, hand_number)
        k = self.P_ICM_SCALE
        if k == 1.0 or risk_chips <= 0:
            return bf, req
        risk = min(float(stacks[hero_idx]), float(risk_chips))
        reward = risk if reward_chips is None else reward_chips
        cev = risk / max(1e-9, risk + reward)
        return 1.0 + k * (bf - 1.0), max(0.01, min(0.98, cev + k * (req - cev)))

    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        self._t0 = state.get("_t0") or _perf_ns()
        to_call = state["to_call"]
        hero_idx = state["hero_idx"]
        stacks = state["stacks"]
        hero_stack = stacks[hero_idx]
        if hero_stack <= 0:
            return ("check", 0)

        hand_num = state.get("hand_number", 1)
        if hand_num != self._last_seen_hand:
            if self._last_seen_hand > self._last_recorded_hand and self._cached_summary:
                self.on_hand_end(hero_idx, self._cached_summary)
            self._last_seen_hand = hand_num
        self._cached_summary = {
            "hand_number": hand_num,
            "starting_stacks": state.get("starting_stacks", stacks),
            "action_history": state.get("action_history", ()),
            "board": state.get("board", ()),
        }

        active = state["active_seats"]
        active_opps = [i for i in range(8) if i != hero_idx and active[i]]
        if not active_opps:
            return ("check", 0)

        if state["street"] == 0:
            res = self._act_preflop(state, active_opps)
            if (res[0] == "raise" and res[1] >= hero_stack and max(state["street_bets"]) <= state["big_blind"]):
                self._pending_jam = (hand_num, list(self._last_behind))
            if self.P_W4_ON > 0 and res[0] == "fold":
                w4 = self._w4_try(state, hero_idx, hero_stack, hand_num)
                if w4 is not None:
                    return w4
            if self.P_D3_ON > 0 and res[0] == "fold":
                d3 = self._d3_try(state, hero_idx, hero_stack, hand_num)
                if d3 is not None:
                    return d3
            if self.P_X4_ON > 0 and res[0] in ("fold", "call"):
                x4 = self._x4_try(state, hero_idx, hero_stack, hand_num)
                if x4 is not None:
                    return x4
            if self.P_XR_ON > 0 and res[0] in ("fold", "call", "check"):
                xr = self._xr_try(state, hero_idx, hero_stack, hand_num)
                if xr is not None:
                    return xr
            return res
        return self._act_postflop(state, active_opps)

    def _profile(self, **kw):
        """Postflop equity / continuation profile (Monte Carlo). ApexPlus may override (exact river)."""
        return conditional_postflop_profile(**kw)

    def _xjam_ok(self, h0, h1, opps_behind, stacks, street_bets, hero_idx, total_stacks, pot, bb, hand_num) -> bool:
        """Open-jam EV (vs folding) using each player's observed call-vs-jam rate; caller range = top c."""
        if self.P_XJ_ON <= 0 or not opps_behind:
            return False
        risk = stacks[hero_idx]
        if risk > self.P_XJ_MAX_BB * bb:
            return False
        p_fold_all = 1.0
        ev_called = 0.0
        w_tot = 0.0
        for o in opps_behind:
            c = (self.called_jam[o] + self.P_XJ_PRIOR * self.P_XJ_N) / (self.faced_jam[o] + self.P_XJ_N)
            c = max(0.02, min(0.85, c))
            p_fold_all *= 1.0 - c
            m = min(risk, stacks[o] + street_bets[o] - street_bets[hero_idx])
            if m <= 0:
                continue
            eq = lookup_preflop_eq_vs_range(h0, h1, max(0.15, min(0.97, 1.0 - c)))
            bf, _ = self._bubble(total_stacks, hero_idx, o, m, pot + m, hand_num)
            ev_called += c * (eq * (pot + m) - (1.0 - eq) * m * bf)
            w_tot += c
        if w_tot <= 0:
            return False
        ev = p_fold_all * pot + (1.0 - p_fold_all) * (ev_called / w_tot)
        return ev >= self.P_XJ_MARGIN * bb

    def _w4_try(self, state, hero_idx, hero_stack, hand_num):
        """We 3-bet, the opener 4-bet, nobody else in: call / 5-bet jam if equity vs their 4-bet range pays."""
        stacks = state["stacks"]; sb = state["street_bets"]; bb = max(1, state["big_blind"])
        rs = []; calls = 0
        for st, s_, act_t, _amt in state.get("action_history", ()):
            if st != 0:
                continue
            if act_t == "raise":
                rs.append(s_)
            elif act_t == "call" and rs:
                calls += 1
        if len(rs) != 3 or rs[1] != hero_idx or rs[2] != rs[0] or calls:
            return None
        v = rs[2]
        h0, h1 = state["hole_cards"]
        if _unbiased_allin_pct(h0, h1) < 0.55:
            return None
        f4 = (self.fourbet_count[v] + self.P_W4_PRIOR * self.P_W4_K) / (self.faced_3bet[v] + self.P_W4_K)
        v_min = max(0.50, min(0.97, 1.0 - self.P_W4_SPREAD * f4))
        obs = self.tb4[v]
        if len(obs) >= 2:
            srt = sorted(obs)
            v_min = max(0.30, min(0.97, 1.0 - 2.0 * srt[len(srt) // 2]))
        eq = lookup_preflop_eq_vs_range(h0, h1, v_min)
        hero_tot = hero_stack + sb[hero_idx]; v_tot = stacks[v] + sb[v]
        m_tot = min(hero_tot, v_tot)
        risk = m_tot - sb[hero_idx]
        if risk <= 0:
            return None
        pot = state["pot"]
        # chips we can win if it goes to showdown: everything in the pot that we match, plus villain's rest up to m_tot
        reward = pot - max(0, sb[v] - m_tot) + max(0, m_tot - sb[v])
        hand_contrib = state.get("hand_contributions", sb)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        _bf, req = self._bubble(total_stacks, hero_idx, v, risk, reward, hand_num)
        if eq < req + self.P_W4_SAFETY:
            return None
        if stacks[v] <= 0 or risk >= hero_stack:
            return ("call", min(hero_stack, sb[v] - sb[hero_idx]))
        return ("raise", hero_stack)

    def _d3_try(self, state, hero_idx, hero_stack, hand_num):
        """We opened, one player 3-bet (not all-in), nobody else in, deep: call / 4-bet by equity vs their range."""
        stacks = state["stacks"]; sb = state["street_bets"]; bb = max(1, state["big_blind"])
        rs = []; calls = 0
        for st, s_, act_t, _amt in state.get("action_history", ()):
            if st != 0:
                continue
            if act_t == "raise":
                rs.append(s_)
            elif act_t == "call" and rs:
                calls += 1
        if len(rs) != 2 or rs[0] != hero_idx or calls:
            return None
        v = rs[1]
        if stacks[v] <= 0:
            return None
        hero_tot = hero_stack + sb[hero_idx]; v_tot = stacks[v] + sb[v]
        eff = min(hero_tot, v_tot) / bb
        if eff < self.P_D3_MIN_BB:
            return None
        to_call = min(hero_stack, sb[v] - sb[hero_idx])
        h0, h1 = state["hole_cards"]
        # villain 3-bet range: model estimate, widened by observed frequency and shown hands
        v_min = max(0.48, min(0.95, self.threebet_min_pct[v]))
        n_fo = self.faced_steal[v]
        if n_fo >= self.P_W3_MIN_N:
            f3 = (self.threebet_count[v] + 0.06 * 6.0) / (n_fo + 6.0)
            v_min = min(v_min, max(0.30, 1.0 - self.P_W3_SPREAD * f3))
        obs = self.tb_deep[v]
        f3_emp = (self.threebet_count[v] + 0.06 * 6.0) / (n_fo + 6.0) if n_fo else 0.0
        if len(obs) >= 2:
            # shown deep 3-bet hands override the frequency estimate in both directions
            srt = sorted(obs)
            med = srt[len(srt) // 2]
            if self.P_D3_GATE > 0 and med < self.P_D3_GATE_MED and f3_emp < self.P_D3_GATE_F3:
                return None      # a normal 3-bettor: keep the base (fold-heavy) response
            v_min = max(0.30, min(0.97, 1.0 - 2.0 * med))
        elif v_min < 0.80:
            return None          # wide only by mixed-depth frequency: unproven when deep
        elif self.P_D3_GATE > 0 and f3_emp < self.P_D3_GATE_F3:
            return None
        eq = lookup_preflop_eq_vs_range(h0, h1, v_min)
        hand_contrib = state.get("hand_contributions", sb)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        bf, _ = self._bubble(total_stacks, hero_idx, v, to_call, state["pot"], hand_num)
        if bf > 1.35:
            return None
        if eq >= self.P_D3_EQ_4B and _unbiased_allin_pct(h0, h1) >= 0.90:
            if eff <= 45.0:
                return ("raise", hero_stack)
            add = int(2.3 * sb[v]) - sb[hero_idx]
            if 0 < add < 0.45 * hero_stack:
                return ("raise", max(state["min_raise_add"], add))
        if eq >= self.P_D3_EQ_CALL and to_call <= self.P_D3_MAX_CALL * hero_stack:
            return ("call", to_call)
        return None

    def _x4_try(self, state, hero_idx, hero_stack, hand_num):
        """We opened, one player 3-bet (not all-in), nobody else in: 4-bet if their shown deep 3-bets are light."""
        stacks = state["stacks"]; sb = state["street_bets"]; bb = max(1, state["big_blind"])
        rs = []; calls = 0
        for st, s_, act_t, _amt in state.get("action_history", ()):
            if st != 0:
                continue
            if act_t == "raise":
                rs.append(s_)
            elif act_t == "call" and rs:
                calls += 1
        if len(rs) != 2 or rs[0] != hero_idx or calls:
            return None
        v = rs[1]
        if stacks[v] <= 0:
            return None
        obs = self.tb_deep[v]
        if len(obs) < self.P_X4_MIN_OBS:
            return None
        hero_tot = hero_stack + sb[hero_idx]; v_tot = stacks[v] + sb[v]
        eff = min(hero_tot, v_tot) / bb
        if eff < self.P_X4_MIN_BB:
            return None
        srt = sorted(obs)
        med = srt[len(srt) // 2]
        width = min(1.0, 2.0 * med)
        prior_f = max(0.05, min(0.85, 1.0 - self.P_X4_CONT / max(1e-6, width)))
        if prior_f < self.P_X4_MIN_PRIOR:
            return None
        F = (self.f4_fold[v] + self.P_X4_K * prior_f) / (self.f4_n[v] + self.P_X4_K)
        pot = state["pot"]
        h0, h1 = state["hole_cards"]
        hand_contrib = state.get("hand_contributions", sb)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        if eff <= self.P_X4_JAM_BB:
            m = min(hero_tot, v_tot) - sb[hero_idx]
            eq = lookup_preflop_eq_vs_range(h0, h1, 1.0 - self.P_X4_CONT)
            bfv, _ = self._bubble(total_stacks, hero_idx, v, m, pot + m, hand_num)
            ev = F * pot + (1.0 - F) * (eq * (pot + m) - (1.0 - eq) * m * bfv)
            if ev >= self.P_X4_MARGIN * bb:
                return ("raise", hero_stack)
            return None
        tgt = int(2.3 * sb[v])
        add = tgt - sb[hero_idx]
        if add <= 0 or add >= 0.45 * hero_stack:
            return None
        ev = F * pot - (1.0 - F) * add
        if ev >= self.P_X4_MARGIN * bb:
            return ("raise", max(state["min_raise_add"], add))
        return None

    def _xr_try(self, state, hero_idx, hero_stack, hand_num):
        """Re-jam over exactly one (non-all-in) open with no callers, by EV vs observed fold rates."""
        stacks = state["stacks"]; sb = state["street_bets"]; bb = max(1, state["big_blind"])
        raisers = []; callers_after = 0
        for st, s_, act_t, _amt in state.get("action_history", ()):
            if st != 0:
                continue
            if act_t == "raise":
                raisers.append(s_)
            elif act_t == "call" and raisers:
                callers_after += 1
        if len(raisers) != 1 or callers_after or raisers[0] == hero_idx:
            return None
        o = raisers[0]
        if stacks[o] <= 0:
            return None
        hero_tot = hero_stack + sb[hero_idx]
        o_tot = stacks[o] + sb[o]
        eff = min(hero_tot, o_tot) / bb
        if eff > self.P_XR_MAX_BB or eff < self.P_XR_MIN_BB:
            return None
        h0, h1 = state["hole_cards"]
        hand_contrib = state.get("hand_contributions", sb)
        total_stacks = [stacks[i] + hand_contrib[i] for i in range(8)]
        pot = state["pot"]
        f_o = (self.folded_3bet[o] + self.P_XR_PRIOR * self.P_XR_N) / (self.faced_3bet[o] + self.P_XR_N)
        cands = [(o, max(0.02, min(0.95, 1.0 - f_o)))]
        for b in getattr(self, "_last_behind", []):
            if b == o or not state["active_seats"][b]:
                continue
            c = (self.called_jam[b] + self.P_XJ_PRIOR * self.P_XJ_N) / (self.faced_jam[b] + self.P_XJ_N)
            cands.append((b, max(0.02, min(0.85, c))))
        p_fold_all = 1.0; ev_called = 0.0; w_tot = 0.0
        for v, c in cands:
            p_fold_all *= 1.0 - c
            m = min(hero_tot, stacks[v] + sb[v]) - sb[hero_idx]
            if m <= 0:
                continue
            eq = lookup_preflop_eq_vs_range(h0, h1, max(0.15, min(0.97, 1.0 - c)))
            bfv, _ = self._bubble(total_stacks, hero_idx, v, m, pot + m, hand_num)
            ev_called += c * (eq * (pot + m) - (1.0 - eq) * m * bfv)
            w_tot += c
        if w_tot <= 0:
            return None
        ev = p_fold_all * pot + (1.0 - p_fold_all) * (ev_called / w_tot)
        # baseline: folding = 0; calling is not modelled -> require margin either way
        if ev < self.P_XR_MARGIN * bb:
            return None
        self._pending_rejam = (hand_num, o, sb[o])
        return ("raise", hero_stack)

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
        allin_pct = _unbiased_allin_pct(h0, h1)
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
        hero_already_raised = False
        for st, s, act_t, _amt in action_history:
            if st == 0:
                if act_t == "raise":
                    pre_raises += 1
                    last_raiser = s
                    if s == hero_idx:
                        hero_already_raised = True
                elif act_t == "call":
                    limpers += 1

        # Robustness: if the harness provides no (or an incomplete) action history, infer the
        # preflop raise structure from street-bet levels instead of treating raised pots as 4-bets.
        max_bet_now = max(street_bets)
        if pre_raises == 0 and max_bet_now > bb:
            levels = sorted({b for b in street_bets if b > bb})
            pre_raises = max(1, len(levels))
            hero_already_raised = street_bets[hero_idx] > bb
            cands = [o for o in active_opps if street_bets[o] == max_bet_now]
            last_raiser = cands[0] if cands else -1
            limpers = 0
        elif pre_raises == 0:
            limpers = sum(
                1 for o in active_opps if o != bb_seat and street_bets[o] >= bb
            )

        opps_behind: List[int] = []
        self._last_behind = opps_behind
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

        # Exact contestable pot & effective call (stripping uncalled excess chips from covering stacks)
        eff_to_call = min(hero_stack, to_call)
        hero_match_bet = street_bets[hero_idx] + eff_to_call
        uncalled_excess = sum(max(0, street_bets[o] - hero_match_bet) for o in active_opps)
        eff_pot = max(bb, pot - uncalled_excess)

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

            # Ensure steal total bet is strictly >= 2 * bb + 1 so BB faces to_call > bb!
            min_steal_total = max(2 * bb + 1, int(2.05 * bb))
            steal_add = max(min_raise_add, min_steal_total - street_bets[hero_idx])

            if all_afk and len(opps_behind) <= 3 and limpers == 0:
                if steal_add <= hero_stack:
                    return ("raise", steal_add)

            bf, _ = self._bubble(
                total_stacks, hero_idx, primary_villain, min(hero_stack, int(2.2 * bb)), eff_pot, hand_num
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
                    return ("raise", min(hero_stack, steal_add))
                if avg_fold_steal >= 0.68 and len(opps_behind) <= 2 and pct >= 0.32:
                    return ("raise", min(hero_stack, steal_add))

            # Short-Stack Nash Push/Fold
            if eff_bb <= self.P_PUSH_ZONE_BB:
                shove_bf, _ = self._bubble(
                    total_stacks, hero_idx, primary_villain, eff_chips, eff_pot, hand_num
                )
                fold_scale = 0.85 + 0.45 * p_all_fold
                pos_div = (1.0 + 0.12 * max(0, len(opps_behind) - 1)) * (shove_bf ** 1.15)
                req_push_bb = (NASH_PUSH_BB[h0][h1] * fold_scale * self.P_PUSH_MULT) / max(0.5, pos_div)
                if eff_bb <= req_push_bb and (pct >= self.P_PUSH_MIN_PCT or hu_eq >= 0.54):
                    return ("raise", hero_stack)
                if limpers == 0 and self._xjam_ok(h0, h1, opps_behind, stacks, street_bets, hero_idx,
                                                  total_stacks, pot, bb, hand_num):
                    return ("raise", hero_stack)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            base_open = self.P_OPEN_BASE + self.P_OPEN_PER_BEHIND * len(opps_behind) + (bf - 1.08) * 0.16
            if any_station_or_lag:
                base_open += 0.05

            if limpers >= 1:
                if pct >= max(0.72, base_open + 0.08):
                    iso_add = max(
                        min_raise_add,
                        int((2.65 + 0.8 * limpers) * bb) - street_bets[hero_idx],
                    )
                    return ("raise", min(hero_stack, iso_add))
                if to_call <= bb and (PREFLOP_STRENGTH[h0][h1] >= 0.60 or (h0 >> 2) == (h1 >> 2)):
                    return ("call", to_call)
                return ("check", 0) if to_call == 0 else ("fold", 0)

            if pct >= base_open:
                if pct >= 0.66 and eff_bb >= 15.0:
                    open_target = int(self.P_OPEN_SIZE_BB * bb) + 1
                else:
                    open_target = min_steal_total
                open_add = max(min_raise_add, open_target - street_bets[hero_idx])
                return ("raise", min(hero_stack, open_add))

            if (limpers == 0 and eff_bb <= self.P_XJ_MAX_BB
                    and self._xjam_ok(h0, h1, opps_behind, stacks, street_bets, hero_idx,
                                      total_stacks, pot, bb, hand_num)):
                return ("raise", hero_stack)
            return ("check", 0) if to_call == 0 else ("fold", 0)

        # CASE 2: Facing a Raise, 3-Bet, or All-In Shove (`pre_raises >= 1`)
        v_post = self.posterior[primary_villain]
        p_v_afk = v_post[ARCH_AFK]
        p_v_lag = v_post[ARCH_LAG]
        p_v_station = v_post[ARCH_STATION]

        v_dist = (button - primary_villain) & 7
        is_v_late = (v_dist <= 2) or (primary_villain == sb_seat)
        pos_adj = -0.05 if is_v_late else (0.025 * (v_dist - 2.5) if primary_villain != bb_seat else 0.0)
        raise_total_bb = max(street_bets) / bb

        if p_v_afk > 0.65 and raise_total_bb >= 8.0:
            v_min_pct = 0.975
        elif pre_raises == 1:
            base_o = (
                self.steal_min_pct[primary_villain]
                if (is_v_late and raise_total_bb <= 2.4)
                else (self.open_min_pct[primary_villain] + pos_adj)
            )
            v_min_pct = max(0.18, min(0.93, base_o))
            if raise_total_bb >= 6.0:
                v_min_pct = max(0.42, min(0.95, v_min_pct + 0.06 * (1.0 - p_v_lag)))
        elif pre_raises == 2:
            v_min_pct = max(0.48, min(0.95, self.threebet_min_pct[primary_villain] + 0.4 * pos_adj))
        else:
            v_min_pct = 0.93 if p_v_lag < 0.45 else 0.78

        wide3 = False
        if self.P_W3_ON > 0 and pre_raises == 2 and hero_already_raised:
            n_fo = self.faced_steal[primary_villain]
            if n_fo >= self.P_W3_MIN_N:
                f3 = (self.threebet_count[primary_villain] + 0.06 * 6.0) / (n_fo + 6.0)
                if f3 >= self.P_W3_MIN_F:
                    wide3 = True
                    emp_min = max(0.30, min(0.95, 1.0 - self.P_W3_SPREAD * f3))
                    v_min_pct = min(v_min_pct, emp_min)

        hero_eq_vs_v = lookup_preflop_eq_vs_range(h0, h1, v_min_pct)
        bf, req_eq = self._bubble(
            total_stacks, hero_idx, primary_villain, eff_to_call, eff_pot, hand_num
        )

        r0, r1 = h0 >> 2, h1 >> 2
        hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
        is_pair = r0 == r1
        is_suited = (h0 & 3) == (h1 & 3)
        has_ace_blocker = (hi == 12) and (is_suited or lo >= 8)
        is_suited_connector = is_suited and (hi - lo <= 2) and (4 <= hi <= 9)

        # 2a. Facing an All-In or committing raise (`eff_to_call >= 0.38 * hero_stack` or `eff_bb <= 12.5`)
        if eff_to_call >= int(0.38 * hero_stack) or eff_bb <= self.P_PUSH_ZONE_BB + 0.5:
            if pre_raises >= 2 and p_v_lag < 0.45 and not wide3:
                if allin_pct >= 0.948 and hero_eq_vs_v >= req_eq + 0.03:
                    return ("call", to_call) if to_call >= hero_stack else ("raise", hero_stack)
                return ("fold", 0)
            if to_call < hero_stack and hero_eq_vs_v >= req_eq + 0.11 and allin_pct >= 0.88:
                return ("raise", hero_stack)
            safety = self.P_CALL_SAFETY_HU if len(active_opps) == 1 else self.P_CALL_SAFETY_MW
            if hero_eq_vs_v >= req_eq + safety and allin_pct >= max(self.P_CALL_MIN_ALLIN_PCT, v_min_pct - 0.08):
                return ("call", to_call)
            return ("fold", 0)

        # 2b. Elastic 3-Bet Value & Re-Steal Engine (`pre_raises == 1`)
        if pre_raises == 1 and not hero_already_raised:
            v_rem_stack = stacks[primary_villain]
            v_cur_bet = street_bets[primary_villain]
            elastic_v_call = int(4.05 * bb) + 1
            if v_rem_stack <= 25 * bb:
                elastic_v_call = max(elastic_v_call, int(0.22 * v_rem_stack) + 2)
            elastic_3b_total = v_cur_bet + elastic_v_call
            elastic_3b_add = max(min_raise_add, elastic_3b_total - street_bets[hero_idx])

            val_3bet_pct = max(self.P_3BET_MIN, min(0.96, v_min_pct + 0.14))
            if allin_pct >= val_3bet_pct and hero_eq_vs_v >= 0.58 and bf <= 1.42:
                if eff_bb <= 18.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, elastic_3b_add))

            # Re-steal vs wide openers that over-fold to re-raises. Demo replays of the
            # real entrants: most open 35-90% of pots and fold 77-92% to a 3-bet.
            if self.P_RESTEAL_ON > 0 and raise_total_bb <= 3.6 and eff_bb >= 15.0:
                v = primary_villain
                n_seen = self.hands_dealt[v]
                v_pfr = self.pfr_count[v] / max(1, n_seen)
                fold_est = (self.folded_3bet[v] + self.P_RS_PRIOR_N * self.P_RS_PRIOR) / (
                    self.faced_3bet[v] + self.P_RS_PRIOR_N)
                others = len(active_opps) - 1  # players still to act behind / limpers
                if n_seen >= 6 and v_pfr >= self.P_RS_MIN_PFR and others <= 2 and bf <= 1.5:
                    rs_total = int(self.P_RS_SIZE * v_cur_bet)
                    rs_add = max(min_raise_add, rs_total - street_bets[hero_idx])
                    if rs_add <= int(0.30 * hero_stack):
                        p_fold = fold_est * (0.85 ** others)
                        loss_frac = 0.45 if (pct >= 0.55 or is_pair or is_suited) else 0.65
                        ev = p_fold * eff_pot - (1.0 - p_fold) * rs_add * loss_frac
                        if pct >= self.P_RS_MIN_PCT and ev > 0.25 * bb:
                            return ("raise", min(hero_stack, rs_add))

            # High-EV Elastic 3-Bet Re-Steal against CO/BTN/SB small opens (`<= 2.55x BB`)
            if (
                is_v_late
                and raise_total_bb <= 2.55
                and len(active_opps) <= 2
                and bf <= 1.30
                and self.fold_to_3bet[primary_villain] >= 0.72
                and p_v_station < 0.25
                and p_v_lag < 0.35
                and elastic_3b_add <= int(0.35 * hero_stack)
                and (
                    pct >= 0.74
                    or (has_ace_blocker and pct >= 0.58)
                    or is_suited_connector
                    or (is_suited and pct >= 0.64)
                )
            ):
                return ("raise", min(hero_stack, elastic_3b_add))
        else:
            # Facing a 3-bet or 4-bet: use unbiased allin_pct so AKo/QQ+ 4-bet while 66-88 do not!
            if allin_pct >= 0.960 and hero_eq_vs_v >= req_eq + 0.04:
                if eff_bb <= 28.0:
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, max(min_raise_add, int(2.4 * to_call))))
            if pre_raises >= 3 and allin_pct < 0.960:
                return ("fold", 0)
            if wide3 and eff_bb <= self.P_W3_JAM_BB and to_call < hero_stack:
                v_all = stacks[primary_villain] + street_bets[primary_villain]
                risk = min(hero_stack, v_all - street_bets[hero_idx])
                _bfj, req_j = self._bubble(total_stacks, hero_idx, primary_villain, risk, eff_pot, hand_num)
                if (hero_eq_vs_v >= req_j + self.P_W3_JAM_MARGIN
                        and allin_pct >= max(self.P_CALL_MIN_ALLIN_PCT, v_min_pct - 0.08)):
                    return ("raise", hero_stack)
            # If we opened and face a reasonable 3-bet, defend with strong playable hands
            if (
                hero_already_raised
                and pre_raises == 2
                and eff_to_call <= int(3.5 * bb)
                and eff_to_call <= int(0.22 * hero_stack)
                and bf <= 1.32
                and (pct >= 0.80 or (is_pair and hi >= 5))
            ):
                return ("call", to_call)

        # 2c. Strict Flat-Call Defense
        in_bb_hu = (hero_idx == bb_seat) and (len(active_opps) == 1) and (pre_raises == 1)
        if in_bb_hu:
            if (
                eff_to_call <= int(1.65 * bb)
                and bf <= 1.35
                and hero_eq_vs_v >= req_eq + 0.025
                and (pct >= max(0.54, v_min_pct - 0.14) or is_pair or (is_suited and pct >= 0.46))
            ):
                return ("call", to_call)
            if (
                eff_to_call <= int(2.5 * bb)
                and bf <= 1.30
                and hero_eq_vs_v >= req_eq + 0.045
                and pct >= max(0.72, v_min_pct - 0.04)
            ):
                return ("call", to_call)
            return ("fold", 0)

        if pre_raises == 1 and bf <= 1.32 and eff_to_call <= int(0.20 * hero_stack):
            if is_pair and hero_stack >= 12 * eff_to_call and pct >= 0.58:
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

        raw_made_cat, wetness, has_draw, raw_is_tptk, has_nut_blocker = analyze_board_texture(
            (h0, h1), board
        )
        made_cat, is_tptk = _classify_true_hand(h0, h1, board, raw_made_cat, raw_is_tptk)

        eff_to_call = min(hero_stack, to_call)
        hero_match_bet = street_bets[hero_idx] + eff_to_call
        uncalled_excess = sum(max(0, street_bets[o] - hero_match_bet) for o in active_opps)
        eff_pot = max(bb, pot - uncalled_excess)

        pot_before_bet = max(bb, eff_pot - eff_to_call)
        bet_ratio = eff_to_call / pot_before_bet if eff_to_call > 0 else 0.0

        # Track street-level action history for Capped-Range Turn Takeover & Anti-Spew Gating
        action_history = state.get("action_history", ())
        flop_had_bet = False
        turn_had_bet = False
        hero_called_flop_bet = False
        v_called_prev_street = False
        v_checked_this_street = False
        for st, s, act_t, _amt in action_history:
            if st == 1:
                if act_t == "raise":
                    flop_had_bet = True
                elif act_t == "call":
                    if s == hero_idx and flop_had_bet:
                        hero_called_flop_bet = True
                    elif s == primary_villain:
                        v_called_prev_street = True
                elif st == street and act_t == "check" and s == primary_villain:
                    v_checked_this_street = True
            elif st == 2:
                if act_t == "raise":
                    turn_had_bet = True
                elif act_t == "call" and s == primary_villain:
                    v_called_prev_street = True
                elif st == street and act_t == "check" and s == primary_villain:
                    v_checked_this_street = True
            elif st == 3 and act_t == "check" and s == primary_villain:
                v_checked_this_street = True

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

        # Fast-exit on River with pure air / board-only hand facing a bet
        if street == 3 and to_call > 0 and made_cat == 0:
            return ("fold", 0)

        n_samples = 56 if n_opp == 1 else 44
        eq_cur, eq_sm, fold_sm, eq_bg, fold_bg, eq_jm, fold_jm = (
            self._profile(
                hole=(h0, h1),
                board=board,
                num_opponents=n_opp,
                num_samples=n_samples,
                villain_min_pct=v_min_pct,
                villain_bet_ratio=bet_ratio,
                villain_bluff_prob=eff_bluff,
                villain_fold_tendency=eff_fold,
                deadline_ns=self._t0 + self.MC_BUDGET_NS,
            )
        )

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

            if made_cat >= 2 or (is_tptk and eq_cur >= self.P_VALUE_TPTK_EQ) or eq_cur >= self.P_VALUE_EQ:
                inelastic_weight = p_v_station + 0.75 * p_v_lag
                elastic_weight = p_v_nit + 0.75 * p_v_tag

                if made_cat >= 2 or eq_cur >= 0.78:
                    if inelastic_weight >= 0.42:
                        val_frac = 0.88 if street < 3 else 0.96
                    elif street == 1 and n_opp == 1 and made_cat >= 3 and wetness <= 0.36:
                        # Induce check-raise bluffs (<=0.42x) and floats (<=0.38x) when holding Set/Trips+
                        val_frac = 0.36
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

                target_bet = max(min_raise_add, int(val_frac * self.P_VALUE_MULT * eff_pot) + 1)
                if target_bet >= int(0.78 * hero_stack) and (made_cat >= 2 or eq_cur >= 0.76):
                    return ("raise", hero_stack)
                return ("raise", min(hero_stack, target_bet))

            if (
                street == 3
                and n_opp == 1
                and made_cat == 1
                and eq_cur >= 0.57
                and p_v_station >= 0.35
            ):
                thin_bet = max(min_raise_add, int(0.55 * eff_pot))
                return ("raise", min(hero_stack, thin_bet))

            if self.P_STAB_ON > 0 and n_opp == 1 and made_cat < 2:
                nfb = self.post_faced_bet[primary_villain]
                if nfb >= self.P_STAB_MIN_N:
                    f_emp = (self.post_folded_bet[primary_villain] + 3.0 * 0.50) / (nfb + 3.0)
                    stab = max(min_raise_add, int(self.P_STAB_FRAC * eff_pot) + 1)
                    if stab < 0.5 * hero_stack:
                        bf_s, _ = self._bubble(total_stacks, hero_idx, primary_villain, stab, eff_pot, hand_num)
                        # called: assume our equity drops (caller range is stronger) and we realize it poorly
                        eq_c = max(0.0, eq_cur - 0.12)
                        ev_stab = f_emp * eff_pot + (1.0 - f_emp) * (eq_c * (eff_pot + 2 * stab) - stab * bf_s)
                        ev_check = eq_cur * eff_pot * 0.85
                        if ev_stab - ev_check >= self.P_STAB_MARGIN * eff_pot:
                            return ("raise", min(hero_stack, stab))

            if n_opp == 1 and p_v_station < 0.35:
                bf_bluff, _ = self._bubble(
                    total_stacks, hero_idx, primary_villain, int(0.42 * eff_pot), eff_pot, hand_num
                )
                if street == 1 and bf_bluff <= 1.32:
                    if (p_v_nit >= 0.35 or eff_fold >= self.P_CBET_FOLD_GATE) and (
                        eq_cur >= 0.36 or has_draw or wetness <= 0.42
                    ):
                        cbet = max(min_raise_add, int(0.355 * eff_pot) + 1)
                        return ("raise", min(hero_stack, cbet))
                    if wetness <= 0.38 and (eq_cur >= 0.40 or has_draw or has_nut_blocker):
                        # 43% pot + 1 bypasses <= 38% floats and <= 42% flop check-raises!
                        cbet = max(min_raise_add, int(0.43 * eff_pot) + 1)
                        return ("raise", min(hero_stack, cbet))

                if street == 2 and bf_bluff <= 1.26:
                    # 1) Capped-Range Turn Takeover (we floated Villain's Flop C-bet IP and Villain checked Turn)
                    if (
                        hero_called_flop_bet
                        and v_checked_this_street
                        and p_v_nit < 0.42
                        and self.check_fold_rate[primary_villain] >= 0.52
                        and (made_cat >= 1 or has_draw or has_nut_blocker or eq_cur >= 0.32)
                    ):
                        takeover = max(min_raise_add, int(0.48 * eff_pot) + 1)
                        return ("raise", min(hero_stack, takeover))

                    # 2) Delayed Turn Probe (when Flop checked through) or Semi-Bluff Barrel
                    if (
                        eff_fold >= 0.48
                        and (not v_called_prev_street or has_draw or made_cat >= 1)
                        and (
                            has_draw
                            or has_nut_blocker
                            or eq_cur >= 0.46
                            or (not flop_had_bet and v_checked_this_street and eq_cur >= 0.36)
                        )
                        and eq_cur >= 0.34
                    ):
                        probe = max(min_raise_add, int(0.44 * eff_pot) + 1)
                        return ("raise", min(hero_stack, probe))

                if (
                    street == 3
                    and bf_bluff <= 1.20
                    and eff_fold >= 0.56
                    and (p_v_nit + p_v_tag) >= 0.58
                    and made_cat == 0
                    and (has_nut_blocker or (not turn_had_bet and v_checked_this_street))
                    and fold_sm >= 0.52
                ):
                    r_bluff = max(min_raise_add, int(0.44 * eff_pot))
                    if r_bluff <= int(0.35 * hero_stack):
                        return ("raise", min(hero_stack, r_bluff))

            return ("check", 0)

        # =====================================================================
        # BRANCH B: FACING A BET OR RAISE (`to_call > 0`)
        # =====================================================================
        adj_eq = eq_cur
        if v_raised_street and made_cat <= 1:
            if street >= 2:
                adj_eq *= 0.32 + 0.52 * p_v_lag
            else:
                adj_eq *= 0.60 + 0.32 * p_v_lag
        elif p_v_lag < 0.45 and 0.30 <= bet_ratio <= 0.49 and made_cat >= 1:
            adj_eq = min(0.92, adj_eq + 0.065 * p_v_tag)
        elif p_v_lag < 0.45 and 0.52 <= bet_ratio <= 0.72 and made_cat <= 1:
            dom_pen = (0.06 if is_tptk else 0.14) * (p_v_nit + p_v_tag + p_v_station)
            adj_eq = max(0.05, adj_eq - dom_pen)
        elif street >= 2 and made_cat <= 1:
            if (p_v_nit + p_v_station) >= 0.48 and bet_ratio >= 0.45:
                dom_pen = (0.10 if is_tptk else 0.18) * (p_v_nit + p_v_station)
                adj_eq = max(0.05, adj_eq - dom_pen)
            elif p_v_lag >= 0.40 and made_cat == 1:
                adj_eq = min(0.92, adj_eq + 0.065 * p_v_lag)

        bf, req_eq = self._bubble(
            total_stacks, hero_idx, primary_villain, eff_to_call, eff_pot, hand_num
        )

        if (
            (made_cat >= 3 and adj_eq >= 0.68)
            or (made_cat == 2 and adj_eq >= 0.74 and not v_raised_street)
            or adj_eq >= 0.82
        ):
            if bf <= 1.50 or adj_eq >= 0.84:
                if (p_v_station + p_v_lag) >= 0.42:
                    raise_target = eff_to_call + int(0.88 * eff_pot)
                else:
                    raise_target = eff_to_call + int(0.62 * eff_pot)
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
            and hero_stack >= 4 * eff_to_call
        ):
            cr_target = max(min_raise_add, eff_to_call + int(0.62 * eff_pot))
            return ("raise", min(hero_stack, cr_target))

        draw_bonus = 0.0
        if street < 3 and has_draw and hero_stack >= 2.5 * eff_to_call:
            draw_bonus = 0.055 if street == 1 else 0.038
            if (p_v_station + p_v_nit) >= 0.45:
                draw_bonus += 0.02

        float_bonus = 0.0
        if (
            street == 1
            and n_opp == 1
            and bet_ratio <= 0.38
            and (made_cat >= 1 or has_draw or has_nut_blocker)
            and p_v_nit < 0.50
        ):
            float_bonus = 0.040

        if p_v_lag >= 0.42:
            margin = 0.015 + 0.01 * street
        elif (p_v_nit + p_v_station) >= 0.52:
            margin = 0.055 + 0.028 * street
        else:
            margin = 0.042 + 0.022 * street
        margin += self.P_CALL_MARGIN_ADD

        if adj_eq + draw_bonus + float_bonus >= req_eq + margin:
            return ("call", to_call)

        return ("fold", 0)
