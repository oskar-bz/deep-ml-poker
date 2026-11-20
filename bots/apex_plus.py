"""
ApexPlus: ApexBot + exact (enumerated) heads-up river equity profile.

The starter-bot best-response layer that used to live here (exact model of the public starter bot,
identified by bet-size signatures) never fired against the round-3 field and was removed; the old
version is kept in bots/legacy/apex_plus_starter_br.py.
"""

from __future__ import annotations

from time import perf_counter_ns as _perf_ns

import numpy as np

from bots.apex_bot import ApexBot
from poker_engine.evaluator import (
    CARD_PRIME,
    CARD_SUIT_BIT,
    CARD_SUIT_WEIGHT,
    FLUSH_SCORE,
    FLUSH_SUIT_SHIFT,
    PRIME_TO_SCORE,
)
from poker_engine.preflop_tables import PREFLOP_PERCENTILE
from poker_engine.range_engine import COMBO_C0, COMBO_C1, COMBO_INDEX, COMBO_MASK, combo_scores
from bots.fingerprints import FP_PROFILES, FP_SIGNATURES
from bots.tells import TELL_EDGES, TELLS
from bots.fp_core import Fingerprinter, features_from_history

_PCT_COMBO = np.array([PREFLOP_PERCENTILE[a][b] for a, b in zip(COMBO_C0.tolist(), COMBO_C1.tolist())])


# call-probability tables of conditional_postflop_profile (river, heads-up), by hand class index
_RIV_IDX = [0, 1, 2, 3, 4, 5, 6, 7]
_RIV_SM = np.array([0.04, 0.98, 0.98, 0.98, 0.98, 0.88, 0.62, 0.38, 0.04])
_RIV_BG = np.array([0.01, 0.92, 0.92, 0.70, 0.70, 0.58, 0.22, 0.08, 0.03])
_RIV_JM = np.array([0.002, 0.85, 0.52, 0.85, 0.52, 0.22, 0.04, 0.01, 0.005])


def river_exact_profile(hole, board, villain_min_pct=0.25, villain_bet_ratio=0.0, villain_bluff_prob=0.22,
                        villain_fold_tendency=0.50, **_ignored):
    """Exact heads-up river version of evaluator.conditional_postflop_profile: the same per-hand weights,
    averaged over every villain combo instead of sampled (the Monte Carlo's limit, zero variance)."""
    h0, h1 = hole[0], hole[1]
    sc, valid = combo_scores(board)
    valid = valid & ((COMBO_MASK & ((1 << h0) | (1 << h1))) == 0)
    h_score = int(sc[COMBO_INDEX[h0, h1]])
    max_r = 0; second_r = 0
    for b in board:
        r = b >> 2
        if r > max_r:
            second_r = max_r; max_r = r
        elif r > second_r:
            second_r = r
    b_sw = 0; b_p = 1; b_sb = 0
    for c in board:
        b_sw += CARD_SUIT_WEIGHT[c]; b_p *= CARD_PRIME[c]; b_sb |= CARD_SUIT_BIT[c]
    b4f = any(((b_sw >> (3 * s_)) & 7) >= 4 for s_ in range(4))
    b3f = any(((b_sw >> (3 * s_)) & 7) >= 3 for s_ in range(4))
    shift = FLUSH_SUIT_SHIFT[b_sw]
    bos = PRIME_TO_SCORE[b_p] if shift < 0 else FLUSH_SCORE[(b_sb >> shift) & 0x1FFF]
    bcat = bos >> 4
    tp = (1 << 20) | (max_r << 12)
    tptk = tp | (9 << 8)
    mid = (1 << 20) | (second_r << 12)
    v = sc[valid]
    vn_cat = v >> 20
    if villain_min_pct > 0.05:
        w = np.where(_PCT_COMBO[valid] >= villain_min_pct, 1.0, 0.03)
    else:
        w = np.ones(v.shape[0])
    is_pb = (v <= bos) | ((v >> 4) == bcat)
    if villain_bet_ratio > 0.05:
        air_w = max(0.02, min(0.60, villain_bluff_prob / (1.0 + 1.5 * villain_bet_ratio)))
        weak_w = max(0.12, min(0.75, 0.50 / (1.0 + 0.9 * villain_bet_ratio)))
        air = is_pb | (vn_cat == 0) | (b4f & (vn_cat < 5))
        weak = (~air) & (v < tp) & (vn_cat == 1)
        w = w * np.where(air, air_w, np.where(weak, weak_w, 1.0))
    res = (h_score > v).astype(np.float64) + 0.5 * (h_score == v)
    c1 = is_pb | (b4f & (vn_cat < 5))
    c2 = (vn_cat >= 2) | (v >= tptk)
    c3 = v >= tp
    c4 = v >= mid
    c5 = vn_cat == 1
    good_bg = (vn_cat >= 5) if b3f else np.ones(v.shape[0], dtype=bool)
    hi_jm = vn_cat >= 2
    idx = np.select([c1, c2 & good_bg & hi_jm, c2 & good_bg, c2 & hi_jm, c2, c3, c4, c5],
                    _RIV_IDX, 8)
    p_sm = _RIV_SM[idx]; p_bg = _RIV_BG[idx]; p_jm = _RIV_JM[idx]
    w_cur = float(w.sum()); win_cur = float((w * res).sum())
    ws = w * p_sm; wb = w * p_bg; wj = w * p_jm
    w_sm = float(ws.sum()); w_bg = float(wb.sum()); w_jm = float(wj.sum())
    eq_cur = win_cur / max(1e-9, w_cur)
    eq_sm = float((ws * res).sum()) / max(1e-9, w_sm)
    eq_bg = float((wb * res).sum()) / max(1e-9, w_bg)
    eq_jm = float((wj * res).sum()) / max(1e-9, w_jm)
    t = villain_fold_tendency
    fold_sm = max(0.05, min(0.92, (1.0 - w_sm / max(1e-9, w_cur)) * (0.55 + 0.90 * t)))
    fold_bg = max(0.10, min(0.96, (1.0 - w_bg / max(1e-9, w_cur)) * (0.65 + 0.70 * t)))
    fold_jm = max(0.15, min(0.98, (1.0 - w_jm / max(1e-9, w_cur)) * (0.75 + 0.50 * t)))
    return eq_cur, eq_sm, fold_sm, eq_bg, fold_bg, eq_jm, fold_jm


# warm numpy's code paths at import (the first call costs ~0.5 ms; startup time is free)
try:
    river_exact_profile((0, 1), (8, 13, 22, 31, 44), villain_min_pct=0.4, villain_bet_ratio=0.5)
except Exception:
    pass


class ApexPlus(ApexBot):
    """ApexBot with the exact heads-up river profile (switch: P_RIVX_ON)."""

    # only take the exact path if the decision is still young (platform runs ~2-3x slower than local)
    RIVX_START_NS: int = 50_000

    # ---------------------------------------------------------------- opponent fingerprinting
    P_FP_ON: float = 0.0     # 1 = identify known finalists by bet-size signatures and load their profiles
    P_FP_W: float = 12.0     # pseudo-observations injected per statistic on identification

    # numerator -> denominator of the counters a profile seeds
    _FP_PAIRS = (("folded_steal", "faced_steal"), ("threebet_count", "faced_steal"),
                 ("folded_3bet", "faced_3bet"), ("fourbet_count", "faced_3bet"),
                 ("called_jam", "faced_jam"), ("post_folded_bet", "post_faced_bet"),
                 ("vu_3b", "vu_faced"), ("f4_fold", "f4_n"), ("check_folded_bet", "check_faced_bet"),
                 ("folded_xr", "faced_xr"))

    # ---------------------------------------------------------------- bet-size tells
    P_TELL_N: float = 40.0   # prior weight (hands) of the population histogram in a bot's cell

    def _tell_equity(self, villain, street, bet_ratio, h0, h1, board):
        """Share of the villain's measured first bets of this size (this street) made with a hand weaker than
        ours right now (board percentile vs all hands; ties half). None if there is no data."""
        if street < 1 or len(board) < 3:
            return None
        b = 0
        while b < len(TELL_EDGES) and bet_ratio >= TELL_EDGES[b]:
            b += 1
        sk, bk = str(street), str(b)
        pop = TELLS.get("_pop", {}).get(sk, {}).get(bk)
        if not pop or sum(pop) < 20:
            return None
        who = self.fp_ident.get(villain) if self.P_FP_ON > 0 else None
        cell = TELLS.get(who, {}).get(sk, {}).get(bk) if who else None
        npop = float(sum(pop))
        hist = [self.P_TELL_N * x / npop for x in pop]
        if cell:
            hist = [a + c for a, c in zip(hist, cell)]
        sc, valid = combo_scores(board)
        sv = sc[valid]
        me = sc[COMBO_INDEX[h0, h1]]
        lo = int(np.count_nonzero(sv < me)); eq_n = int(np.count_nonzero(sv == me))
        pct = (lo + 0.5 * eq_n) / max(1, len(sv))
        tot = sum(hist)
        d = min(9, int(pct * 10))
        below = sum(hist[:d]) + hist[d] * (pct * 10 - d)
        return below / tot

    def _reset_profiles(self) -> None:
        super()._reset_profiles()
        self._fp = Fingerprinter(FP_SIGNATURES)
        self._fp_done = set()
        self._fp_last_hand = -1
        self.fp_ident = {}

    def _fp_inject(self, s: int, bot: str) -> None:
        prof = FP_PROFILES.get(bot)
        if not prof:
            return
        w = self.P_FP_W
        seeded = set()
        for num, den in self._FP_PAIRS:
            rate_n = prof.get(num)
            if not rate_n or rate_n[0] is None:
                continue
            if den not in seeded:
                getattr(self, den)[s] += w
                seeded.add(den)
            getattr(self, num)[s] += w * rate_n[0]
        self._recompute_seat_params(s)

    def on_hand_end(self, hero_idx: int, hand_summary: Dict[str, Any]) -> None:
        super().on_hand_end(hero_idx, hand_summary)
        if self.P_FP_ON <= 0:
            return
        hn = hand_summary.get("hand_number", 0)
        if hn <= self._fp_last_hand:
            return
        self._fp_last_hand = hn
        try:
            for s, feat, val in features_from_history(hand_summary.get("action_history", ()),
                                                      hand_summary.get("starting_stacks", ())):
                if s != hero_idx and s not in self._fp_done:
                    self._fp.observe(s, feat, val)
            for s in range(8):
                if s == hero_idx or s in self._fp_done:
                    continue
                who = self._fp.identify(s)
                if who is not None:
                    self._fp_done.add(s)
                    self.fp_ident[s] = who
                    self._fp_inject(s, who)
        except Exception:
            pass

    def _profile(self, **kw):
        if (self.P_RIVX_ON > 0 and len(kw.get("board", ())) == 5 and kw.get("num_opponents", 1) <= 1
                and _perf_ns() - self._t0 < self.RIVX_START_NS):
            try:
                return river_exact_profile(**kw)
            except Exception:
                pass
        return super()._profile(**kw)
