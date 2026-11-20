"""
ApexPlus: ApexBot + an exact model of the contest's starter-bot family.

The STARTER_BOT every entrant receives plays a fixed function of its made-hand
category (postflop) or preflop rank, with signature bet sizes:
    checked to:   bet  current_bet + pot // 2      if strength > 0.6   (two pair+)
    facing a bet: raise to 2 * min_raise           if strength > 0.75  (trips+)
                  call                             if strength > pot_odds + 0.05
    stack_m < 5:  all-in if strength > 0.5, else check/fold
    strength = 0.2 + 0.6 * preflop_rank()  |  {0: .25, 1: .45, 2: .65}.get(cat, .85)

Seats whose bet sizes keep matching this signature (and whose showdowns never
contradict it) are modelled exactly: each of their actions this hand restricts their
range to the combos consistent with it (vectorized over all 1326 combos), and our
postflop action is chosen by best response — the EV of fold / check / call / each bet
size against their deterministic replies, Monte Carlo over their range and the
runout. Everything else falls back to ApexBot.
"""

from __future__ import annotations

import random
from time import perf_counter_ns as _perf_counter_ns
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bots.apex_bot import ApexBot
from poker_engine.icm import compute_bubble_factor
from poker_engine.evaluator import (
    CARD_PRIME,
    CARD_SUIT_BIT,
    CARD_SUIT_WEIGHT,
    FLUSH_SCORE,
    FLUSH_SUIT_SHIFT,
    PRIME_TO_SCORE,
)
from poker_engine.preflop_tables import PREFLOP_PERCENTILE
from poker_engine.range_engine import combo_scores
from poker_engine.range_engine import (
    COMBO_C0,
    COMBO_C1,
    COMBO_INDEX,
    COMBO_MASK,
    PREFLOP_RANK,
    BoardCache,
    _FSC,
    _FSS,
    _KEYS,
    _NKEYS,
    _P,
    _SB,
    _SW,
    _VALS,
)

_STARTER_POST_STRENGTH = np.array([0.25, 0.45, 0.65, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])


def _eval_np(cols: Sequence[np.ndarray]) -> np.ndarray:
    """Vectorized 5..7 card evaluation; cols = list of int64 card arrays (same length)."""
    prod = _P[cols[0]].copy()
    sw = _SW[cols[0]].copy()
    sb = _SB[cols[0]].copy()
    for c in cols[1:]:
        prod *= _P[c]
        sw += _SW[c]
        sb |= _SB[c]
    idx = np.searchsorted(_KEYS, prod)
    np.minimum(idx, _NKEYS - 1, out=idx)
    score = _VALS[idx]
    sh = _FSS[sw]
    fl = sh >= 0
    if fl.any():
        bits = (sb >> np.where(fl, sh, 0)) & 0x1FFF
        score = np.where(fl, _FSC[bits], score)
    return score


class _Ctx:
    """Replayed context of one action in the hand."""

    __slots__ = ("street", "seat", "act", "add", "to_call", "pot", "stack", "my_bet",
                 "high", "min_raise_to", "n_opp", "total_after", "board_len")


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


class ApexPlus(ApexBot):
    name: str = "ApexPlus"

    # starter-family identification
    SIG_MIN_MATCH: int = 2
    # Tolerances: the real SDK's rank table / pot-odds rounding may differ slightly
    TOL_PF: float = 0.03   # preflop strength
    TOL_PO: float = 0.015  # pot-odds thresholds
    # Monte Carlo samples for the best-response engine
    BR_SAMPLES: int = 160
    BR_BUDGET_NS: int = 380_000

    def __init__(self) -> None:
        super().__init__()
        self._np_rng = np.random.default_rng(random.randrange(1 << 30))
        # Self-calibration to the host CPU: moving average of best-response time
        self._br_ema_ns = 250_000.0
        self._br_samples = self.BR_SAMPLES

    def _reset_profiles(self) -> None:
        super()._reset_profiles()
        self.sig_match = [0] * 8
        self.sig_miss = [0] * 8
        self.sig_violation = [False] * 8
        self._boards = BoardCache()

    # ------------------------------------------------------------------ replay
    @staticmethod
    def _replay(start_stacks: Sequence[int], history, bb: int, sb_amt: int) -> List[_Ctx]:
        n = len(start_stacks)
        stacks = list(start_stacks)
        folded = [False] * n
        live = [s > 0 for s in start_stacks]
        pot = 0
        out: List[_Ctx] = []
        cur = -1
        bets = [0] * n
        high = 0
        lrs = bb
        board_len = (0, 3, 4, 5)
        for st, s, a, add in history:
            if not 0 <= s < n:
                continue
            if st != cur:
                cur = st
                bets = [0] * n
                high = 0
                lrs = bb
            if a in ("sb", "bb"):
                stacks[s] -= add
                bets[s] += add
                pot += add
                high = max(high, bets[s])
                continue
            c = _Ctx()
            c.street = st
            c.seat = s
            c.act = a
            c.add = add
            c.to_call = max(0, min(stacks[s], high - bets[s]))
            c.pot = pot
            c.stack = stacks[s]
            c.my_bet = bets[s]
            c.high = high
            c.min_raise_to = high + lrs
            c.n_opp = sum(1 for i in range(n) if live[i] and not folded[i] and i != s)
            c.board_len = board_len[min(3, st)]
            stacks[s] -= add
            bets[s] += add
            pot += add
            c.total_after = bets[s]
            if a == "fold":
                folded[s] = True
            if bets[s] > high:
                if bets[s] - high >= lrs:
                    lrs = bets[s] - high
                high = bets[s]
            out.append(c)
        return out

    def _sig_check(self, c: _Ctx, bb: int, sb_amt: int) -> Optional[bool]:
        """Does a raise/bet match the starter's sizing? None = not a sizing action."""
        if c.act != "raise":
            if (c.act == "call" and c.stack / max(1, bb + sb_amt) < 4.5
                    and 0 < c.to_call < c.stack):
                return False  # starter never just calls when short (all-in reads as call)
            return None
        max_total = c.my_bet + c.stack
        m_stack = c.stack / max(1, bb + sb_amt)
        if 4.5 <= m_stack < 5.5:
            return None
        if m_stack < 5:
            return c.total_after == max_total or None  # all-in: weak evidence
        if c.to_call == 0:
            want = c.high + c.pot // 2
        else:
            want = 2 * c.min_raise_to
        want = max(min(want, max_total), min(c.min_raise_to, max_total))
        if want >= max_total:
            return None if c.total_after == max_total else False
        return abs(c.total_after - want) <= 1

    def is_starter(self, s: int) -> bool:
        return (not self.sig_violation[s] and self.sig_match[s] >= self.SIG_MIN_MATCH
                and self.sig_miss[s] * 4 <= self.sig_match[s])

    # ------------------------------------------------------------- range masks
    def _strength_post(self, board: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        score, valid = self._boards.get(board)
        return _STARTER_POST_STRENGTH[score >> 20], valid

    def _range_mask(self, seat: int, ctxs: List[_Ctx], full_board: Sequence[int], bb: int,
                    sb_amt: int, dead_mask: int) -> Optional[np.ndarray]:
        mask = (COMBO_MASK & dead_mask) == 0
        for c in ctxs:
            if c.seat != seat:
                continue
            if c.street == 0:
                strength = 0.2 + 0.6 * PREFLOP_RANK[min(7, max(1, c.n_opp))]
            else:
                strength, valid = self._strength_post(full_board[: c.board_len])
                mask &= valid
            t = self.TOL_PF if c.street == 0 else 0.0
            m_stack = c.stack / max(1, bb + sb_amt)
            if 4.5 <= m_stack < 5.5:
                continue  # either regime possible (SDK may count M differently)
            if m_stack < 5:
                # all_in() with stack <= to_call executes as a call
                if c.act == "raise" or (c.act == "call" and c.stack <= c.to_call):
                    mask &= strength > 0.5 - t
                elif c.act in ("fold", "check"):
                    mask &= strength <= 0.5 + t
                else:
                    return None
                continue
            if c.to_call == 0:
                if c.act == "raise":
                    mask &= strength > 0.6 - t
                elif c.act == "check":
                    mask &= strength <= 0.6 + t
            else:
                po = c.to_call / (c.pot + c.to_call)
                tp = t + self.TOL_PO
                raise_legal = c.stack > c.to_call
                if c.act == "raise":
                    mask &= strength > 0.75 - t
                elif c.act == "call":
                    mask &= strength > po + 0.05 - tp
                    if raise_legal:
                        mask &= strength <= 0.75 + t
                elif c.act == "fold":
                    mask &= strength <= po + 0.05 + tp
        return mask

    # ----------------------------------------------------------- hand-end hook
    def on_hand_end(self, hero_idx: int, hand_summary: Dict[str, Any]) -> None:
        hn = hand_summary.get("hand_number", 0)
        fresh = hn > self._last_recorded_hand
        super().on_hand_end(hero_idx, hand_summary)
        if not fresh:
            return
        try:
            self._update_signatures(hero_idx, hand_summary)
        except Exception:
            pass
        self._boards = BoardCache()

    def _update_signatures(self, hero_idx: int, hs: Dict[str, Any]) -> None:
        hist = hs.get("action_history") or []
        start = hs.get("starting_stacks") or []
        if not hist or not start:
            return
        bb = next((amt for st, s, a, amt in hist if a == "bb"), 0) or 20
        sb_amt = next((amt for st, s, a, amt in hist if a == "sb"), 0) or max(1, bb // 2)
        ctxs = self._replay(start, hist, bb, sb_amt)
        for c in ctxs:
            if c.seat == hero_idx:
                continue
            r = self._sig_check(c, bb, sb_amt)
            if r is True:
                self.sig_match[c.seat] += 1
            elif r is False:
                self.sig_miss[c.seat] += 1
        # Showdown verification: revealed cards must be consistent with every action.
        shown = hs.get("showdown_hands") or {}
        board = list(hs.get("board") or [])
        if shown and len(board) == 5:
            for s, cards in shown.items():
                if s == hero_idx or self.sig_violation[s]:
                    continue
                c0, c1 = int(cards[0]), int(cards[1])
                m = self._range_mask(s, ctxs, board, bb, sb_amt, 0)
                if m is None:
                    self.sig_violation[s] = True
                    continue
                idx = _combo_index(c0, c1)
                if not bool(m[idx]):
                    self.sig_violation[s] = True

    # --------------------------------------------------------------- decision
    def act(self, state: Dict[str, Any]) -> Tuple[str, int]:
        if (state["street"] > 0 and state["stacks"][state["hero_idx"]] > 0
                and self._br_ema_ns < 650_000):
            t_start = _now()
            t0 = state.get("_t0") or t_start
            if t_start - t0 < 150_000:
                try:
                    r = self._best_response(state)
                except Exception:
                    r = None
                if r is not None:
                    el = _now() - t_start
                    self._br_ema_ns = 0.9 * self._br_ema_ns + 0.1 * el
                    # slow host: shrink the Monte Carlo before it threatens the 1 ms clock
                    if self._br_ema_ns > 420_000 and self._br_samples > 90:
                        self._br_samples = 90
                    elif self._br_ema_ns < 300_000 and self._br_samples < self.BR_SAMPLES:
                        self._br_samples = self.BR_SAMPLES
                    return r
        return super().act(state)

    def _profile(self, **kw):
        if self.P_RIVX_ON > 0 and len(kw.get("board", ())) == 5 and kw.get("num_opponents", 1) <= 1:
            try:
                return river_exact_profile(**kw)
            except Exception:
                pass
        return super()._profile(**kw)

    def _best_response(self, state: Dict[str, Any]) -> Optional[Tuple[str, int]]:
        t0 = state.get("_t0") or _perf_counter_ns()
        hero = state["hero_idx"]
        active = state["active_seats"]
        opps = [i for i in range(8) if i != hero and active[i]]
        if not opps or len(opps) > 3:
            return None
        if not all(self.is_starter(o) for o in opps):
            return None

        bb = max(1, state["big_blind"])
        sb_amt = state["small_blind"]
        stacks = state["stacks"]
        board = list(state["board"])
        h0, h1 = state["hole_cards"]
        start = state.get("starting_stacks") or [stacks[i] + state["hand_contributions"][i] for i in range(8)]
        hist = state.get("action_history") or []
        ctxs = self._replay(start, hist, bb, sb_amt)
        dead = (1 << h0) | (1 << h1)
        for c in board:
            dead |= 1 << c
        masks = []
        for o in opps:
            m = self._range_mask(o, ctxs, board, bb, sb_amt, dead)
            if m is None:
                return None
            idx = np.flatnonzero(m)
            if idx.size == 0:
                return None  # inconsistent with the model: do not trust it this hand
            masks.append(idx)

        # ---- joint Monte Carlo over opponent ranges and the runout
        S = self._br_samples
        rng = self._np_rng
        nb = len(board)
        need = 5 - nb
        opp_c0 = []
        opp_c1 = []
        ok = np.ones(S, dtype=bool)
        used = np.full(S, np.int64(dead))
        for idx in masks:
            pick = idx[rng.integers(0, idx.size, S)]
            a = COMBO_C0[pick]
            b = COMBO_C1[pick]
            m = COMBO_MASK[pick]
            ok &= (used & m) == 0
            used |= m
            opp_c0.append(a)
            opp_c1.append(b)
        run = []
        if need:
            deck = np.array([c for c in range(52) if not (dead >> c) & 1], dtype=np.int64)
            for _ in range(need):
                cards = deck[rng.integers(0, deck.size, S)]
                bit = np.int64(1) << cards
                ok &= (used & bit) == 0
                used |= bit
                run.append(cards)
        if ok.sum() < 40:
            return None
        sel = np.flatnonzero(ok)
        opp_c0 = [x[sel] for x in opp_c0]
        opp_c1 = [x[sel] for x in opp_c1]
        run = [x[sel] for x in run]
        n = sel.size
        bcols = [np.full(n, c, dtype=np.int64) for c in board] + run
        my_final = _eval_np([np.full(n, h0, dtype=np.int64), np.full(n, h1, dtype=np.int64)] + bcols)
        opp_final = [_eval_np([opp_c0[k], opp_c1[k]] + bcols) for k in range(len(opps))]
        # opponents' current strength (what they act on now)
        cur_score, _ = self._boards.get(board)
        cur_str = []
        for k in range(len(opps)):
            ci = _combo_index_np(opp_c0[k], opp_c1[k])
            cur_str.append(_STARTER_POST_STRENGTH[cur_score[ci] >> 20])
        my_cur_cat = int(_eval_np([np.array([h0]), np.array([h1])] + [np.array([c]) for c in board])[0] >> 20)

        # ---- action EVs (chips relative to folding now)
        pot = state["pot"]
        bets = state["street_bets"]
        my_bet = bets[hero]
        high = max(bets)
        to_call = min(stacks[hero], max(0, high - my_bet))
        my_stack = stacks[hero]
        min_raise_add = state["min_raise_add"]
        order = _order_after(hero, opps)  # opponents in acting order after hero

        def showdown(callers_mask_list, pot_final):
            # hero share of pot among callers (ties split); vectors over samples
            best = my_final.copy()
            win = np.ones(n, dtype=bool)
            ties = np.ones(n)
            for k in range(len(opps)):
                c = callers_mask_list[k]
                of = opp_final[k]
                beat = c & (of > my_final)
                tie = c & (of == my_final)
                win &= ~beat
                ties += tie
            del best
            return np.where(win, pot_final / ties, 0.0)

        evs: Dict[Tuple[str, int], float] = {}
        bf, _ = compute_bubble_factor(
            [stacks[i] + state["hand_contributions"][i] for i in range(8)], hero, opps[0],
            min(my_stack, max(to_call, bb)), max(bb, pot), state.get("hand_number", 1))
        bf = max(1.0, bf)

        def score_ev(payoff, cost):
            # ICM-aware: losses weigh bf times as much as gains
            net = payoff - cost
            return float(np.mean(np.where(net >= 0, net, net * bf)))

        in_hand = [np.ones(n, dtype=bool) for _ in opps]

        # Opponents who have not acted on the current bet level yet (still to act)
        def responses(total_to: int, base_pot: int):
            """Starter replies to a bet/raise to `total_to`; returns (callers, pot, hero_extra_loss_on_reraise, reraised)."""
            callers = []
            pot_now = np.full(n, float(base_pot))
            reraised = np.zeros(n, dtype=bool)
            for k_opp in order:
                k = opps.index(k_opp)
                o_bet = bets[k_opp]
                o_stack = stacks[k_opp]
                tc_o = min(o_stack, total_to - o_bet)
                s = cur_str[k]
                if tc_o <= 0:
                    callers.append((k, np.ones(n, dtype=bool)))
                    continue
                po = tc_o / (pot_now + tc_o)
                short = o_stack / max(1, bb + sb_amt) < 5
                if short:
                    call = s > 0.5
                    rr = np.zeros(n, dtype=bool)
                else:
                    rr = (s > 0.75) & (o_stack > tc_o)
                    call = (~rr) & (s > po + 0.05)
                reraised |= rr
                pot_now = pot_now + np.where(call | rr, tc_o, 0)
                callers.append((k, call | rr))
            return callers, pot_now, reraised

        # FOLD
        if to_call > 0:
            evs[("fold", 0)] = 0.0

        # CHECK / CALL
        if to_call == 0:
            # starters behind us bet pot//2 with strength > 0.6; we call or fold that
            bet_amt = pot // 2
            bettor = np.zeros(n, dtype=bool)
            for k_opp in order:
                k = opps.index(k_opp)
                bettor |= cur_str[k] > 0.6
            all_in = [np.ones(n, dtype=bool) for _ in opps]
            sd_pay = showdown(all_in, pot)
            # if someone bets: call if our equity vs the betting range justifies it
            if bettor.any():
                call_pay = showdown(all_in, pot + 2 * bet_amt)
                eq_call = float(np.mean(call_pay[bettor] > 0))
                po_c = bet_amt / (pot + 2 * bet_amt)
                if eq_call > po_c:
                    pay = np.where(bettor, call_pay, sd_pay)
                    cost = np.where(bettor, float(bet_amt), 0.0)
                else:
                    pay = np.where(bettor, 0.0, sd_pay)
                    cost = np.zeros(n)
            else:
                pay, cost = sd_pay, np.zeros(n)
            evs[("check", 0)] = score_ev(pay, cost)
        else:
            callers, pot_after, _rr = responses(high, pot + to_call)
            cm = [np.ones(n, dtype=bool) for _ in opps]
            for k, c in callers:
                cm[k] = c
            pay = showdown(cm, pot_after)
            evs[("call", to_call)] = score_ev(pay, float(to_call))

        # BETS / RAISES
        if my_stack > to_call and min_raise_add <= my_stack:
            base = pot + to_call
            fracs = (0.34, 0.6, 1.0, 1.6) if to_call == 0 else (0.7, 1.4)
            sizes = set()
            for f in fracs:
                add = to_call + int(f * base)
                add = max(min_raise_add, add)
                if add >= my_stack * 0.85:
                    add = my_stack
                sizes.add(min(my_stack, add))
            sizes.add(my_stack)
            for add in sorted(sizes):
                total = my_bet + add
                callers, pot_after, reraised = responses(total, pot + add)
                cm = [np.zeros(n, dtype=bool) for _ in opps]
                for k, c in callers:
                    cm[k] = c
                any_call = np.zeros(n, dtype=bool)
                for c in cm:
                    any_call |= c
                pay = np.where(any_call, showdown(cm, pot_after), float(pot + add))
                cost = np.full(n, float(add))
                if reraised.any() and add < my_stack:
                    # facing a starter re-raise (trips+): continue only with trips+ ourselves
                    if my_cur_cat >= 3:
                        extra = float(min(my_stack - add, add * 2))
                        pay = np.where(reraised, showdown(cm, pot_after + 2 * extra), pay)
                        cost = np.where(reraised, add + extra, cost)
                    else:
                        pay = np.where(reraised, 0.0, pay)
                evs[("raise", add)] = score_ev(pay, cost)
                if _perf_counter_ns() - t0 > self.BR_BUDGET_NS:
                    break

        best = max(evs.items(), key=lambda kv: kv[1])
        return best[0]


def _now() -> int:
    return _perf_counter_ns()


def _combo_index(c0: int, c1: int) -> int:
    return int(COMBO_INDEX[c0, c1])


def _combo_index_np(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    return COMBO_INDEX[c0, c1]


def _order_after(hero: int, opps: List[int]) -> List[int]:
    return sorted(opps, key=lambda o: (o - hero) % 8)
