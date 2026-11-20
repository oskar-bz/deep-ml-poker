134163,e=>{"use strict";let t=`# Build a no-limit hold'em bot. Every code cell is submitted together as your bot.
# Tables are 8-seat sit-and-gos: 2,000 chips each, blinds 10/20 rising every 25
# hands, played until one bot has every chip (or 200 hands). Only the top three at
# a table score: 10, 5 and 2 points. Surviving without winning chips earns nothing.
# Each decision gets 1 ms of think time (a late answer is a check or fold). During
# matches your bot has numpy and PyTorch (CPU) and no network. Want a trained
# model? The contest page's Bot tab has a complete example that stays under 1 ms.
from pokerbot_sdk import Bot, Action


class MyBot(Bot):
    def act(self, state):
        # state.hole / state.board: cards like "As", "Td"
        # state.to_call, state.pot, state.min_raise, state.max_raise (raise-to totals)
        # state.legal: e.g. ["fold", "call", "raise"]
        # state.preflop_rank(): 1.0 = aces .. 0.0 = the worst hand, vs the players still in
        # state.made_hand(): {"name": "two pair", "category": 2, ...} from the flop on
        # state.stack_m: your stack in orbits of blinds (push-or-fold territory below ~5)
        # All of these are table lookups: microseconds, not Monte Carlo.
        if state.street == "preflop":
            strength = 0.2 + 0.6 * state.preflop_rank()
        else:
            strength = {0: 0.25, 1: 0.45, 2: 0.65}.get(state.made_hand()["category"], 0.85)

        if state.stack_m < 5:  # short: shove good hands, fold the rest
            if strength > 0.5:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()

        if state.to_call == 0:
            if strength > 0.6 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + state.pot // 2)
            return Action.check()

        if strength > 0.75 and "raise" in state.legal:
            return Action.raise_to(state.min_raise * 2)
        if strength > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()
`,a=`# Test against the house bots: 20 sit-and-go tables, with your think-time stats
# against the 1 ms clock. Lines that call play_local are stripped from your
# submitted bot, so rename MyBot here too if you rename the class.
from pokerbot_sdk import play_local

play_local(MyBot(), tables=20)
`,s=`# One cell: train a small equity model with PyTorch here, run it with numpy in matches.
# Run it once in the notebook (a minute or two: it labels random spots with Monte
# Carlo equity, fits the net, and saves the weights as numpy arrays with
# save_weights). Test with play_local, then press Submit. In matches the same cell
# finds the saved weights, skips training, and the bot's forward pass is two small
# numpy matmuls, well inside the 1 ms clock.
import random, time
import numpy as np
from pokerbot_sdk import Bot, Action, parse_card, card_str, equity, save_weights, load_arrays


def features(hole, board, opponents):
    # 52 one-hot hole cards + 52 one-hot board cards + opponents + street -> win share
    x = np.zeros(106, dtype=np.float32)
    for c in hole:
        x[parse_card(c)] = 1.0
    for c in board:
        x[52 + parse_card(c)] = 1.0
    x[104] = opponents / 7.0
    x[105] = len(board) / 5.0
    return x


def train(n_samples=4000, epochs=30):
    """Label random situations with Monte Carlo equity, fit a net, export numpy weights."""
    import torch
    import torch.nn as nn
    rng = random.Random(0)
    X, y, t0 = [], [], time.time()
    for i in range(n_samples):
        deck = list(range(52)); rng.shuffle(deck)
        n_board = rng.choice([0, 0, 3, 4, 5])
        opponents = rng.randint(1, 7)
        hole, board = deck[:2], deck[2:2 + n_board]
        X.append(features([card_str(c) for c in hole], [card_str(c) for c in board], opponents))
        y.append(equity(hole, board, opponents=opponents, samples=100, rng=rng))
        if i % 1000 == 999:
            print(f"labelled {i + 1}/{n_samples} ({time.time() - t0:.0f}s)")
    X = torch.tensor(np.stack(X)); y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    net = nn.Sequential(nn.Linear(106, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for b in range(0, len(X), 256):
            idx = perm[b:b + 256]
            loss = nn.functional.mse_loss(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                print(f"epoch {epoch + 1}: rmse {torch.sqrt(nn.functional.mse_loss(net(X), y)):.3f}")
    # Plain numpy arrays: the bot never needs torch at the table.
    return {k: v.detach().numpy() for k, v in net.state_dict().items()}


W = load_arrays()                   # None the first time, the saved weights after that
if W is None:
    W = train()
    print(f"saved {save_weights(W) // 1024} KB of weights; submit whenever you're ready")


def predict(x):
    h = np.maximum(x @ W["0.weight"].T + W["0.bias"], 0)
    h = np.maximum(h @ W["2.weight"].T + W["2.bias"], 0)
    return float(1 / (1 + np.exp(-(h @ W["4.weight"].T + W["4.bias"])[0])))


class MyBot(Bot):
    def act(self, state):
        eq = predict(features(state.hole, state.board, max(1, state.opponents)))
        if state.stack_m < 5:  # short stack: shove good hands, fold the rest
            if eq > 0.5:
                return Action.all_in()
            return Action.check() if state.to_call == 0 else Action.fold()
        if state.to_call == 0:
            if eq > 0.6 and "raise" in state.legal:
                return Action.raise_to(state.current_bet + state.pot // 2)
            return Action.check()
        if eq > 0.75 and "raise" in state.legal:
            return Action.raise_to(state.min_raise * 2)
        if eq > state.pot_odds() + 0.05:
            return Action.call()
        return Action.fold()
`;e.s(["POKERBOT_API_DOC",0,[["act(state)","Return Action.fold(), Action.check(), Action.call(), Action.raise_to(total) or Action.all_in()."],["state.hole, state.board",'Your two cards and the board, as strings like "As" or "Td".'],["state.to_call","Chips you need to add to call (0 means you can check)."],["state.min_raise / max_raise",'Smallest and largest legal "raise to" total for this street.'],["state.pot, state.stacks, state.bets","Chips in the pot, behind each seat, and bet this street."],["state.big_blind, state.small_blind, state.level","The blinds this hand and the blind level (they rise every 25 hands)."],["state.stack_m","Your stack in orbits of blinds; below about 5 it is push or fold."],["state.preflop_equity() / preflop_rank()","Table lookup of your hole cards against the players still in: equity, or rank among the 169 hands (1.0 = aces). Microseconds."],["state.made_hand()",'The hand you hold now: {"name", "category" (0 high card .. 8 straight flush), "value"}. Microseconds.'],["state.equity(samples)","Monte Carlo pot share. About 3 ms: fine in the notebook, too slow for a 1 ms decision."],["state.history","Every action so far this hand: street, seat, action, amount."],["on_hand_end(result)","Optional hook with the deltas, stacks and any cards shown, for opponent modeling. Keep it cheap."],["save_weights(model)","Save an nn.Module, state_dict, dict of numpy arrays or bytes; submitted with your code."],["load_state_dict() / load_arrays() / load_weights()","Read the saved weights back, in the notebook or in a match (None if none saved)."],["clear_weights()","Delete the saved weights so your cell trains again."],["play_local(bot, tables=20)","Sit-and-go tables against house bots in the notebook: points, finishes and think-time stats."]],"STARTER_BOT",0,t,"STARTER_TEST",0,a,"TRAINED_BOT_EXAMPLE",0,s])},