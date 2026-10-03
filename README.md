# Castellan: a Code Bot entry

This is a single-file bot for the Code Bot hackathon. It uses the generals.bot competition ruleset and the engine pinned at `strakam/generals-bots@13db8f6`.

| File | Purpose |
|---|---|
| `bot.py` | Policy source. Exposes `act(observation)`, uses the standard library only. |
| `submission.py` | The file to submit: `bot.py` with the RL-trained weights baked in. Rename it to `<participant_id>.py`. |
| `tools/arena.py` | In-process match runner on the pinned engine, run with the competition rules (castle building and deathtouch). |
| `tools/train_es.py` | Evolution-strategies RL trainer for `PARAMS`. |
| `tools/export.py` | Bakes trained parameters into a submission file. |
| `tools/expander.py` | The organizer's reference expander bot, ported to `act(dict)`. |

## Research: how this ruleset is won

* **Actions are the bottleneck.** You get one move per turn. A castle costs 35 army plus a proximity surcharge. Building it takes one action, and it then produces 0.5 army per turn, so it pays back in about 70 turns. Land produces only 1 army per tile every 50 turns. Early castles therefore pay off much more than land, provided the general survives.
* **Fog type 0 is always passable.** Mountains and castles in fog show up as type 5. That makes BFS through fog a reliable pathfinder.
* **Deathtouch (from turn 800).** Any move that executes onto the enemy general wins. From then on, an enemy tile next to your general holding 2 or more army is lethal on the next turn. The only defense is to capture that tile, because a "chase" move resolves first. The general's own counter-move does not save you, since the smaller army resolves first. So the bot:
  * clears enemy tiles near its general with high priority after turn 760
  * starts a push toward the (estimated) enemy general at about turn 700, so it is in touching range at turn 800.
* **Aggression beats pure economy.** In testing, an all-in "walk the biggest army at the enemy" variant beat the castle-heavy policy until defense was added: intercepting big incoming armies, gathering to the general, and the general keeping half its army after the opening. The RL opponent pool therefore always includes that aggressor.
* **Draws at turn 1200 are worth only 0.5**, and territory does not break ties. The bot uses its army lead to finish games rather than farm.

## RL training

The policy is a linear scorer over hand-built action features (expand, explore, capture, gather). It also has strategic thresholds: castle timing and limit, when to attack, rush turn, defense radius, and how far an interceptor may be from a threat.

`tools/train_es.py` tunes these weights with OpenAI-style evolution strategies:

* antithetic Gaussian perturbations in log space
* rank-normalized fitness
* reward per game: 1 + 0.25·(speed bonus) for a win, 0.4 for a draw, 0 for a loss
* games played on freshly generated competition maps, with both starting slots, against:
  * the current champion (self-play)
  * the aggressor
  * the expander

```bash
git clone https://github.com/strakam/generals-bots /tmp/generals-bots
git -C /tmp/generals-bots checkout 13db8f69a422380ea184d2f4ca262a38866c5fc6
pip install jax numpy
python tools/arena.py bot.py tools/expander.py 4     # evaluate
python tools/train_es.py 60                          # train (4 processes)
python tools/export.py tools/params_best.json theta submission.py
```

Per-move time is about 2–6 ms. The limit is 150 ms.
