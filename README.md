# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/watgawd/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/watgawd/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each Grand Theft Auto VI (PS5) trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's
recorded reads up to the case checkpoint. Entry point: `main.py`
(Python 3.13, standard library only; run with `python3 main.py`, reading
`$MAKEFUN_INPUT_DIR` / `/input` and writing `$MAKEFUN_OUTPUT_DIR` /
`/output`).

## Method

An earn rate is a ratio, `rate = earners / players_tracked`, and its two
sides move for different reasons: new players joining Exophase dilute
every rate, while tracked players keep earning trophies. So the program
models the two sides separately and divides at the target:

- `players_tracked N(t)`: log N fitted against `x = ln(1 + hours since
  2026-11-19T00:00Z)` over the case's *distinct* reads (consecutive
  identical reads collapsed, so flat stretches between Exophase refreshes
  cannot outweigh the jumps).
- Earners `E_i(t)` per trophy: `log(E_i + 0.5)` fitted against the same
  `x`, per trophy, slope shrunk toward its tier's mean slope.
- Rate forecast `100 * E_i(target) / N(target)`.

That ratio forecast is ensembled in logit space with persistence, a
per-trophy logit-vs-log-time trend, a pooled (one shared slope)
logit-vs-log-time trend, and a logit-vs-clock-hours trend — all anchored
on the last read. Ensemble weights start from a prior and are tilted by
walk-forward validation inside the case's own history (fit up to a cut,
score the prediction of the last distinct read).

Spread: `sigma = clip(sqrt(horizon^2 + resid^2 + flat^2), 0.55, 1.6)`
logit units, where `horizon = 0.40 * sqrt(horizon in days)`,
`resid` is the ensemble's own walk-forward logit error for that trophy,
and `flat = 0.85` when the history is a single distinct read. With no
movement ever observed, the median is also shifted down 0.35 logit: a
refresh before the target is live, and a refresh dilutes rates. The
trophy at 100% in the last read is forecast near 99.9% with a 0.8-logit
spread.

## Local checks

- Practice cases (`p00`, `p01`): valid output, 55 trophies, monotonic
  quantiles, byte-identical across two runs, ~0.08 s for both cases.
- Against two plausible practice answers (the launch-night refresh with
  earners frozen, and with earners growing as the totals imply), this
  entry scores ~0.85 / ~0.66 logit pinball loss vs the starter baseline's
  ~1.69 / ~1.40. The true practice answers are not in the kit, so these
  are sanity checks on assumed answers, not measured scores.
- On a synthetic final-like history (players growing, early trophies
  rising to 10–35% by day 7), the trend machinery tracks the medians
  and scores ~0.21. That history was generated with the same log-time
  family the model fits, so it validates the machinery, not the real
  world; the final's true history is unknown until the deadline.

## Files

- `main.py` — the whole entry (one file, stdlib only).
- `LICENSE` — MIT (the kit starter's license).
- `.makefun/owner.json` — payout wallet for @watgawd.

---

<sub>Entered on [make.fun](https://make.fun/?ref=91fyv507&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
