"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target, as five quantiles per trophy (Python 3.13, stdlib only).

Model, in one paragraph
-----------------------
An earn rate is a ratio, rate = earners / players_tracked, and the two sides
move for different reasons: new players joining Exophase dilute every rate,
while the players already tracked keep earning trophies. So this program
models the two sides separately and divides at the target, instead of
extrapolating the ratio itself:

  * players_tracked N(t): log N is fitted against x = ln(1 + hours since the
    2026-11-19T00:00Z release) over the case's distinct reads (consecutive
    identical reads collapsed, so a long flat stretch between Exophase
    refreshes cannot outweigh the jumps).
  * earners E_i(t) per trophy: log(E_i + 0.5) is fitted against the same x,
    per trophy, with each trophy's slope shrunk toward its tier's mean slope
    (rare trophies have few earners, so their own slope is noisy).
  * the rate forecast is 100 * E_i(target) / N(target).

That ratio forecast is ensembled, in logit space, with four other views of
the same history, all anchored on the last read so a level error cannot
creep in through a regression intercept:

  A  persistence: the last read's rate;
  B  per-trophy logit(rate) linear in x, slope shrunk toward the tier mean;
  C  pooled logit(rate) linear in x: one slope shared by all trophies
     (the adoption curve's shape is mostly common), per-trophy anchor;
  D  the earners / players power-law ratio above;
  E  per-trophy logit(rate) linear in clock hours, slope shrunk, as a
     guard against log-time being the wrong clock.

The ensemble weights start from a prior (A .15, B .20, C .20, D .30, E .15)
and are tilted by walk-forward validation inside the case's own history:
each candidate is fitted on the history up to a cut (halfway, three
quarters) and scored on its prediction of the last distinct read, and the
weights move toward the candidates that actually predicted the future
best. With fewer than four distinct reads there is nothing to validate
on, so the prior weights stand.

Spread (the quantiles)
----------------------
The predictive spread in logit units is
    sigma = clip(sqrt(horizon^2 + resid^2 + flat^2), 0.55, 1.6),
where horizon = 0.40 * sqrt(horizon in days) is a random-walk term,
resid is the ensemble's own walk-forward logit error for that trophy
(how predictable this trophy has been), and flat = 0.85 when the whole
history is a single distinct read: with no movement ever observed, a
refresh before the target is a live possibility, so the band is wide.
In that flat case the median is also shifted down 0.35 logit: Exophase
refreshes a game's counts in jumps, and a jump adds many fresh players
who have earned little yet, so the first refresh after a flat stretch
dilutes rates; the shift is a one-refresh mixture prior, and it is only
applied when the history shows no movement at all.

The trophy at 100% in the last read is special-cased: it survived every
read at 100%, so its median is 99.9% with a 0.8-logit spread.

Everything is deterministic: plain float arithmetic, sorted file order,
no randomness, no network, no imports beyond the standard library.
Two runs on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; x = ln(1 + hours since then).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior ensemble weights for candidates A..E (see module docstring).
PRIOR_W = [0.15, 0.20, 0.20, 0.30, 0.15]
# Candidate names, in the same order.
CANDIDATES = ["persist", "logit_logt", "logit_pooled", "earners_ratio", "logit_linear"]


def parse_time(s):
    """ISO 8601 UTC string -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def x_of(dt):
    """Log-time clock: ln(1 + hours since release), floored at 0."""
    return math.log1p(max(0.0, hours_since_release(dt)))


def clamp_pct(p):
    return min(99.99, max(0.01, p))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    lg = max(-12.0, min(12.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when it cannot be fitted."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def distinct_snapshots(history):
    """Collapse consecutive identical reads (same players + same rates).

    Exophase refreshes in jumps, so the raw history is long flat stretches
    punctuated by jumps; fitting on the raw reads would let the flat
    stretches outweigh the jumps. Returns a list of dicts with keys
    at (datetime), x (log-time), players, rates (list, may hold None).
    """
    snaps = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(
            {
                "at": parse_time(read["at"]),
                "x": x_of(parse_time(read["at"])),
                "tau": hours_since_release(parse_time(read["at"])),
                "players": read.get("players_tracked"),
                "rates": rates,
            }
        )
    return snaps


def last_known_rates(snaps, n_trophies):
    """Last non-null rate per trophy over the snapshots, else 0.5%."""
    last = [None] * n_trophies
    for snap in snaps:
        for i, r in enumerate(snap["rates"][:n_trophies]):
            if r is not None:
                last[i] = r
    return [r if r is not None else 0.5 for r in last]


def candidate_logits(snaps, trophies, target_dt):
    """Per-candidate median logits at target_dt, anchored on the last snap.

    Returns {candidate: [logit per trophy]}. Fits use only `snaps`.
    """
    n = len(trophies)
    if not snaps:
        return {c: [logit(0.5)] * n for c in CANDIDATES}
    last = snaps[-1]
    x_last = last["x"]
    tau_last = last["tau"]
    x_star = x_of(target_dt)
    tau_star = hours_since_release(target_dt)
    dx = x_star - x_last
    dt = tau_star - tau_last
    base = last_known_rates(snaps, n)
    base_logit = [logit(r) for r in base]
    n_last = last["players"] or 0

    xs = [s["x"] for s in snaps]
    taus = [s["tau"] for s in snaps]

    # --- per-trophy slopes, then tier means for shrinkage ---
    # logit(rate) on x
    slope_logit = []
    # log(E + 0.5) on x, E = rate/100 * players
    slope_loge = []
    # logit(rate) on clock hours
    slope_lin = []
    for i in range(n):
        lx, ly, le, lt = [], [], [], []
        for s in snaps:
            r = s["rates"][i] if i < len(s["rates"]) else None
            if r is None or s["players"] is None:
                continue
            lx.append(s["x"])
            ly.append(logit(r))
            le.append(math.log(max(0.0, r / 100.0 * s["players"]) + 0.5))
            lt.append(s["tau"])
        slope_logit.append(ols_slope(lx, ly))
        slope_loge.append(ols_slope(lx, le))
        slope_lin.append(ols_slope(lt, ly))
    tier_mean_logit = {}
    tier_mean_loge = {}
    tier_mean_lin = {}
    for tier in sorted({t.get("tier") for t in trophies}):
        idx = [i for i, t in enumerate(trophies) if t.get("tier") == tier]
        tier_mean_logit[tier] = mean([slope_logit[i] for i in idx])
        tier_mean_loge[tier] = mean([slope_loge[i] for i in idx])
        tier_mean_lin[tier] = mean([slope_lin[i] for i in idx])
    pooled_logit = mean(slope_logit)
    # log N on x (one shared series)
    logn_slope = ols_slope(
        [s["x"] for s in snaps if s["players"]],
        [math.log(s["players"]) for s in snaps if s["players"]],
    )
    m = len(snaps)
    shrink = m / (m + 2.0)  # weight on a trophy's own slope vs its tier mean

    out = {c: [0.0] * n for c in CANDIDATES}
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier")
        s_logit = shrink * slope_logit[i] + (1 - shrink) * tier_mean_logit.get(tier, pooled_logit)
        s_loge = shrink * slope_loge[i] + (1 - shrink) * tier_mean_loge.get(tier, 0.0)
        s_lin = shrink * slope_lin[i] + (1 - shrink) * tier_mean_lin.get(tier, 0.0)
        # Clip extrapolated moves: no candidate may move a rate by more
        # than 6 logit units from its last read, whatever the fit says.
        def move(delta):
            return max(-6.0, min(6.0, delta))

        out["persist"][i] = base_logit[i]
        out["logit_logt"][i] = base_logit[i] + move(s_logit * dx)
        out["logit_pooled"][i] = base_logit[i] + move(pooled_logit * dx)
        # Earners / players ratio: E* and N* anchored on the last read.
        e_last = max(0.0, base[i] / 100.0 * n_last) if n_last else 0.0
        e_star = (e_last + 0.5) * math.exp(move(s_loge * dx)) - 0.5
        n_star = n_last * math.exp(move(logn_slope * dx)) if n_last else 0.0
        if n_star > 0 and e_star > 0:
            out["earners_ratio"][i] = logit(100.0 * e_star / n_star)
        else:
            out["earners_ratio"][i] = base_logit[i]
        out["logit_linear"][i] = base_logit[i] + move(s_lin * dt)
    return out


def ensemble_weights(snaps, trophies, target_dt):
    """Prior weights tilted by walk-forward validation on this case.

    Fits each candidate on the history up to a cut and scores its
    prediction of the last distinct read; returns (weights, resid) where
    resid[i] is the ensemble's mean |logit error| for trophy i at the cuts
    (0.0 when validation was impossible).
    """
    n = len(trophies)
    weights = list(PRIOR_W)
    resid = [0.0] * n
    if len(snaps) < 4:
        return weights, resid
    last_rates = last_known_rates(snaps, n)
    actual = [logit(r) for r in last_rates]
    cuts = sorted({len(snaps) // 2, (3 * len(snaps)) // 4})
    errs = {c: [] for c in CANDIDATES}
    ens_err = [[] for _ in range(n)]
    for cut in cuts:
        if cut < 3 or cut >= len(snaps):
            continue
        sub = snaps[:cut]
        cand = candidate_logits(sub, trophies, snaps[-1]["at"])
        for ci, c in enumerate(CANDIDATES):
            errs[c].append(
                mean([abs(cand[c][i] - actual[i]) for i in range(n)])
            )
        for i in range(n):
            pred = sum(PRIOR_W[ci] * cand[c][i] for ci, c in enumerate(CANDIDATES))
            ens_err[i].append(abs(pred - actual[i]))
    if not any(errs[c] for c in CANDIDATES):
        return weights, resid
    mean_err = {}
    for c in CANDIDATES:
        mean_err[c] = mean(errs[c]) if errs[c] else 99.0
    best = min(mean_err.values())
    tilt = [math.exp(-(mean_err[c] - best) / 0.6) for c in CANDIDATES]
    raw = [0.5 * PRIOR_W[i] + 0.5 * tilt[i] / sum(tilt) for i in range(len(CANDIDATES))]
    total = sum(raw)
    weights = [w / total for w in raw]
    resid = [mean(e) if e else 0.0 for e in ens_err]
    return weights, resid


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    target_dt = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    snaps = distinct_snapshots(case.get("history") or [])
    cand = candidate_logits(snaps, trophies, target_dt)
    weights, resid = ensemble_weights(snaps, trophies, target_dt)
    flat = len(snaps) <= 2
    horizon_term = 0.40 * math.sqrt(max(0.0, horizon_days))
    base = last_known_rates(snaps, n)

    rows = []
    for i, trophy in enumerate(trophies):
        median_logit = sum(
            weights[ci] * cand[c][i] for ci, c in enumerate(CANDIDATES)
        )
        if flat and case["horizon_minutes"] >= 60:
            # No movement ever observed: a refresh before the target is
            # live, and a refresh dilutes rates (see module docstring).
            median_logit -= 0.35
        if base[i] >= 99.99:
            # Held at 100% through every read: stay near the top, but
            # leave a lower tail in case new players dilute it.
            median_logit = logit(99.9)
            sigma = 0.8
        else:
            flat_term = 0.85 if flat else 0.0
            resid_term = min(1.0, max(0.0, resid[i]))
            sigma = math.sqrt(
                horizon_term ** 2 + resid_term ** 2 + flat_term ** 2
            )
            sigma = max(0.55, min(1.6, sigma))
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(median_logit + z * sigma))
            q = min(99.99, max(0.01, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
