"""Multi-product variant of the trailer-allocation MILP (Section V-F).

The base model in ``milp.py`` sends every trip to one station-product pair,
so each trip carries a single grade. The fleet's trailers are multi-
compartment, and in the historical records most visits deliver two or three
grades at once. This variant indexes trips by station instead of by
station-product pair, so one trip can carry several grades:

    sum_{i at s} x_it <= C_t * y_st          whole load fits the trailer
    x_it              <= min(C_t, V_i) * y_st each grade fits its station tank

where y_st is the number of trips trailer t makes to station s. Every other
constraint, including whole operating days and the daily-packing
inequalities, carries over unchanged with d_s in place of d_i.

The data records each trailer's total tank capacity but not how it is
divided into compartments, so this variant lets a trailer split its tank
between grades in any proportion. That is optimistic, and the paper treats
its result as an upper bracket on the achievable saving rather than as a
plan.

The aggregate constraints above are sufficient for per-trip feasibility:
dividing each grade's volume evenly over the y_st trips puts at most V_i of
each grade, and at most C_t in total, on every trip.

Usage::

    import milp, multiproduct
    p = milp.load()
    multiproduct.build_lp(p, "mp.lp")
    r = milp.solve("mp.lp", gap=0.10)
    multiproduct.summarise(r["values"], p)
"""

from __future__ import annotations

import math

import milp


def _index(p):
    T, I = p["T"], p["I"]
    S = sorted({i[0] for i in I})
    return (T, I, S,
            {t: k for k, t in enumerate(T)},
            {i: k for k, i in enumerate(I)},
            {s: k for k, s in enumerate(S)},
            {s: [i for i in I if i[0] == s] for s in S})


def build_lp(p, path, kmax=milp.KMAX, buffer=milp.BUFFER, lam=milp.LAMBDA):
    """Write the multi-product MILP to ``path`` in CPLEX LP format."""
    T, I, S, ti, ii, si, pairs = _index(p)
    d = p["d"]
    N = {t: math.floor(p["days"] * p["A"][t]) for t in T}
    x = lambda i, t: f"x_{ii[i]}_{ti[t]}"     # litres of pair i on trailer t
    y = lambda s, t: f"y_{si[s]}_{ti[t]}"     # trips by trailer t to station s
    D = lambda t: f"dist_{ti[t]}"

    out = ["Minimize",
           " obj: " + " ".join(f"+ 1 {D(t)}" for t in T) + f" + {lam} maxdist",
           "Subject To"]
    n = 0

    def con(expr):
        nonlocal n
        n += 1
        out.append(f" c{n}: {expr}")

    for i in I:                                                    # demand
        con(" ".join(f"+ {x(i,t)}" for t in T) + f" >= {p['Q'][i]:.6f}")
    for s in S:
        for t in T:                                                # tank
            con(" ".join(f"+ {x(i,t)}" for i in pairs[s])
                + f" - {p['C'][t]:.6f} {y(s,t)} <= 0")
            for i in pairs[s]:                                     # storage
                cap = min(p["C"][t], p["V"][i])
                con(f"{x(i,t)} - {cap:.6f} {y(s,t)} <= 0")
    for t in T:
        con(" ".join(f"+ {y(s,t)}" for s in S)                     # trips
            + f" <= {p['M'][t] * N[t]}")
        con(" ".join(f"+ {d[s]:.6f} {y(s,t)}" for s in S)          # mileage
            + f" <= {p['L'][t] * N[t] * buffer:.6f}")
        con(f"{D(t)} " + " ".join(f"- {d[s]:.6f} {y(s,t)}" for s in S) + " = 0")
        con(f"{D(t)} - maxdist <= 0")
        for k in range(1, min(kmax, int(p["M"][t])) + 1):          # packing
            long_s = [s for s in S if d[s] > p["L"][t] / (k + 1) + 1e-9]
            if long_s:
                con(" ".join(f"+ {y(s,t)}" for s in long_s) + f" <= {k * N[t]}")
    for i in I:                                                    # refill
        con(" ".join(f"+ {x(i,t)}" for t in T) + f" <= {p['R'][i]:.6f}")

    out += ["General", " " + " ".join(y(s, t) for s in S for t in T), "End"]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    n_int = len(S) * len(T)
    return n, n_int + len(I) * len(T) + len(T) + 1, n_int


def summarise(values, p):
    """Distance, trips, trailers used, busiest trailer and load utilization."""
    T, I, S, ti, ii, si, pairs = _index(p)
    trips = {t: sum(round(values.get(f"y_{si[s]}_{ti[t]}", 0.0)) for s in S)
             for t in T}
    litres = sum(v for k, v in values.items() if k.startswith("x_"))
    capacity = sum(round(values.get(f"y_{si[s]}_{ti[t]}", 0.0)) * p["C"][t]
                   for s in S for t in T)
    return dict(
        distance=sum(v for k, v in values.items() if k.startswith("dist_")),
        max_trailer=values.get("maxdist", 0.0),
        trips=sum(trips.values()),
        trailers_used=sum(1 for t in T if trips[t] > 0),
        litres=litres,
        load_utilization=litres / capacity if capacity else 0.0,
    )


def as_pair_trips(values, p):
    """Re-express station trips in the base model's variable names.

    ``day_feasibility.check`` only needs each trailer's trip distances, so
    every trip to station s is attributed to the first pair at s.
    """
    T, I, S, ti, ii, si, pairs = _index(p)
    return {f"y_{ii[pairs[s][0]]}_{ti[t]}": round(values.get(f"y_{si[s]}_{ti[t]}", 0.0))
            for s in S for t in T}
