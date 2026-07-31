"""Day-level feasibility check (Section V-D of the paper).

The MILP plans at monthly granularity: constraints (4) and (5) bound a
trailer's trips and mileage using D*A_t, which is generally a fractional
number of operating days. A plan satisfying them need not decompose into
whole days.

This module tests decomposition directly. For a given trailer, the trips
assigned to the same station are interchangeable, so the question reduces to
an integer feasibility problem over the distinct round-trip distances: can
n[k][day] trips of distance dist[k] be placed across the available days so
that no day exceeds M_t trips or L_t kilometres? That is solved exactly with
HiGHS, so a positive answer is a constructive proof, not an estimate.

``min_days`` additionally reports the smallest number of operating days the
trailer's assignment can be squeezed into, which is the quantity a dispatcher
actually cares about.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter

import highspy


def _feasible(counts, dists, ndays, max_trips, max_km, workdir):
    """Can these trips be packed into ndays days? Exact."""
    K = len(dists)
    lines = ["Minimize", " obj: 0 z", "Subject To"]
    n = 0
    for k in range(K):                       # every trip is scheduled once
        n += 1
        lines.append(f" c{n}: " + " ".join(f"+ n_{k}_{d}" for d in range(ndays))
                     + f" = {counts[k]}")
    for d in range(ndays):                   # per-day trip cap
        n += 1
        lines.append(f" c{n}: " + " ".join(f"+ n_{k}_{d}" for k in range(K))
                     + f" <= {max_trips}")
    for d in range(ndays):                   # per-day distance cap
        n += 1
        lines.append(f" c{n}: " + " ".join(
            f"+ {dists[k]:.6f} n_{k}_{d}" for k in range(K))
            + f" <= {max_km:.6f}")
    lines += ["Bounds", " z = 0", "General",
              " " + " ".join(f"n_{k}_{d}" for k in range(K)
                             for d in range(ndays)), "End"]

    path = os.path.join(workdir, "pack.lp")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 30.0)
    h.readModel(path)
    h.run()
    return h.modelStatusToString(h.getModelStatus()) == "Optimal"


def check(values, p, compute_min_days=True):
    """Per-trailer day-level feasibility for a solved plan.

    Returns one record per trailer that carries at least one trip.
    """
    T, I, days = p["T"], p["I"], p["days"]
    ti = {t: k for k, t in enumerate(T)}
    ii = {i: k for k, i in enumerate(I)}
    rows = []

    with tempfile.TemporaryDirectory() as workdir:
        for t in T:
            c = Counter()
            for i in I:
                k = round(values.get(f"y_{ii[i]}_{ti[t]}", 0.0))
                if k:
                    c[round(p["d"][i[0]], 6)] += k
            if not c:
                continue

            dists = sorted(c)
            counts = [c[v] for v in dists]
            total_trips = sum(counts)
            total_km = sum(v * c[v] for v in dists)
            available = int(math.floor(days * p["A"][t]))
            max_trips, max_km = p["M"][t], p["L"][t]

            ok = _feasible(counts, dists, available, max_trips, max_km, workdir)

            min_days = None
            if compute_min_days:
                lo = max(math.ceil(total_trips / max_trips),
                         math.ceil(total_km / max_km - 1e-9))
                for nd in range(lo, available + 4):
                    if _feasible(counts, dists, nd, max_trips, max_km, workdir):
                        min_days = nd
                        break

            rows.append(dict(trailer=t, type=p["TYPE"][t],
                             available_days=available, trips=total_trips,
                             km=round(total_km, 2), feasible=ok,
                             min_days=min_days))
    return rows
