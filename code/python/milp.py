"""Trailer-allocation MILP for multi-product fuel distribution.

A second implementation, in Python, of the model in ``code/model_julia.jl``.
Both build the same formulation (2,690 constraints, 4,899 variables, 2,380 of
them integer) and are solved with HiGHS; this one goes through the CPLEX LP
file format and the ``highspy`` bindings rather than JuMP.

Formulation (Section III-C of the paper)::

    min  sum_t D_t + l1 * Dmax + l2 * sum_i z_i

    s.t. sum_t x_it            >= Q_i                       for all i   (2)
         x_it                  <= y_it * min(C_t, V_i)      for all i,t (3)
         sum_i y_it            <= D * M_t * A_t             for all t   (4)
         sum_i y_it * d_i      <= D * L_t * A_t             for all t   (5)
         D_t                    = sum_i y_it * d_i          for all t   (6)
         D_t                   <= Dmax                      for all t   (7)
         sum_t x_it            <= R_i + z_i                 for all i   (8)

    x_it, z_i, D_t, Dmax >= 0 ;  y_it integer >= 0

where x_it is litres delivered to station-product pair i by trailer t, y_it is
the number of round trips, z_i is overfill, D_t is trailer distance and Dmax is
the largest trailer distance. u_t = sum_i y_it is carried as a reporting
variable so the instance dimensions match the reference implementation.

Usage::

    from milp import load, build_lp, solve
    p = load()
    build_lp(p, "model.lp")
    r = solve("model.lp", gap=0.10)
"""

from __future__ import annotations

import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data", "processed")

DAYS = 30           # planning horizon, D
LAMBDA_1 = 0.1      # imbalance weight, km per km
LAMBDA_2 = 20.0     # overfill weight, km per litre
AVAIL_RANGE = (0.20, 0.80)   # trailers retained by availability factor


# --------------------------------------------------------------------------
# instance
# --------------------------------------------------------------------------

def load(trailer_csv=None, demand_csv=None, dist_scale=None,
         avail_range=AVAIL_RANGE, days=DAYS):
    """Build a problem dictionary from the two input tables.

    dist_scale, if given, maps a station id to a multiplier applied to its
    round-trip distance. Used by the sensitivity experiments.
    """
    trailer_csv = trailer_csv or os.path.join(DATA, "truck_config_julia.csv")
    demand_csv = demand_csv or os.path.join(DATA, "station_demand_julia.csv")

    truck = pd.read_csv(trailer_csv)
    dem = pd.read_csv(demand_csv)

    A = dict(zip(truck.Trailer, truck.Availability_Percent / 100.0))
    C = dict(zip(truck.Trailer, truck.Truck_Tank_Capacity_Liters))
    L = dict(zip(truck.Trailer, truck.Distance_Limit_Per_Day))
    M = dict(zip(truck.Trailer, truck.Max_Trips_Per_Day))
    TYPE = dict(zip(truck.Trailer, truck.Truck_Type))

    # trailers outside the availability band are neither reliably active nor
    # plausibly assignable, and are excluded (Section IV)
    T = [t for t in truck.Trailer if avail_range[0] <= A[t] <= avail_range[1]]

    I = [(r.Destination_ID, r.Product) for r in dem.itertuples()]
    Q = {(r.Destination_ID, r.Product): r.Monthly_Demand_Liters
         for r in dem.itertuples()}
    V = {(r.Destination_ID, r.Product): r.Product_Storage_Capacity_Liters
         for r in dem.itertuples()}

    # distances in the input table are one-way; every journey is a round trip
    d = {r.Destination_ID: r.Distance_km * 2 for r in dem.itertuples()}
    if dist_scale:
        d = {s: v * dist_scale.get(s, 1.0) for s, v in d.items()}

    # R_i = ceil(Q_i / V_i) * V_i, i.e. a whole number of tank-fills, which is
    # always at least Q_i. This is why overfill is zero at optimality.
    R = {i: math.ceil(Q[i] / V[i]) * V[i] for i in I}

    return dict(T=T, I=I, A=A, C=C, L=L, M=M, TYPE=TYPE,
                Q=Q, V=V, R=R, d=d, days=days)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_lp(p, path, integer_days=False, packing_cuts=False, kmax=4,
             buffer=1.0, lambda_1=LAMBDA_1, lambda_2=LAMBDA_2):
    """Write the MILP to ``path`` in CPLEX LP format.

    integer_days   replace D*A_t by floor(D*A_t) in (4) and (5)
    packing_cuts   add the daily-packing inequalities, equation (14)
    buffer         scale the right-hand side of (5), e.g. 0.95 for a 5% margin
    """
    T, I, days = p["T"], p["I"], p["days"]
    ti = {t: k for k, t in enumerate(T)}
    ii = {i: k for k, i in enumerate(I)}
    x = lambda i, t: f"x_{ii[i]}_{ti[t]}"     # litres delivered
    y = lambda i, t: f"y_{ii[i]}_{ti[t]}"     # round trips
    D = lambda t: f"dist_{ti[t]}"             # distance per trailer
    u = lambda t: f"use_{ti[t]}"              # trips per trailer (reporting)
    z = lambda i: f"of_{ii[i]}"               # overfill

    out = ["Minimize"]
    obj = [f"+ 1 {D(t)}" for t in T]
    obj.append(f"+ {lambda_1} maxdist")
    obj += [f"+ {lambda_2} {z(i)}" for i in I]
    out.append(" obj: " + " ".join(obj))
    out.append("Subject To")
    n = 0

    for i in I:                                                    # (2)
        n += 1
        out.append(f" c{n}: " + " ".join(f"+ {x(i,t)}" for t in T)
                   + f" >= {p['Q'][i]:.6f}")

    for i in I:                                                    # (3)
        for t in T:
            n += 1
            cap = min(p["C"][t], p["V"][i])
            out.append(f" c{n}: {x(i,t)} - {cap:.6f} {y(i,t)} <= 0")

    for t in T:                                                    # u_t
        n += 1
        out.append(f" c{n}: {u(t)} " + " ".join(f"- {y(i,t)}" for i in I)
                   + " = 0")

    for t in T:                                                    # (6)
        n += 1
        out.append(f" c{n}: {D(t)} " + " ".join(
            f"- {p['d'][i[0]]:.6f} {y(i,t)}" for i in I) + " = 0")

    for t in T:                                                    # (7)
        n += 1
        out.append(f" c{n}: {D(t)} - maxdist <= 0")

    for i in I:                                                    # (8)
        n += 1
        out.append(f" c{n}: " + " ".join(f"+ {x(i,t)}" for t in T)
                   + f" - {z(i)} <= {p['R'][i]:.6f}")

    for t in T:                                                    # (4)
        n += 1
        dt = math.floor(days * p["A"][t]) if integer_days else days * p["A"][t]
        out.append(f" c{n}: " + " ".join(f"+ {y(i,t)}" for i in I)
                   + f" <= {p['M'][t] * dt:.6f}")

    for t in T:                                                    # (5)
        n += 1
        dt = math.floor(days * p["A"][t]) if integer_days else days * p["A"][t]
        out.append(f" c{n}: " + " ".join(
            f"+ {p['d'][i[0]]:.6f} {y(i,t)}" for i in I)
            + f" <= {p['L'][t] * dt * buffer:.6f}")

    if packing_cuts:                                               # (14)
        # A round trip longer than L_t/(k+1) cannot be made more than k times
        # in one day, so at most k*floor(D*A_t) such trips fit in the horizon.
        for t in T:
            dt = math.floor(days * p["A"][t])
            for k in range(1, min(kmax, int(p["M"][t])) + 1):
                thr = p["L"][t] / (k + 1)
                long_i = [i for i in I if p["d"][i[0]] > thr + 1e-9]
                if not long_i:
                    continue
                n += 1
                out.append(f" c{n}: " + " ".join(f"+ {y(i,t)}" for i in long_i)
                           + f" <= {k * dt}")

    out.append("Bounds")
    out += [f" {y(i,t)} >= 0" for i in I for t in T]
    out.append("General")
    out.append(" " + " ".join(y(i, t) for i in I for t in T))
    out.append("End")

    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")

    n_vars = len(T) * len(I) * 2 + 2 * len(T) + 1 + len(I)
    return n, n_vars


def solve(path, gap=0.10, relax=False, log=False, time_limit=None):
    """Solve the LP file with HiGHS. relax=True drops integrality."""
    import highspy

    if relax:
        # HiGHS picks the reader from the file extension, so the relaxed copy
        # must also end in .lp
        src = open(path).read()
        path = path[:-3] + "_relaxed.lp" if path.endswith(".lp") else path + "_relaxed.lp"
        with open(path, "w") as fh:
            fh.write(src[:src.index("General")] + "End\n")

    h = highspy.Highs()
    h.setOptionValue("output_flag", log)
    h.setOptionValue("mip_rel_gap", gap)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    h.readModel(path)
    h.run()

    info = h.getInfo()
    vals = dict(zip([h.getColName(j)[1] for j in range(h.getNumCol())],
                    h.getSolution().col_value))
    return dict(objective=h.getObjectiveValue(),
                bound=None if relax else info.mip_dual_bound,
                gap=None if relax else info.mip_gap,
                status=h.modelStatusToString(h.getModelStatus()),
                values=vals)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarise(values, p):
    """Pull the quantities the paper reports out of a solution vector."""
    T, I = p["T"], p["I"]
    ti = {t: k for k, t in enumerate(T)}
    ii = {i: k for k, i in enumerate(I)}
    trips = {t: sum(round(values.get(f"y_{ii[i]}_{ti[t]}", 0.0)) for i in I)
             for t in T}
    return dict(
        distance=sum(v for k, v in values.items() if k.startswith("dist_")),
        max_trailer=values.get("maxdist", 0.0),
        overfill=sum(v for k, v in values.items() if k.startswith("of_")),
        trips=sum(trips.values()),
        litres=sum(v for k, v in values.items() if k.startswith("x_")),
        trailers_used=sum(1 for t in T if trips[t] > 0),
    )


def trip_plan(values, p):
    """Round trips per (trailer, station-product) pair, as a DataFrame."""
    T, I = p["T"], p["I"]
    ti = {t: k for k, t in enumerate(T)}
    ii = {i: k for k, i in enumerate(I)}
    rows = []
    for i in I:
        for t in T:
            k = round(values.get(f"y_{ii[i]}_{ti[t]}", 0.0))
            if k:
                rows.append(dict(trailer=t, type=p["TYPE"][t], station=i[0],
                                 product=i[1], trips=k,
                                 litres=values.get(f"x_{ii[i]}_{ti[t]}", 0.0),
                                 km=k * p["d"][i[0]]))
    return pd.DataFrame(rows)
