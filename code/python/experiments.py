"""Reproduce the computational results reported in the paper.

Run all of them::

    python experiments.py

or one at a time::

    python experiments.py baseline
    python experiments.py bound
    python experiments.py dayfeas
    python experiments.py sensitivity
    python experiments.py weights
    python experiments.py utilization

Each block prints the figures that appear in the paper next to the section or
table they appear in. Total runtime is a few minutes, dominated by the
sensitivity sweep, which solves the full MILP 21 times.
"""

from __future__ import annotations

import os
import random
import statistics as st
import sys
import tempfile

import pandas as pd

import day_feasibility
import milp

GAP = 0.10
TMP = tempfile.gettempdir()


def _solve(tag, **kw):
    p = milp.load(dist_scale=kw.pop("dist_scale", None))
    path = os.path.join(TMP, f"fd_{tag}.lp")
    ncon, nvar = milp.build_lp(p, path, **kw)
    r = milp.solve(path, gap=GAP)
    return p, r, ncon, nvar


def _true_cost(values, p_true):
    """Evaluate a trip plan under the unperturbed road distances."""
    ti = {t: k for k, t in enumerate(p_true["T"])}
    ii = {i: k for k, i in enumerate(p_true["I"])}
    return sum(round(values.get(f"y_{ii[i]}_{ti[t]}", 0.0)) * p_true["d"][i[0]]
               for i in p_true["I"] for t in p_true["T"])


# --------------------------------------------------------------------------

def baseline():
    print("== Baseline MILP (Sections IV-A and V-A) ==")
    p, r, ncon, nvar = _solve("base")
    s = milp.summarise(r["values"], p)
    print(f"  instance          {ncon} constraints, {nvar} variables, "
          f"{len(p['T']) * len(p['I'])} integer")
    print(f"  status            {r['status']}")
    print(f"  total distance    {s['distance']:>12,.2f} km")
    print(f"  trips             {s['trips']:>12,.0f}")
    print(f"  trailers used     {s['trailers_used']:>12,d}")
    print(f"  litres delivered  {s['litres']:>12,.2f}")
    print(f"  max trailer       {s['max_trailer']:>12,.2f} km")
    print(f"  overfill          {s['overfill']:>12,.4f} L")
    print(f"  objective         {r['objective']:>12,.2f}")
    print(f"  dual bound        {r['bound']:>12,.2f}   gap {r['gap']*100:.2f}%")
    print("  paper: 2,690 constraints / 4,899 variables / 2,380 integer;")
    print("         41,398 km (Julia reference) vs 41,323 km here, 0.2% apart")
    return p, r


def bound():
    print("== LP relaxation lower bound (Section V-A, Table III) ==")
    p = milp.load()
    path = os.path.join(TMP, "fd_bound.lp")
    milp.build_lp(p, path)
    r = milp.solve(path, relax=True)
    dist = sum(v for k, v in r["values"].items() if k.startswith("dist_"))
    print(f"  LP objective      {r['objective']:>12,.2f}")
    print(f"  distance part     {dist:>12,.2f} km")
    print("  paper: bound 39,056.35, of which 38,902.30 km is distance;")
    print("         incumbent objective 41,583.89 gives the 6.08% gap")


def dayfeas():
    print("== Day-level feasibility (Section V-D) ==")
    for label, kw in (("as formulated", {}),
                      ("integer days + packing cuts + 5% buffer",
                       dict(integer_days=True, packing_cuts=True,
                            buffer=0.95))):
        p, r, ncon, _ = _solve("df", **kw)
        s = milp.summarise(r["values"], p)
        rows = day_feasibility.check(r["values"], p)
        df = pd.DataFrame(rows)
        ok = int(df.feasible.sum())
        print(f"  {label}")
        print(f"    distance        {s['distance']:>12,.2f} km   "
              f"({ncon} constraints)")
        print(f"    day-feasible    {ok} of {len(df)} trailers")
        if ok < len(df):
            for _, x in df[~df.feasible].iterrows():
                print(f"      {x.trailer} ({x.type}): {x.trips} trips, "
                      f"{x.km:,.0f} km, needs {x.min_days} of "
                      f"{x.available_days} days")
        print(f"    operating days  {int(df.min_days.sum())} across the fleet")
    print("  paper: 27 of 28 as formulated, 28 of 28 hardened at 41,247 km")


def sensitivity():
    print("== Distance sensitivity (Section V-E, Table VI) ==")
    p_true = milp.load()
    stations = sorted({i[0] for i in p_true["I"]})
    _, r, _, _ = _solve("sens_base")
    base = _true_cost(r["values"], p_true)
    print(f"  baseline          {base:>12,.0f} km")

    for f in (0.80, 1.20):
        _, r, _, _ = _solve(f"sens_u{int(f*100)}",
                            dist_scale={s: f for s in stations})
        ac = _true_cost(r["values"], p_true)
        print(f"  all x{f:<4.2f}        {ac:>12,.0f} km   "
              f"{100*(ac/base-1):+.2f}%")

    for e in (0.10, 0.20):
        res = []
        for seed in range(8):
            rnd = random.Random(1000 * seed + int(e * 100))
            sm = {s: 1 + rnd.uniform(-e, e) for s in stations}
            _, r, _, _ = _solve(f"sens_r{seed}", dist_scale=sm)
            res.append(_true_cost(r["values"], p_true))
        print(f"  random +/-{e:.0%}      {st.mean(res):>12,.0f} km "
              f"+/- {st.pstdev(res):,.0f}   {100*(st.mean(res)/base-1):+.2f}%")

    res = []
    for seed in range(8):
        rnd = random.Random(77 + seed)
        sm = {s: 1.0 / rnd.uniform(1.2, 1.6) for s in stations}
        _, r, _, _ = _solve(f"sens_p{seed}", dist_scale=sm)
        res.append(_true_cost(r["values"], p_true))
    print(f"  straight-line     {st.mean(res):>12,.0f} km "
          f"+/- {st.pstdev(res):,.0f}   {100*(st.mean(res)/base-1):+.2f}%")
    print("  paper: every variant within 0.06% of the baseline plan")


def weights():
    print("== Penalty weights (Section III-D) ==")
    for l1 in (0.0, 0.1, 1.0):
        p, r, _, _ = _solve(f"w1_{l1}", lambda_1=l1)
        s = milp.summarise(r["values"], p)
        print(f"  lambda_1 = {l1:<4} distance {s['distance']:>11,.2f} km   "
              f"busiest trailer {s['max_trailer']:>8,.1f} km")
    for l2 in (1.0, 5.0, 20.0, 50.0):
        p, r, _, _ = _solve(f"w2_{l2}", lambda_2=l2)
        s = milp.summarise(r["values"], p)
        print(f"  lambda_2 = {l2:<4} overfill {s['overfill']:>11,.4f} L    "
              f"distance {s['distance']:>11,.2f} km")
    print("  paper: dropping lambda_1 costs 0.6% distance and triples the")
    print("         busiest trailer; overfill is zero for every lambda_2")


def utilization():
    """Table V is computed from the Julia reference solution, which is what
    the paper reports. The Python incumbent is a different but equivalent
    solution inside the same 10% gap, so its per-class split differs slightly;
    both are printed."""
    print("== Fleet and station utilization (Section V-C, Table V) ==")

    def report(plan, cap_col, label):
        tr = plan.groupby(["trailer", "type"]).agg(
            trips=("trips", "sum"), litres=("litres", "sum"),
            km=("km", "sum"), capacity=(cap_col, "first")).reset_index()
        print(f"  {label}")
        print(f"    {'class':<8}{'n':>4}{'trips':>8}{'km':>12}"
              f"{'load util':>12}{'km/1000L':>11}")
        for cls in ("Large", "Medium", "Small"):
            g = tr[tr.type == cls]
            if g.empty:
                continue
            util = 100 * g.litres.sum() / (g.trips * g.capacity).sum()
            print(f"    {cls:<8}{len(g):>4}{g.trips.sum():>8,.0f}"
                  f"{g.km.sum():>12,.0f}{util:>11.1f}%"
                  f"{g.km.sum()/(g.litres.sum()/1000):>11.2f}")
        util = 100 * tr.litres.sum() / (tr.trips * tr.capacity).sum()
        print(f"    {'fleet':<8}{len(tr):>4}{tr.trips.sum():>8,.0f}"
              f"{tr.km.sum():>12,.0f}{util:>11.1f}%"
              f"{tr.km.sum()/(tr.litres.sum()/1000):>11.2f}")
        return tr

    # --- reference solution, as reported in the paper ---
    out = pd.read_csv(os.path.join(milp.DATA, "model_output.csv"))
    cfg = pd.read_csv(os.path.join(milp.DATA, "truck_config_julia.csv"))
    ref = out.merge(cfg[["Trailer", "Truck_Tank_Capacity_Liters",
                         "Truck_Type"]], on="Trailer")
    ref = ref.rename(columns={"Trailer": "trailer", "Truck_Type": "type",
                              "Destination_ID": "station", "Trips": "trips",
                              "Liters_Delivered": "litres",
                              "Distance_Travelled_km": "km",
                              "Truck_Tank_Capacity_Liters": "capacity"})
    report(ref, "capacity", "Julia reference run (Table V in the paper)")

    sd = ref.groupby("station").agg(litres=("litres", "sum"),
                                    km=("km", "sum")).reset_index()
    sd["km_per_1000L"] = sd.km / (sd.litres / 1000)
    sd = sd.sort_values("km", ascending=False)
    print(f"    km per 1,000 L across stations: "
          f"{sd.km_per_1000L.min():.2f} to {sd.km_per_1000L.max():.2f}")
    print(f"    five costliest stations account for "
          f"{100 * sd.head(5).km.sum() / sd.km.sum():.0f}% of mileage")

    # --- this implementation's own incumbent ---
    p, r, _, _ = _solve("util")
    plan = milp.trip_plan(r["values"], p)
    plan["capacity"] = plan.trailer.map(p["C"])
    report(plan, "capacity", "this implementation")


ALL = dict(baseline=baseline, bound=bound, dayfeas=dayfeas,
           sensitivity=sensitivity, weights=weights, utilization=utilization)

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ALL)
    for name in wanted:
        if name not in ALL:
            sys.exit(f"unknown experiment {name!r}; choose from {list(ALL)}")
        ALL[name]()
        print()
