"""Reproduce the computational results reported in the paper.

Run all of them::

    python experiments.py

or one at a time::

    python experiments.py history
    python experiments.py baseline
    python experiments.py bound
    python experiments.py utilization
    python experiments.py dayfeas
    python experiments.py sensitivity
    python experiments.py uncertainty
    python experiments.py scalability
    python experiments.py multiproduct
    python experiments.py depotfleet
    python experiments.py weights

Each block prints the figures that appear in the paper next to the section or
table they appear in. Total runtime is roughly fifteen minutes, dominated by
the sensitivity sweep, which solves the full MILP 21 times, and by the
scalability block, which solves a doubled network.
"""

from __future__ import annotations

import os
import random
import statistics as st
import sys
import tempfile
import time

import pandas as pd

import day_feasibility
import history as hist
import milp
import multiproduct

GAP = 0.10
TMP = tempfile.gettempdir()


def _solve(tag, dist_scale=None, demand_csv=None, trailer_csv=None,
           days=milp.DAYS, **kw):
    p = milp.load(trailer_csv=trailer_csv, demand_csv=demand_csv,
                  dist_scale=dist_scale, days=days)
    path = os.path.join(TMP, f"fd_{tag}.lp")
    ncon, nvar = milp.build_lp(p, path, **kw)
    t0 = time.time()
    r = milp.solve(path, gap=GAP)
    r["seconds"] = time.time() - t0
    return p, r, ncon, nvar


def _true_cost(values, p_true):
    """Evaluate a trip plan under the unperturbed road distances."""
    ti = {t: k for k, t in enumerate(p_true["T"])}
    ii = {i: k for k, i in enumerate(p_true["I"])}
    return sum(round(values.get(f"y_{ii[i]}_{ti[t]}", 0.0)) * p_true["d"][i[0]]
               for i in p_true["I"] for t in p_true["T"])


# --------------------------------------------------------------------------

PAPER_KM = 41281.80        # the reported plan, from baseline()
PRICE_USD_PER_GAL = 3.739
LITRES_PER_GAL = 3.785411784
CO2_KG_PER_L = 2.689


def history():
    print("== Historical baseline (Sections IV and V-A, Table III) ==")
    s = hist.summary()
    print(f"  records           {s['first']} to {s['last']}, {s['months']} months, "
          f"depot Source_2")
    print(f"  product lines     {s['lines']:,}, grouped into {s['visits']:,} "
          f"trailer visits")
    print(f"  multi-line visits {100*s['share_lines_in_multi']:.1f}% of lines; "
          f"{100*s['multi_distinct_grades']:.0f}% all distinct grades, "
          f"{100*s['multi_fits_one_load']:.0f}% fit one tank load")
    print(f"  per product line  {s['product_lines']:>10,.0f} km / 30 days "
          f"(not used: counts one visit several times)")
    print(f"  per visit         {s['physical_trips']:>10,.0f} km / 30 days "
          f"(the paper's baseline)")
    print(f"  reduction         {100*(1-PAPER_KM/s['physical_trips']):.1f}% "
          f"for the {PAPER_KM:,.0f} km plan")
    for eff in (30, 40):
        saved_l = (s["physical_trips"] - PAPER_KM) * eff / 100
        print(f"  at {eff} L/100 km    saves ${saved_l/LITRES_PER_GAL*PRICE_USD_PER_GAL:,.0f} "
              f"and {saved_l*CO2_KG_PER_L/1000:.1f} t CO2 a month")
    vol = s["litres_per_30_days"]
    print(f"  volume            {vol/1e6:.2f} ML / 30 days; plan moves "
          f"{100*(15800165.45/vol-1):.1f}% more")
    print(f"  per 1,000 L       {s['physical_trips']/(vol/1000):.2f} km historically, "
          f"{PAPER_KM/15800.16545:.2f} in the plan, "
          f"{100*(1-(PAPER_KM/15800.16545)/(s['physical_trips']/(vol/1000))):.1f}% less")
    print(f"  historical trips  {s['trips_per_month']:.0f} a month, "
          f"{100*s['load_utilization']:.1f}% load utilization")
    print(f"  busiest trailer   {s['busiest_trips']:.0f} trips and "
          f"{s['busiest_km']:,.0f} km a month on average")
    print("  paper: 50,832 km, 18.8%; $2,830-$3,773 and 7.7-10.3 t CO2;")
    print("         725 trips, about 55% full; busiest 57 trips, 4,168 km;")
    print("         14.85 ML, 6% more volume, 3.42 vs 2.61 km per 1,000 L, 23.7%")


def baseline():
    print("== Baseline MILP (Sections IV-A and V-A) ==")
    p, r, ncon, nvar = _solve("base")
    s = milp.summarise(r["values"], p)
    print(f"  instance          {ncon} constraints, {nvar} variables, "
          f"{len(p['T']) * len(p['I'])} integer")
    print(f"  status            {r['status']} in {r['seconds']:.1f}s")
    print(f"  total distance    {s['distance']:>12,.2f} km")
    print(f"  trips             {s['trips']:>12,.0f}")
    print(f"  trailers used     {s['trailers_used']:>12,d}")
    print(f"  litres delivered  {s['litres']:>12,.2f}")
    print(f"  busiest trailer   {s['max_trailer']:>12,.2f} km, "
          f"{s['max_trips']} trips")
    print(f"  objective         {r['objective']:>12,.2f}")
    print(f"  dual bound        {r['bound']:>12,.2f}   gap {r['gap']*100:.2f}%")
    print("  paper: 2,809 constraints / 4,829 variables / 2,380 integer;")
    print("         41,281.80 km, 577 trips, 29 trailers, 15.80 ML")
    return p, r


def bound():
    print("== LP relaxation lower bound (Section V-A) ==")
    p = milp.load()
    path = os.path.join(TMP, "fd_bound.lp")
    milp.build_lp(p, path)
    r = milp.solve(path, relax=True)
    dist = sum(v for k, v in r["values"].items() if k.startswith("dist_"))
    print(f"  LP objective      {r['objective']:>12,.2f}")
    print(f"  distance part     {dist:>12,.2f} km")
    print("  paper: bound 39,660.68, of which 38,917.41 km is distance;")
    print("         incumbent objective 42,208.47 gives the 6.04% gap")


def utilization():
    print("== Fleet and station utilization (Section V-C, Table IV) ==")
    p, r, _, _ = _solve("util")
    plan = milp.trip_plan(r["values"], p)
    plan["capacity"] = plan.trailer.map(p["C"])
    tr = plan.groupby(["trailer", "type"]).agg(
        trips=("trips", "sum"), litres=("litres", "sum"),
        km=("km", "sum"), capacity=("capacity", "first")).reset_index()
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

    tr["lu"] = 100 * tr.litres / (tr.trips * tr.capacity)
    tr["ki"] = tr.km / (tr.litres / 1000)
    print(f"    per trailer: load util {tr.lu.min():.1f}-{tr.lu.max():.1f}% "
          f"(median {tr.lu.median():.1f}), km/1,000 L {tr.ki.min():.2f}-"
          f"{tr.ki.max():.2f} (median {tr.ki.median():.2f})")

    sd = plan.groupby("station").agg(litres=("litres", "sum"),
                                     km=("km", "sum"),
                                     trips=("trips", "sum")).reset_index()
    sd["km_per_1000L"] = sd.km / (sd.litres / 1000)
    sd = sd.sort_values("km", ascending=False)
    print(f"    km per 1,000 L across stations: "
          f"{sd.km_per_1000L.min():.2f} to {sd.km_per_1000L.max():.2f}")
    print(f"    five costliest stations account for "
          f"{100 * sd.head(5).km.sum() / sd.km.sum():.0f}% of mileage")
    worst = sd.sort_values("km_per_1000L", ascending=False).iloc[0]
    print(f"    costliest station {worst.station}: {worst.trips:.0f} trips, "
          f"{worst.km_per_1000L:.2f} km per 1,000 L")
    print("  paper: large 2.24, medium 3.21, small 10.23; fleet 80.5% at 2.61")

    # the Julia reference solution, for cross-checking the two implementations
    out = pd.read_csv(os.path.join(milp.DATA, "model_output.csv"))
    print(f"    (Julia reference run in data/processed: "
          f"{out.Distance_Travelled_km.sum():,.0f} km)")


def dayfeas():
    print("== Day-level feasibility (Section V-D) ==")
    for label, kw in (
            ("as reported (whole days, buffer, packing inequalities)", {}),
            ("without them (fractional days, no buffer, no inequalities)",
             dict(integer_days=False, packing_cuts=False, buffer=1.0))):
        p, r, ncon, _ = _solve("df", **kw)
        s = milp.summarise(r["values"], p)
        df = pd.DataFrame(day_feasibility.check(r["values"], p))
        ok = int(df.feasible.sum())
        print(f"  {label}")
        print(f"    distance        {s['distance']:>12,.2f} km   "
              f"({ncon} constraints)")
        print(f"    day-feasible    {ok} of {len(df)} trailers")
        for _, x in df[~df.feasible].iterrows():
            print(f"      {x.trailer} ({x.type}): {x.trips} trips, "
                  f"{x.km:,.0f} km, needs {x.min_days} of "
                  f"{x.available_days} days")
        print(f"    operating days  {int(df.min_days.sum())} across the fleet")
    print("  paper: 29 of 29 at 41,282 km; 30 of 31 and 41,381 km without")


def sensitivity():
    print("== Distance sensitivity (Section V-E) ==")
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
    print("  paper: every variant within 0.09% of the baseline plan")


def uncertainty():
    print("== Demand uncertainty (Section V-F) ==")
    p = milp.load()
    dem = pd.read_csv(os.path.join(milp.DATA, "station_demand_julia.csv"))
    _, r, _, _ = _solve("unc_base")
    s = milp.summarise(r["values"], p)
    ti = {t: k for k, t in enumerate(p["T"])}
    ii = {i: k for k, i in enumerate(p["I"])}
    delivered = {i: sum(r["values"].get(f"x_{ii[i]}_{ti[t]}", 0.0)
                        for t in p["T"]) for i in p["I"]}
    total_q = sum(p["Q"].values())
    print(f"  mean-demand plan  {s['distance']:>12,.2f} km")

    for e in (0.10, 0.20):
        dists, shorts, pairs = [], [], []
        for seed in range(8):
            rnd = random.Random(4000 + 100 * seed + int(e * 100))
            f = {i: 1 + rnd.uniform(-e, e) for i in p["I"]}
            shorts.append(sum(max(0.0, p["Q"][i] * f[i] - delivered[i])
                              for i in p["I"]))
            pairs.append(sum(1 for i in p["I"]
                             if p["Q"][i] * f[i] - delivered[i] > 1e-6))
            d2 = dem.copy()
            d2["Monthly_Demand_Liters"] = [
                d2.Monthly_Demand_Liters[k]
                * f[(d2.Destination_ID[k], d2.Product[k])]
                for k in range(len(d2))]
            path = os.path.join(TMP, f"unc_{seed}_{int(e*100)}.csv")
            d2.to_csv(path, index=False)
            p2, r2, _, _ = _solve(f"unc{seed}_{int(e*100)}", demand_csv=path)
            dists.append(milp.summarise(r2["values"], p2)["distance"])
        print(f"  +/-{e:.0%} demand    re-solved {st.mean(dists):>10,.0f} km "
              f"+/- {st.pstdev(dists):,.0f}  "
              f"({100*(st.mean(dists)/s['distance']-1):+.2f}%)")
        print(f"                    mean-demand plan leaves "
              f"{st.mean(shorts):,.0f} L unserved = "
              f"{100*st.mean(shorts)/total_q:.2f}% of volume on "
              f"{st.mean(pairs):.1f} of {len(p['I'])} pairs")
    print("  paper: +1.31% and -0.05% distance; 2.52% and 5.05% unserved")


def scalability():
    print("== Instance size and horizon (Section V-F) ==")
    dem = pd.read_csv(os.path.join(milp.DATA, "station_demand_julia.csv"))
    truck = pd.read_csv(os.path.join(milp.DATA, "truck_config_julia.csv"))

    for days in (30, 60, 90):
        d = dem.copy()
        d["Monthly_Demand_Liters"] = d.Monthly_Demand_Liters * days / 30.0
        path = os.path.join(TMP, f"hz_{days}.csv")
        d.to_csv(path, index=False)
        p, r, ncon, nvar = _solve(f"hz{days}", demand_csv=path, days=days)
        s = milp.summarise(r["values"], p)
        print(f"  {days:>3}-day horizon  {ncon} con / {nvar} var   "
              f"{r['seconds']:>6.1f}s   {s['distance']:>10,.0f} km   "
              f"gap {r['gap']*100:.2f}%")

    rnd = random.Random(9)
    for mult in (2,):
        ds, ts = [], []
        for c in range(mult):
            d = dem.copy()
            d["Destination_ID"] = d.Destination_ID.astype(str) + f"_c{c}"
            if c:
                d["Distance_km"] = d.Distance_km * (1 + rnd.uniform(-0.3, 0.3))
            ds.append(d)
            t = truck.copy()
            t["Trailer"] = t.Trailer.astype(str) + f"_c{c}"
            ts.append(t)
        dp = os.path.join(TMP, f"sc_d{mult}.csv")
        tp = os.path.join(TMP, f"sc_t{mult}.csv")
        pd.concat(ds).to_csv(dp, index=False)
        pd.concat(ts).to_csv(tp, index=False)
        p, r, ncon, nvar = _solve(f"sc{mult}", demand_csv=dp, trailer_csv=tp)
        s = milp.summarise(r["values"], p)
        print(f"  {mult}x network      {ncon} con / {nvar} var   "
              f"{r['seconds']:>6.1f}s   {s['distance']:>10,.0f} km   "
              f"gap {r['gap']*100:.2f}%   "
              f"({len(p['I'])} pairs, {len(p['T'])} trailers)")
    print("  paper: 5 s and 19 s for 60 and 90 days; 4 s to 203 s for 2x")


def depot_fleet():
    print("== Trailers based at other depots (Section IV) ==")
    p = milp.load()
    served = set(hist.load_records().Trailer.unique())
    other = [t for t in p["T"] if t not in served]
    _, r, _, _ = _solve("fleet_all")
    plan = milp.trip_plan(r["values"], p)
    used = plan[plan.trailer.isin(other)]
    print(f"  eligible trailers {len(p['T'])}, of which {len(other)} never served "
          f"the principal depot")
    print(f"  used in the plan  {used.trailer.nunique()} of them, "
          f"{100*used.litres.sum()/plan.litres.sum():.1f}% of volume")
    p2 = dict(p, T=[t for t in p["T"] if t in served])
    path = os.path.join(TMP, "fd_fleet_s2.lp")
    milp.build_lp(p2, path)
    r2 = milp.solve(path, gap=GAP)
    s2 = milp.summarise(r2["values"], p2)
    rows = day_feasibility.check(r2["values"], p2, compute_min_days=False)
    print(f"  principal-depot fleet only ({len(p2['T'])} trailers): "
          f"{s2['distance']:,.2f} km ({100*(s2['distance']/PAPER_KM-1):+.2f}%), "
          f"day-feasible {sum(x['feasible'] for x in rows)} of {len(rows)}")
    print("  paper: 13 trailers, 8 used for 9.4% of volume; 21-trailer fleet")
    print("         +0.3%, two trailers needing an extra day")


def multiproduct_variant():
    print("== Multi-product trips (Section V-F) ==")
    p = milp.load()
    path = os.path.join(TMP, "fd_multiproduct.lp")
    ncon, nvar, nint = multiproduct.build_lp(p, path)
    t0 = time.time()
    r = milp.solve(path, gap=GAP)
    secs = time.time() - t0
    lp = milp.solve(path, relax=True)
    s = multiproduct.summarise(r["values"], p)
    rows = day_feasibility.check(multiproduct.as_pair_trips(r["values"], p), p,
                                 compute_min_days=False)
    base = hist.summary()["physical_trips"]
    print(f"  instance          {ncon} constraints, {nvar} variables, {nint} integer")
    print(f"  status            {r['status']} in {secs:.1f}s")
    print(f"  total distance    {s['distance']:>12,.0f} km "
          f"({100*(1-s['distance']/base):.1f}% below history)")
    print(f"  trips / trailers  {s['trips']} / {s['trailers_used']}")
    print(f"  load utilization  {100*s['load_utilization']:.1f}% "
          f"(optimistic: compartments split freely)")
    print(f"  proven gap        "
          f"{100*(r['objective']-lp['objective'])/r['objective']:.2f}%")
    print(f"  day-feasible      {sum(x['feasible'] for x in rows)} of {len(rows)}")
    print("  paper: 29,393 km on 414 trips, 42.2% below history, 2.83%, all")
    print("         trailers day-feasible, 96% load utilization")


def weights():
    print("== Imbalance weight (Section III-D) ==")
    for lam in (0.0, 0.1, 0.5, 1.0, 2.0):
        p, r, _, _ = _solve(f"w_{lam}", lam=lam)
        s = milp.summarise(r["values"], p)
        print(f"  lambda = {lam:<4} distance {s['distance']:>11,.2f} km   "
              f"busiest trailer {s['max_trailer']:>8,.1f} km   "
              f"trailers {s['trailers_used']:>3d}")
    print("  paper: 0 gives 7,033 km; 0.1, 0.5, 1.0, 2.0 give 2,814, 1,853,")
    print("         1,884, 1,545 km, with distance varying by about 0.5%")


ALL = dict(history=history, baseline=baseline, bound=bound,
           utilization=utilization, dayfeas=dayfeas, sensitivity=sensitivity,
           uncertainty=uncertainty, scalability=scalability,
           multiproduct=multiproduct_variant, depotfleet=depot_fleet,
           weights=weights)

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ALL)
    for name in wanted:
        if name not in ALL:
            sys.exit(f"unknown experiment {name!r}; choose from {list(ALL)}")
        ALL[name]()
        print()
