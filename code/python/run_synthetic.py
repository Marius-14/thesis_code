"""Solve the small synthetic instance and check it at day level.

The synthetic instance reproduces the structure of the real data at a size
anyone can inspect by hand: 5 trailers across the three size classes, 6
stations, 12 station-product pairs. It exists so the formulation can be
verified without the anonymised production data.

    python run_synthetic.py

Expected output: 7,329.20 km over 95 trips, zero overfill, and all five
trailers day-feasible.
"""

from __future__ import annotations

import os
import tempfile
import time

import day_feasibility
import milp

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(HERE, "..", "..", "data", "synthetic")

if __name__ == "__main__":
    p = milp.load(
        trailer_csv=os.path.join(INST, "truck_config_synthetic.csv"),
        demand_csv=os.path.join(INST, "station_demand_synthetic.csv"),
    )
    path = os.path.join(tempfile.gettempdir(), "synthetic.lp")
    ncon, nvar = milp.build_lp(p, path)

    t0 = time.time()
    r = milp.solve(path, gap=0.10)
    elapsed = time.time() - t0
    s = milp.summarise(r["values"], p)

    print(f"{len(p['T'])} trailers, {len(p['I'])} station-product pairs, "
          f"{ncon} constraints, {nvar} variables")
    print(f"  status      {r['status']}")
    print(f"  distance    {s['distance']:>10,.2f} km")
    print(f"  trips       {s['trips']:>10,.0f}")
    print(f"  litres      {s['litres']:>10,.2f}")
    print(f"  overfill    {s['overfill']:>10,.2f} L")
    print(f"  objective   {r['objective']:>10,.2f}   solved in {elapsed:.2f}s")

    rows = day_feasibility.check(r["values"], p)
    ok = sum(1 for x in rows if x["feasible"])
    print(f"\nday-level feasibility: {ok} of {len(rows)} trailers")
    for x in rows:
        print(f"  {x['trailer']:<12}{x['trips']:>3} trips {x['km']:>10,.1f} km"
              f"   needs {x['min_days']} of {x['available_days']} days"
              f"   {'ok' if x['feasible'] else 'INFEASIBLE'}")
