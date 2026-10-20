# Python implementation and paper experiments

A second implementation, in Python, of the trailer-allocation MILP in
`../model_julia.jl`. Both build the same formulation and are solved with
HiGHS; this one goes through the CPLEX LP file format and the `highspy`
bindings rather than JuMP.

It exists because the paper reports results the Julia model does not produce
on its own: the LP relaxation bound, the day-level feasibility certificate,
the distance and demand robustness studies, the scalability sweep and the
imbalance-weight calibration. Having two builds of the same specification
also cross-checks the reference run. Every figure reported in Section V of
the paper comes from this implementation, so the baseline, the feasibility
certificate and the robustness studies are all drawn from one solution.

The instance has 2,809 constraints and 4,829 variables, of which 2,380 are
integer, and solves in under five seconds at a 10 percent relative MIP gap.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9 or later. No solver installation is needed; `highspy`
ships HiGHS.

## Run

```bash
cd code/python
python experiments.py              # everything, about fifteen minutes
python experiments.py baseline     # or one block at a time
python run_synthetic.py            # the small instance, under a second
```

Each block prints the figures that appear in the paper alongside the section
or table they appear in, so a discrepancy is visible immediately.

| Command | Reproduces |
| --- | --- |
| `experiments.py history` | historical baseline, fuel and CO2 savings, Section V-A and Table III |
| `experiments.py baseline` | instance size and headline plan, Sections IV-A and V-A |
| `experiments.py bound` | LP relaxation lower bound, Section V-A |
| `experiments.py utilization` | fleet and station utilization, Table IV |
| `experiments.py dayfeas` | day-level feasibility, Section V-D |
| `experiments.py sensitivity` | distance robustness, Section V-E |
| `experiments.py uncertainty` | demand uncertainty, Section V-F |
| `experiments.py scalability` | horizon and network size, Section V-F |
| `experiments.py multiproduct` | multi-product trips, Section V-F |
| `experiments.py depotfleet` | trailers based at other depots, Section IV |
| `experiments.py weights` | imbalance-weight calibration, Section III-D |
| `run_synthetic.py` | the synthetic instance in `data/synthetic/` |

## Files

`milp.py` builds and solves the model. `load()` reads the two input tables
from `data/processed/`, `build_lp()` writes the formulation, `solve()` calls
HiGHS. Its defaults are the model as reported in the paper: whole operating
days `N_t = floor(D * A_t)`, the daily-distance buffer `rho = 0.95`, the
daily-packing inequalities, and the imbalance weight `lam = 0.5`. The flags
`integer_days` and `packing_cuts` exist so `experiments.py dayfeas` can
re-solve the weaker variant and show what the discrete-day treatment is
doing; `dist_scale` perturbs road distances for the sensitivity study.

`day_feasibility.py` tests whether a monthly plan decomposes into whole
operating days. Because trips to the same station are interchangeable, the
question reduces to an integer feasibility problem over the distinct
round-trip distances, which is solved exactly. A positive answer is a
constructive proof rather than an estimate. It also reports the smallest
number of operating days each trailer's assignment can be compressed into.

`history.py` computes the historical baseline from the anonymised delivery
records in `data/raw/`. The records list one line per product, so a trailer
that delivers three grades on one visit appears three times; the baseline
counts that visit once, because fuel is burned per kilometre driven.

`multiproduct.py` is the variant of Section V-F that indexes trips by
station rather than by station-product pair, so one trip can carry several
grades. The data has no compartment sizes, so it splits each tank freely and
is an optimistic upper bracket rather than a plan.

`experiments.py` runs the eleven blocks above.

`run_synthetic.py` solves the small released instance, whose two input
tables live in `data/synthetic/` alongside the real ones.

## Notes on reproducing exactly

The model is solved to a 10 percent relative MIP gap, so different solver
versions and platforms may return different incumbents of near-identical
objective value. Total distance is stable to a few tenths of a percent;
per-trailer assignments are less so, because the objective is nearly
indifferent to which trailer serves a given station. That degeneracy is what
the imbalance penalty exists to break, and `experiments.py weights` shows
what happens when the weight is too small: at `lam = 0.1` the imbalance term
is worth less than the gap tolerance itself and the busiest trailer drifts to
2,814 km, against 1,853 km at `lam = 0.5`.

`data/processed/model_output.csv` holds the Julia reference run of the
earlier, pre-hardening formulation and is retained for comparison only; the
paper's tables come from this implementation.

The sensitivity block solves the full MILP 21 times, and the scalability
block solves a doubled network that takes three to four minutes. Together they
account for most of the runtime.
