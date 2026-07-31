# Python implementation and paper experiments

A second implementation, in Python, of the trailer-allocation MILP in
`../model_julia.jl`. Both build the same formulation and are solved with
HiGHS; this one goes through the CPLEX LP file format and the `highspy`
bindings rather than JuMP.

It exists because the paper reports results that the Julia model does not
produce on its own: the LP relaxation bound, the day-level feasibility
certificate, the distance sensitivity study and the penalty-weight
calibration. Having two builds of the same specification also cross-checks
the reference run. They agree on the instance dimensions exactly (2,690
constraints, 4,899 variables, 2,380 integer), on the LP relaxation bound to
within 0.01, and on total distance to 0.2 percent (41,323 km here against
41,398 km in Julia, both admissible incumbents under the 10 percent relative
MIP gap, both with 579 trips, 28 trailers, 15.80 ML delivered and no
overfill).

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9 or later. No solver installation is needed; `highspy`
ships HiGHS.

## Run

```bash
cd code/python
python experiments.py              # everything, a few minutes
python experiments.py baseline     # or one block at a time
python run_synthetic.py            # the small instance, under a second
```

Each block prints the figures that appear in the paper alongside the section
or table they appear in, so a discrepancy is visible immediately.

| Command | Reproduces |
| --- | --- |
| `experiments.py baseline` | instance size and headline plan, Sections IV-A and V-A |
| `experiments.py bound` | LP relaxation lower bound, Table III |
| `experiments.py dayfeas` | day-level feasibility, Section V-D |
| `experiments.py sensitivity` | distance robustness, Table VI |
| `experiments.py weights` | penalty-weight calibration, Section III-D |
| `experiments.py utilization` | fleet and station utilization, Table V |
| `run_synthetic.py` | the synthetic instance released with the paper |

## Files

`milp.py` builds and solves the model. `load()` reads the two input tables
from `data/processed/`, `build_lp()` writes the formulation, `solve()` calls
HiGHS. The options `integer_days`, `packing_cuts` and `buffer` produce the
hardened variant of Section V-D; `dist_scale` perturbs road distances for the
sensitivity study; `lambda_1` and `lambda_2` set the penalty weights.

`day_feasibility.py` tests whether a monthly plan decomposes into whole
operating days. Because trips to the same station are interchangeable, the
question reduces to an integer feasibility problem over the distinct
round-trip distances, which is solved exactly. A positive answer is a
constructive proof rather than an estimate. It also reports the smallest
number of operating days each trailer's assignment can be compressed into.

`experiments.py` runs the six blocks above.

`run_synthetic.py` and `synthetic/` are the small released instance.

## Notes on reproducing exactly

The model is solved to a 10 percent relative MIP gap, so different solver
versions and platforms may return different incumbents of near-identical
objective value. Total distance is stable to a few tenths of a percent;
per-trailer assignments are not, because the objective is nearly indifferent
to which trailer serves a given station. This is the degeneracy that the
imbalance penalty exists to break, and it is why `experiments.py utilization`
prints Table V from the reference solution in `data/processed/model_output.csv`
as well as from this implementation's own incumbent.

The sensitivity block solves the full MILP 21 times and takes the bulk of the
runtime.
