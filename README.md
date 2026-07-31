# Fuel Delivery Optimization in Wholesale Petroleum Logistics

[![DOI](https://zenodo.org/badge/1318744146.svg)](https://doi.org/10.5281/zenodo.21728160)

Model, data and experiments for a trailer-allocation MILP that schedules
multi-product fuel deliveries from a depot to retail stations at minimum
transportation cost.

Given a month of historical deliveries, the model decides how many round
trips each trailer makes to each station-product pair and how much it carries,
subject to tanker capacity, station storage, per-trailer daily trip and
distance limits, and an availability factor estimated from the historical
record. On the dataset here it cuts total distance from 74,275 km to
41,398 km, a 44.3 percent reduction, and is provably within 6.08 percent of
the linear-programming lower bound.

This repository accompanies a conference paper and an MSc thesis at The
University of the West Indies, St. Augustine.

## Layout

    code/                   Julia reference model and the data-preparation notebooks
      model_julia.jl        the MILP, in JuMP, solved with HiGHS
      model_julia_simple.jl the relaxed LP benchmark
      *.ipynb               anonymisation, exploration, figures, comparison
      python/               second implementation and the paper's experiments
    data/
      raw/                  anonymised delivery records
      processed/            model inputs and the reference solution
      synthetic/            a small synthetic instance for verification
    figures/                figures used in the thesis and the paper

## Reproducing the results

Everything reported in the paper can be regenerated from `code/python`, which
reimplements the same formulation and adds the experiments the Julia model
does not perform on its own: the LP relaxation bound, the day-level
feasibility certificate, the distance sensitivity study and the penalty-weight
calibration.

```bash
cd code/python
pip install -r requirements.txt
python experiments.py          # all six blocks, a few minutes
python run_synthetic.py        # the small instance, under a second
```

Each block prints the paper's figures next to the section or table they
appear in, so a discrepancy is visible immediately. See
[`code/python/README.md`](code/python/README.md) for what each script does
and for notes on reproducing exactly.

If you would rather verify the formulation without touching the real data,
`data/synthetic/` holds a five-trailer, six-station instance that solves in a
fraction of a second and is documented in
[`data/synthetic/README.md`](data/synthetic/README.md).

## A note on the data

The delivery records in `data/raw/` are anonymised. Depots, stations,
tractors and trailers are relabelled, product names are standardised, and
free-text and location fields are removed. Geographic coordinates were used
only to retrieve one-way road distances between each depot-station pair and
to render the demand map; they are not included here, and the distances
themselves are what the model consumes.

The identified source data is commercially confidential and is not part of
this repository. `code/create_fuel_raw_data_anon.ipynb` documents the
anonymisation procedure but cannot be run without that input, which is
intentional.

## Citing

Please cite the archived release rather than this repository directly, since
the DOI resolves to a fixed snapshot of the code that produced the published
results:

> D. Padmore, "Fuel delivery optimization in wholesale petroleum logistics,"
> software repository, 2026, doi: 10.5281/zenodo.21728160.

## Licence

MIT, see [`LICENSE`](LICENSE). The anonymised datasets under `data/` are
released under CC BY 4.0.
