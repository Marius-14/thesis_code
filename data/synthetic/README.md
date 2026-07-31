# Synthetic instance

A small, fully synthetic instance that reproduces the structure of the real
wholesale-fuel dataset at a size that can be checked by hand. It exists so
the formulation can be verified without the anonymised production data.

| | real instance | synthetic instance |
| --- | --- | --- |
| trailers (after the availability filter) | 34 | 5 |
| station-product pairs | 70 | 12 |
| stations / products | 25 / 3 | 6 / 3 |
| constraints / variables | 2,690 / 4,899 | 109 / 143 |
| solve time at a 10% gap | under 3 s | under 0.1 s |

It keeps the features that make the real instance interesting: three trailer
size classes with different daily trip and distance allowances, availability
factors spanning the 20 to 80 percent band, an 80 percent overfill buffer on
nominal station storage, and one remote station whose per-product tanks are
far smaller than a large trailer's load. That last case is what drives the
distance-per-litre outlier discussed in Section V-C of the paper, and it is
reproduced here as Station_5.

No figure in this instance comes from the real network. The values were
chosen to sit in the same ranges as Table II of the paper.

## Files

`truck_config_synthetic.csv` gives trailer capacity, size class, daily trip
and distance caps, and availability. `station_demand_synthetic.csv` gives
monthly demand, buffered storage, one-way distance and the estimated refill
count per station-product pair. Both use the same column names as the real
input tables, so they are drop-in replacements.

## Running

From `code/python`:

```bash
pip install -r requirements.txt
python run_synthetic.py
```

The script reads both tables from this directory.

Expected output: 7,329.20 km over 95 trips with zero overfill, and all five
trailers day-feasible.
