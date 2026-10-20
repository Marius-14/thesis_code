"""Historical baseline for the comparison in Section V-A.

Distance is computed from the anonymised delivery records the same way the
model's input tables are: deliveries from the principal depot (Source_2),
round-trip distance, mean per delivery day scaled to 30 days.

The records list one line per product delivered. The fleet's trailers are
multi-compartment, so a trailer that visits a station once and discharges
three grades appears as three lines. ``physical_trips`` counts that visit as
one trip, which is what the trailer actually drove; ``product_lines`` counts
every line as its own trip, which is how the earlier comparison notebook
(code/comparison_model_vs_actual.ipynb) produced 74,275 km. The paper uses
the physical figure, because fuel is burned per kilometre driven.

Evidence that same-day lines are single visits: in 97% of multi-line groups
every line is a different grade, and in 92% the combined volume fits in one
tank load.
"""

from __future__ import annotations

import os

import pandas as pd

import milp

RAW = os.path.join(milp.REPO, "data", "raw", "fuel_data_raw_anon.csv")
DEPOT = "Source_2"
VISIT = ["Trailer", "Destination_ID", "Delivery_Date"]


def load_records(path=RAW, depot=DEPOT):
    df = pd.read_csv(path, low_memory=False)
    df = df[df.Origin_ID == depot].copy()
    df["Delivery_Date"] = pd.to_datetime(df.Delivery_Date)
    df["YearMonth"] = df.Delivery_Date.dt.to_period("M")
    df["RoundTrip_km"] = df.Distance_km * 2
    return df


def visits(df):
    """One row per trailer visit to a station on a day."""
    return (df.groupby(VISIT)
              .agg(lines=("Product", "size"),
                   grades=("Product", "nunique"),
                   litres=("Delivered_Volume_Liters", "sum"),
                   capacity=("Truck_Tank_Capacity_Liters", "first"),
                   RoundTrip_km=("RoundTrip_km", "first"))
              .reset_index())


def monthly_km(rows):
    """Mean round-trip km per delivery day, scaled to 30 days."""
    return rows.groupby("Delivery_Date").RoundTrip_km.sum().mean() * 30


def summary(path=RAW):
    df = load_records(path)
    v = visits(df)
    v["YearMonth"] = v.Delivery_Date.dt.to_period("M")
    multi = v[v.lines > 1]
    per_trailer = (v.groupby(["Trailer", "YearMonth"])
                     .agg(trips=("RoundTrip_km", "size"),
                          km=("RoundTrip_km", "sum"))
                     .groupby("Trailer").mean())
    busiest = per_trailer.km.idxmax()
    return dict(
        first=df.Delivery_Date.min().date(), last=df.Delivery_Date.max().date(),
        months=df.YearMonth.nunique(),
        lines=len(df), visits=len(v),
        share_lines_in_multi=v.lines[v.lines > 1].sum() / len(df),
        multi_distinct_grades=(multi.lines == multi.grades).mean(),
        multi_fits_one_load=(multi.litres <= multi.capacity * 1.001).mean(),
        product_lines=monthly_km(df),
        physical_trips=monthly_km(v),
        trips_per_month=v.groupby("YearMonth").size().mean(),
        litres_per_30_days=df.Delivered_Volume_Liters.sum()
                           / df.Delivery_Date.nunique() * 30,
        load_utilization=v.litres.sum() / v.capacity.sum(),
        busiest_trailer=busiest,
        busiest_trips=per_trailer.loc[busiest, "trips"],
        busiest_km=per_trailer.loc[busiest, "km"],
    )
