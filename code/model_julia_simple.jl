# Import libraries/packages
using JuMP
using CSV
using DataFrames
using HiGHS
using FilePathsBase

const ROOT = dirname(dirname(@__FILE__))

# === Load Datasets ===
trailer_config = CSV.read(joinpath(ROOT,"data/processed/truck_config_julia.csv"), DataFrame)
station_demand = CSV.read(joinpath(ROOT,"data/processed/station_demand_julia.csv"), DataFrame)

# === Parameters ===
T = 30  # total days
M = size(station_demand, 1)  # number of station-product pairs
N = size(trailer_config, 1)  # number of trailers

Vi = station_demand.Product_Storage_Capacity_Liters              # Storage capacity per station-product
ri = station_demand.Monthly_Demand_Liters ./ T                   # Daily demand per station-product
di = station_demand.Distance_km                                  # One-way distance to station
Cj = trailer_config.Truck_Tank_Capacity_Liters                   # Trailer capacity
Dj = trailer_config.Distance_Limit_Per_Day                       # Per-trailer distance limit

# === Initial fuel levels (30% of capacity) ===
y0 = 0.2 .* Vi

# === Create Model ===
model = Model(HiGHS.Optimizer)

@variable(model, x[1:M, 1:N, 1:T] >= 0)   # amount of trailer loads delivered (can be fractional)
@variable(model, y[1:M, 0:T])            # fuel level in storage

# === Initial tank levels ===
for i in 1:M
    fix(y[i, 0], y0[i]; force = true)
end

# === Tank dynamics: y(i,t) = y(i,t-1) + fuel delivered - consumed ===
@constraint(model, [i=1:M, t=1:T],
    y[i, t] == y[i, t-1] + sum(x[i, j, t] * Cj[j] for j in 1:N) - ri[i])

# === Storage safety bounds (10% to 90%) ===
@constraint(model, [i=1:M, t=1:T], y[i, t] >= 0.1 * Vi[i])
@constraint(model, [i=1:M, t=1:T], y[i, t] <= 0.9 * Vi[i])

# === Trailer distance cap per day ===
@constraint(model, [j=1:N, t=1:T],
    sum(x[i, j, t] * di[i] for i in 1:M) <= Dj[j])

# === Objective: minimize total distance across all partial deliveries ===
@objective(model, Min,
    sum(x[i, j, t] * di[i] for i in 1:M, j in 1:N, t in 1:T))

# === Solve ===
optimize!(model)

# === Output ===
status = termination_status(model)
println("Termination status: ", status)

if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    println("Objective value (total distance): ", objective_value(model))
    println("Feasibility check: Any deliveries? ", sum(value.(x)))
else
    println("Model did not solve successfully.")
end
