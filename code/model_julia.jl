# Trailer-allocation MILP for multi-product fuel distribution.
#
# Reference implementation of the model in Section III-C of the paper. The
# Python implementation in code/python/milp.py builds the same formulation
# through the CPLEX LP file format; the two agree within the 10% MIP gap.
#
#   min  sum_t Delta_t + lambda * Delta_max
#   s.t. (2)  sum_t x[i,t] >= Q[i]
#        (3)  x[i,t] <= y[i,t] * min(C[t], V[i])
#        (4)  sum_i y[i,t] <= M[t] * N[t]
#        (5)  sum_i y[i,t] * d[i] <= rho * L[t] * N[t]
#        (6)  Delta_t = sum_i y[i,t] * d[i]
#        (7)  Delta_t <= Delta_max
#        (8)  sum_t x[i,t] <= R[i]
#        (9)  sum_{i : d[i] > L[t]/(k+1)} y[i,t] <= k * N[t],  k = 1..4
#
# N[t] = floor(30 * A[t]) is the number of whole operating days available to
# trailer t. Constraints (4), (5) and (9) are what make the monthly plan
# decompose into a legal daily schedule; code/python/day_feasibility.py
# certifies that constructively.

# Import libraries/packages
using JuMP
using CSV
using DataFrames
using HiGHS
using FilePathsBase

const ROOT = dirname(dirname(@__FILE__))

const DAYS   = 30      # planning horizon, D
const LAMBDA = 0.5     # imbalance weight
const RHO    = 0.95    # daily-distance buffer
const KMAX   = 4       # largest k in the daily-packing inequalities

# === Load Datasets ===
truck_df = CSV.read(joinpath(ROOT,"data/processed/truck_config_julia.csv"), DataFrame)
demand_df = CSV.read(joinpath(ROOT,"data/processed/station_demand_julia.csv"), DataFrame)

# === Sets ===
T = truck_df.Trailer
I = [(row.Destination_ID, row.Product) for row in eachrow(demand_df)]  # Tuple set: (station, product)

# === Parameters ===
A = Dict(truck_df.Trailer .=> truck_df.Availability_Percent ./ 100)
C = Dict(truck_df.Trailer .=> truck_df.Truck_Tank_Capacity_Liters)
L = Dict(truck_df.Trailer .=> truck_df.Distance_Limit_Per_Day)
M = Dict(truck_df.Trailer .=> truck_df.Max_Trips_Per_Day)

# Filter T to 20-80% availability
T = [t for t in T if 0.20 ≤ A[t] ≤ 0.80]

# Whole operating days available to each trailer: a trailer cannot work a
# fraction of a day, so the fractional D * A[t] is floored (Section III-C).
N = Dict(t => floor(Int, DAYS * A[t]) for t in T)

# Dicts indexed by station or (station, product)
demand = Dict((row.Destination_ID, row.Product) => row.Monthly_Demand_Liters for row in eachrow(demand_df))
distance = Dict(row.Destination_ID => row.Distance_km * 2 for row in eachrow(demand_df))  # round-trip
capacity_limit = Dict((row.Destination_ID, row.Product) => row.Product_Storage_Capacity_Liters for row in eachrow(demand_df))
estimated_refills = Dict((row.Destination_ID, row.Product) => row.Estimated_Refills for row in eachrow(demand_df))
refill_limit = Dict(i => max(demand[i], estimated_refills[i] * capacity_limit[i]) for i in I)

# === Model ===
model = Model(HiGHS.Optimizer)
set_optimizer_attribute(model, "mip_rel_gap", 0.1)

# === Variable ===
@variable(model, y[i in I, t in T] >= 0, Int)      # Number of trips
@variable(model, x[i in I, t in T] >= 0)           # Liters delivered
@variable(model, truck_usage[t in T] >= 0)         # Total trips per truck
@variable(model, trailer_distance[t in T] >= 0)    # Total distance per truck
@variable(model, max_trailer_distance >= 0)        # Max distance by any truck

# === Constraints ===
@constraint(model, [i in I], sum(x[i, t] for t in T) >= demand[i])                               # (2) Meet station demand
@constraint(model, [i in I, t in T], x[i, t] <= y[i, t] * min(C[t], capacity_limit[i]))          # (3) Liters delivered per trip
@constraint(model, [t in T], truck_usage[t] == sum(y[i, t] for i in I))                          #     Total trips per truck
@constraint(model, [t in T], trailer_distance[t] == sum(y[i, t] * distance[i[1]] for i in I))    # (6) Total distance per truck
@constraint(model, [t in T], trailer_distance[t] <= max_trailer_distance)                        # (7) Max distance by any truck
@constraint(model, [i in I], sum(x[i, t] for t in T) <= refill_limit[i])                         # (8) Refill limits respected

# Truck trip and distance constraints over whole operating days, with the
# buffer that absorbs residual packing waste
for t in T
    @constraint(model, sum(y[i, t] for i in I) <= M[t] * N[t])                                   # (4)
    @constraint(model, sum(y[i, t] * distance[i[1]] for i in I) <= RHO * L[t] * N[t])            # (5)
end

# Daily-packing inequalities: a round trip longer than L[t]/(k+1) cannot be
# performed more than k times in one day, so at most k * N[t] fit in the horizon
for t in T
    for k in 1:min(KMAX, Int(M[t]))
        threshold = L[t] / (k + 1)
        long_i = [i for i in I if distance[i[1]] > threshold + 1e-9]
        if !isempty(long_i)
            @constraint(model, sum(y[i, t] for i in long_i) <= k * N[t])                         # (9)
        end
    end
end

# === Objective ===
@objective(model, Min,
    sum(trailer_distance[t] for t in T) +
    LAMBDA * max_trailer_distance
)

# === Solve ===
optimize!(model)

# === Output ===
status = termination_status(model)

if status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT
    println("\nObjective Value: ", objective_value(model))
    println("Total distance:  ", sum(value(trailer_distance[t]) for t in T))
    println("Busiest trailer: ", value(max_trailer_distance))
    println("Trailers used:   ", count(t -> value(truck_usage[t]) > 0.5, T))

    detailed_output = DataFrame(
        Trailer = String[],
        Destination_ID = String[],
        Product = String[],
        Liters_Delivered = Float64[],
        Trips = Float64[],
        Distance_Travelled_km = Float64[]
    )

    for i in I, t in T
        trips = value(y[i, t])
        if trips > 0.5
            lit = value(x[i, t])
            dist = distance[i[1]] * trips
            push!(detailed_output, (t, i[1], i[2], lit, trips, dist))
            println("Truck ", t, " delivers ", round(lit, digits=2), " L of ", i[2], " to station ", i[1],
                    " over ", round(dist, digits=2), " km (", round(trips, digits=2), " trips)")
        end
    end

    #CSV.write(joinpath(ROOT, "data/processed/model_output.csv"), detailed_output)

    for i in I
        delivered = sum(value(x[i, t]) for t in T)
        if delivered > refill_limit[i] + 1e-6
            println("Overfilled: ", i, " delivered = ", delivered, " > limit = ", refill_limit[i])
        end
    end

else
    println("Model status: ", status)
    println("No feasible solution found.")
end
