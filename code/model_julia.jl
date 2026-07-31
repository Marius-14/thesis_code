# Import libraries/packages
using JuMP
using CSV
using DataFrames
using HiGHS
using FilePathsBase

const ROOT = dirname(dirname(@__FILE__))

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

# Filter T to 20–80% availability
T = [t for t in T if 0.20 ≤ A[t] ≤ 0.80]

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
@variable(model, overfill[i in I] >= 0)            # Overfill amount for each station

# === Constraints ===
@constraint(model, [i in I], sum(x[i, t] for t in T) >= demand[i])                               # Meet station demand
@constraint(model, [i in I, t in T], x[i, t] <= y[i, t] * min(C[t], capacity_limit[i]))          # Liters delivered per trip
@constraint(model, [t in T], truck_usage[t] == sum(y[i, t] for i in I))                          # Total trips per truck
@constraint(model, [t in T], trailer_distance[t] == sum(y[i, t] * distance[i[1]] for i in I))    # Total distance per truck
@constraint(model, [t in T], trailer_distance[t] <= max_trailer_distance)                        # Max distance by any truck
@constraint(model, [i in I], sum(x[i, t] for t in T) <= refill_limit[i] + overfill[i])           # Ensure refill limits are respected

# Truck trip and distance constraints (scaled by availability)
for t in T
    availability = A[t]
    DAYS = 30
    max_trips = M[t] * DAYS * availability
    max_distance = L[t] * DAYS * availability

    @constraint(model, sum(y[i, t] for i in I) <= max_trips)
    @constraint(model, sum(y[i, t] * distance[i[1]] for i in I) <= max_distance)
end

# === Objective ===
@objective(model, Min,
    sum(trailer_distance[t] for t in T) +
    0.1 * max_trailer_distance +
    20.0 * sum(overfill[i] for i in I)
)

# === Solve ===
optimize!(model)

# === Output ===
status = termination_status(model)

if status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT
    println("\nObjective Value (Total Distance): ", objective_value(model))

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

    #CSV.write(rel("data/processed/model_output.csv"), detailed_output)

    for i in I
        delivered = sum(value(x[i, t]) for t in T)
        if delivered > refill_limit[i]
            println("Overfilled: ", i, " delivered = ", delivered, " > limit = ", refill_limit[i])
        end
    end


else
    println("Model status: ", status)
    println("No feasible solution found.")
end