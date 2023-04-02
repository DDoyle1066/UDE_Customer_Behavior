using ModelingToolkit, StochasticDiffEq
using Plots
using CSV: CSV
using DataFrames
using CategoricalArrays
using DataConvenience
using Random
using SciMLBase
## M5-Forecasting Example
sales_data = CSV.read("data/sales_train_evaluation.csv", DataFrame)
sell_prices = CSV.read("data/sell_prices.csv", DataFrame)
calendar = CSV.read("data/calendar.csv", DataFrame)
transform!(sales_data, :dept_id => CategoricalArray => :dept_id)
transform!(sales_data, :cat_id => CategoricalArray => :cat_id)
transform!(sales_data, :store_id => CategoricalArray => :store_id)
transform!(sales_data, :state_id => CategoricalArray => :state_id)
transform!(sales_data, :item_id => CategoricalArray => :item_id)
transform!(sales_data, :dept_id => (x -> levelcode.(x)) => :dept_id_num)
transform!(sales_data, :cat_id => (x -> levelcode.(x)) => :cat_id_num)
transform!(sales_data, :store_id => (x -> levelcode.(x)) => :store_id_num)
transform!(sales_data, :state_id => (x -> levelcode.(x)) => :state_id_num)
transform!(sales_data, :item_id => (x -> levelcode.(x)) => :item_id_num)
function process_sales(sales, day_start, day_end)
    included_cols = [:id, :item_id, :dept_id, :cat_id, :store_id, :state_id]
    day_cols = Symbol.("d_" .* string.(day_start:day_end))
    long_sales = stack(sales[:, vcat(included_cols..., day_cols...)],
                       day_cols)
    leftjoin!(long_sales,
              calendar[:, Not([:date, :weekday])];
              on=[:variable => :d])
    leftjoin!(long_sales, sell_prices; on=[:wm_yr_wk, :store_id, :item_id])

    return long_sales
end
processed_sales = process_sales(sales_data, 1, 50)

#=
To-Do:
 - Create JLD2 objects that can be interacted with by the model 
 - Build elasticcity model for a single item in a single store
 - Expand to multiple items in the same dept in a single store
 - Exapnd to all items in a single store
Move across stores
=#

## Hypothetical mortatlity Example
#=
To-Do:
- Dynamics of constant hazards
 - Environmental
 - Communicable Diseases
- Dyanmics of age increasing hazards
- Effect of location on constant hazards
- Effect of wealth on constant and age increasing hazards
- Effect of population on communicable diseases
=#

@parameters μ_env Θ_region σ_region
pop_size = 10#_000
num_regions = 4
@variables t
@variables ages_pop(t)[1:pop_size] cum_mort_pop(t)[1:pop_size]
@variables health_region(t)[1:num_regions,
                            1:num_regions]
D = Differential(t)
# Equations for how age rolls forward
age_eqs = D.(ages_pop) .~ 1
mort_eqs = D.(cum_mort_pop) .~ μ_env
age_noise_eqs = Symbolics.scalarize(0 .* ages_pop)
# Equations for how health propagates from region to region
region_eqs = Matrix{Equation}(undef, num_regions, num_regions)
for i in 1:num_regions
    for j in 1:num_regions
        avg_surrounging_health = 0
        num_neighbors = 0
        if i > 1
            avg_surrounging_health += health_region[i - 1, j]
            num_neighbors += 1
        end
        if i < num_regions
            avg_surrounging_health += health_region[i + 1, j]
            num_neighbors += 1
        end
        if j > 1
            avg_surrounging_health += health_region[i, j - 1]
            num_neighbors += 1
        end
        if j < num_regions
            avg_surrounging_health += health_region[i, j + 1]
            num_neighbors += 1
        end
        region_eqs[i, j] = D(health_region[i, j]) ~ Θ_region * (avg_surrounging_health /
                                                                num_neighbors -
                                                                health_region[i, j]) -
                                                    σ_region^2 / 2
    end
end
region_noise_eqs = Symbolics.scalarize(health_region .* σ_region)
eqs = [Symbolics.scalarize(age_eqs)...,
       Symbolics.scalarize(mort_eqs)...,
       Symbolics.scalarize(region_eqs)...]
noiseeqs = [region_noise_eqs...]
# noiseeqs = [0.01σ]
@named de = SDESystem(eqs, noiseeqs, t,
                      [ages_pop..., cum_mort_pop..., health_region...],
                      [μ_env, Θ_region, σ_region]; tspan=(0, 10.0))

# u0map = [health => 1.0,
#          pop => repeat([1.0], length(pop))]
rng = MersenneTwister(20230326)
init_ages = rand(rng, pop_size) .* 60
init_health = exp.(randn(rng, num_regions, num_regions) / 10)
u0map = [Symbolics.scalarize(ages_pop .=> init_ages)...,
         Symbolics.scalarize(cum_mort_pop .=> 0)...,
         Symbolics.scalarize(health_region .=> init_health)...]
parammap = [μ_env => 1e-4,
            Θ_region => 0.2,
            σ_region => 0.1]

prob = SDEProblem(de, u0map, (0.0, 10.0), parammap)
@time sol = solve(prob, SOSRI())
ages_indices = zeros(Int, length(ages_pop))
syms = SciMLBase.getsyms(sol)
for i in 1:length(ages_pop)
    ages_indices[i] = SciMLBase.sym_to_index(ages_pop[i], syms)
end
mort_indices = zeros(Int, length(cum_mort_pop))
for i in 1:length(cum_mort_pop)
    mort_indices[i] = SciMLBase.sym_to_index(cum_mort_pop[i], syms)
end
health_indices = zeros(Int, size(health_region))
for i in 1:num_regions
    for j in 1:num_regions
        health_indices[i, j] = SciMLBase.sym_to_index(health_region[i, j], syms)
    end
end
sol(10; idxs=health_indices)
# example for julia lang
@variables t1 x(t1)[1:10]
@parameters μ σ
Dx = Differential(t1)
x_eq = Symbolics.scalarize(Dx.(x) .~ 1)
x_noise_eq = Symbolics.scalarize(x .* σ)
@named sde_sys = SDESystem(x_eq,
                           x_noise_eq, t1,
                           [x...],
                           [μ, σ]; tspan=(0, 10.0))
sde_prob = SDEProblem(sde_sys, Symbolics.scalarize(x .=> 1), (0.0, 10.0),
                      [μ => 1, σ => 0.2])
sde_sol = solve(sde_prob, SOSRI())
indices = []
for i in 1:2
    push!(indices, SciMLBase.sym_to_index(x[i], SciMLBase.getsyms(sde_sol)))
end
sde_sol(0.5; idxs=indices)
plot(sde_sol; idxs=x[1:2])
plot(sde_sol)
@named ode_sys = ODESystem([Symbolics.scalarize(x_eq)...], t1,
                           [x...],
                           [μ, σ]; tspan=(0, 10.0))
plot(sol; idxs=ages_pop[1:2])
plot(sol; idxs=[health, pop])
@syms (ages_pop(t))[1](var, var2)
sol.du[1]